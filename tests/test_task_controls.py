"""Task controls against SQLite, HTTP, and cooperative executor boundaries."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from batch.api import ApiConfig, create_app, cleanup_deleted, task_view
from batch.scheduler import BatchScheduler
from batch.store import TaskStore


@pytest.fixture
def api(tmp_path):
    config = ApiConfig(tmp_path, poll_interval=0.01)
    app = create_app(config)
    with TestClient(app) as client:
        yield client, app.state.store, config


def upload(client, key='controls'):
    response = client.post('/v1/pdf?wait=false', headers={'Idempotency-Key': key},
                           files={'file': ('sample.pdf', b'%PDF-1.4\nfixture\n%%EOF')})
    assert response.status_code == 202
    return response.json()['task_id']


@pytest.mark.parametrize('action,expected', [('pause','PAUSED'), ('cancel','CANCELLED')])
def test_queued_control_and_resume_are_idempotent(api, action, expected):
    client, store, _ = api
    uid = upload(client)
    for _ in range(2):
        response = client.post(f'/v1/tasks/{uid}/{action}')
        assert response.status_code == 200
        assert response.json()['status'] == expected
    assert store.claim_next('worker') is None
    assert store.requeue_failed_with_attempts(3) == 0
    for _ in range(2):
        response = client.post(f'/v1/tasks/{uid}/resume')
        assert response.status_code == 200
        assert response.json()['status'] == 'PENDING'
    claimed = store.claim_next('worker')
    assert claimed['uuid'] == uid and claimed['attempt_count'] == 1
    assert store.claim_next('worker') is None
    assert client.post(f'/v1/tasks/{uid}/resume').status_code == 409


def test_controls_escalate_and_old_completion_cannot_win(api):
    client, store, _ = api
    uid = upload(client)
    task = store.claim_next('worker')
    assert client.post(f'/v1/tasks/{uid}/pause').status_code == 202
    assert client.post(f'/v1/tasks/{uid}/cancel').json()['status'] == 'CANCELLING'
    assert client.post(f'/v1/tasks/{uid}/pause').status_code == 409
    assert client.post(f'/v1/tasks/{uid}/resume').status_code == 409
    assert client.delete(f'/v1/tasks/{uid}').status_code == 202
    assert client.post(f'/v1/tasks/{uid}/cancel').status_code == 409
    assert not store.mark_done(task['id'], 1, run_id='worker')
    assert not store.mark_failed(task['id'], 'late', run_id='worker', retry_pending=True)
    assert Path(task['pdf_path']).is_file()
    store.settle_stop(task['id'], 'wrong-worker')
    assert store.get(uid)['execution_active']
    store.settle_stop(task['id'], 'worker')
    cleanup_deleted(store, api[2], uid)
    assert client.get(f'/v1/tasks/{uid}').status_code == 404
    assert client.delete(f'/v1/tasks/{uid}').status_code == 204
    assert not Path(task['pdf_path']).exists()
    assert upload(client) == uid


def test_paused_wait_returns_partial_results_and_downloads(api):
    client, store, _ = api
    uid = upload(client)
    root = Path(store.get(uid)['output_dir'])
    (root/'delivery').mkdir(parents=True)
    (root/'delivery/reactions.jsonl').write_text('')
    client.post(f'/v1/tasks/{uid}/pause')
    response = client.get(f'/v1/tasks/{uid}/result?wait=true')
    assert response.status_code == 202
    assert not response.json()['finished']
    assert response.json()['result']['reactions'] == []
    assert client.get(f'/v1/tasks/{uid}/files/delivery/reactions.jsonl').status_code == 200


def test_no_auto_retry_is_terminal_and_manual_retry_resets_budget(api):
    client, store, _ = api
    uid = upload(client)
    task = store.claim_next('worker')
    store.mark_failed(task['id'], 'final failure', run_id='worker')
    assert task_view(store.get(uid))['finished']
    assert store.requeue_failed_with_attempts(3) == 0
    response = client.post(f'/v1/tasks/{uid}/resume')
    assert response.status_code == 200
    assert response.json()['attempts'] == 0
    assert response.json()['task_id'] == uid
    # Old callers still recover via the hidden alias; new integrations see one endpoint.
    alias = client.post(f'/v1/tasks/{uid}/retry')
    assert alias.status_code == 200 and alias.json()['status'] == 'PENDING'
    paths = client.get('/openapi.json').json()['paths']
    assert '/v1/tasks/{task_id}/resume' in paths
    assert '/v1/tasks/{task_id}/retry' not in paths


def test_missing_pdf_and_done_conflicts(api):
    client, store, _ = api
    uid = upload(client)
    client.post(f'/v1/tasks/{uid}/pause')
    Path(store.get(uid)['pdf_path']).unlink()
    assert client.post(f'/v1/tasks/{uid}/resume').status_code == 409
    other = upload(client, 'done')
    task = store.claim_next('worker')
    store.mark_done(task['id'], 1, run_id='worker')
    for action in ['pause', 'cancel', 'resume', 'retry']:
        assert client.post(f'/v1/tasks/{other}/{action}').status_code == 409


def test_cleanup_failure_preserves_record_and_never_follows_source(api, tmp_path):
    client, store, config = api
    uid = upload(client)
    task = store.get(uid)
    external = tmp_path/'history'
    external.mkdir()
    (external/'keep').write_text('shared')
    root = Path(task['output_dir'])
    root.parent.mkdir(parents=True)
    root.symlink_to(external, target_is_directory=True)
    store.request_stop(uid, 'delete')
    cleanup_deleted(store, config, uid)
    assert store.get(uid)['status'] == 'DELETING'
    assert 'Deletion failed' in store.get(uid)['error']
    assert (external/'keep').read_text() == 'shared'
    root.unlink()
    cleanup_deleted(store, config, uid)
    assert store.get(uid) is None
    assert (external/'keep').exists()


@pytest.mark.parametrize('action,expected', [('pause','PAUSED'), ('cancel','CANCELLED'), ('delete','DELETING')])
def test_running_parse_drains_and_skips_later_stages(api, action, expected):
    client, store, _ = api
    uid = upload(client)
    task = store.claim_next('worker')
    scheduler = object.__new__(BatchScheduler)
    scheduler.store = store
    scheduler.auto_retry = False
    entered, release = threading.Event(), threading.Event()
    def parse(task):
        entered.set()
        assert release.wait(5)
        Path(task['output_dir']).mkdir(parents=True)
        (Path(task['output_dir'])/'parsed').write_text('complete')
        return 'content.json', None, 'fingerprint'
    scheduler._stage_parse = parse
    scheduler._stage_mining = mock.Mock()
    scheduler._stage_extracting = mock.Mock()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(scheduler._process_task, task)
        assert entered.wait(5)
        store.request_stop(uid, action)
        assert store.get(uid)['execution_active']
        assert not future.done()
        release.set()
        future.result(timeout=5)
    row = store.get(uid)
    # The frontend cleaner may already have removed an acknowledged deletion.
    assert (row is None and action == 'delete') or row['status'] == expected
    if row:
        assert not row['execution_active']
    scheduler._stage_mining.assert_not_called()
    scheduler._stage_extracting.assert_not_called()


def test_control_recovery_does_not_restart_cancelled_or_paused(api):
    client, store, _ = api
    uid = upload(client)
    task = store.claim_next('crashed')
    store.request_stop(uid, 'pause')
    store.recover_controls('replacement')
    assert store.get(uid)['status'] == 'PAUSED'
    assert store.claim_next('replacement') is None
    assert store.requeue_failed_with_attempts(3) == 0


def test_two_tasks_stop_independently(api):
    client, store, _ = api
    a, b = upload(client, 'a'), upload(client, 'b')
    first, second = store.claim_next('worker'), store.claim_next('worker')
    store.request_stop(a, 'cancel')
    assert store.mark_done(second['id'], 1, run_id='worker')
    store.settle_stop(first['id'], 'worker')
    assert store.get(a)['status'] == 'CANCELLED'
    assert store.get(b)['status'] == 'DONE'


def test_processor_drains_bounded_chunks_and_resumes(tmp_path):
    from test_batch_document_processor import make_processor
    blocks = []
    for i in range(1, 7):
        blocks += [{'type': 'text', 'text': f'Example {i}: Compound {i}', 'page_idx': i},
                   {'type': 'text', 'text': 'A mixture was stirred at 25 C for 2 h.', 'page_idx': i}]
    content = tmp_path/'content.json'
    content.write_text(json.dumps(blocks))
    output = tmp_path/'out'
    processor = make_processor(chunk_workers=2, no_delivery=True)
    prepared = processor.prepare(str(content), output)
    assert len(prepared.chunks) == 6
    original = processor._run_one
    started = threading.Barrier(3)
    release, stop = threading.Event(), threading.Event()
    def blocked(*args):
        started.wait(timeout=5)
        assert release.wait(5)
        return original(*args)
    with mock.patch.object(processor, '_run_one', side_effect=blocked) as calls:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(processor.extract, prepared, output, {}, stop_requested=stop.is_set)
            started.wait(timeout=5)
            stop.set()
            release.set()
            summary = future.result(timeout=15)
        assert calls.call_count == 2
    rows = (output/'results.jsonl').read_text().splitlines()
    assert len(rows) == 2
    assert summary['extraction_outcome'] == 'partial'
    assert len(json.loads((output/'document_result.json').read_text())['missing_chunks']) == 4
    with mock.patch.object(processor, '_run_one', wraps=original) as remaining:
        summary = processor.extract(prepared, output, {}, retry_errors=True)
        assert remaining.call_count == 4
    assert len((output/'results.jsonl').read_text().splitlines()) == 6
    assert summary['extraction_outcome'] != 'partial'


def test_pause_during_auto_retry_backoff_does_not_restart(api):
    import time
    client, store, _ = api
    uid = upload(client)
    task = store.claim_next('worker')
    scheduler = object.__new__(BatchScheduler)
    scheduler.store = store
    scheduler.auto_retry = True
    scheduler.max_attempts = 3
    scheduler.retry_backoff = 30
    scheduler._stop = threading.Event()
    scheduler._stage_parse = mock.Mock(side_effect=RuntimeError('temporary failure'))
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(scheduler._process_task, task)
        deadline = time.monotonic()+5
        while not store.get(uid)['retry_pending']:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert client.post(f'/v1/tasks/{uid}/pause').status_code == 202
        future.result(timeout=5)
    assert store.get(uid)['status'] == 'PAUSED'
    assert not store.get(uid)['execution_active']
    assert store.claim_next('worker') is None


def test_migration_keeps_legacy_data_and_marks_running_as_active(tmp_path):
    import sqlite3
    from batch.store import _SCHEMA
    db = tmp_path/'legacy.db'
    legacy = _SCHEMA.replace('    execution_active INTEGER NOT NULL DEFAULT 0,\n','').replace('    retry_pending INTEGER NOT NULL DEFAULT 0,\n','')
    with sqlite3.connect(db) as connection:
        connection.executescript(legacy)
        connection.execute("INSERT INTO pdf_task (uuid,pdf_path,output_dir,status,created_at,updated_at) VALUES ('old','old.pdf','out','RUNNING',1,1)")
        connection.execute("INSERT INTO pdf_task (uuid,pdf_path,output_dir,status,attempt_count,created_at,updated_at) VALUES ('backoff','failed.pdf','out2','FAILED',1,1,1)")
    store = TaskStore(db)
    try:
        row = store.get('old')
        assert row['status'] == 'RUNNING' and row['execution_active'] == 1
        assert row['created_at'] == 1 and not row['retry_pending']
        assert store.get('backoff')['retry_pending'] == 1
        store.request_stop('old', 'pause')
        store.recover_controls('new-worker')
        assert store.get('old')['status'] == 'PAUSED'
    finally:
        store.close()


def test_reupload_during_deletion_conflicts_without_losing_original(api):
    client, store, _ = api
    uid = upload(client)
    task = store.claim_next('worker')
    store.request_stop(uid, 'delete')
    response = client.post('/v1/pdf?wait=false', headers={'Idempotency-Key':'controls'},
                           files={'file':('sample.pdf',b'%PDF-1.4\nfixture\n%%EOF')})
    assert response.status_code == 409
    assert store.get(uid)['id'] == task['id']
    assert Path(task['pdf_path']).is_file()
    assert len(list(api[2].data_root.joinpath('uploads').iterdir())) == 1


def test_paused_and_cancelled_survive_reopening_queue(api):
    client, store, config = api
    a, b = upload(client, 'persist-pause'), upload(client, 'persist-cancel')
    client.post(f'/v1/tasks/{a}/pause')
    client.post(f'/v1/tasks/{b}/cancel')
    reopened = TaskStore(config.db_path)
    try:
        reopened.recover_controls('new-run')
        reopened.recover_stuck(60, 3, current_run_id='new-run')
        reopened.requeue_failed_with_attempts(3)
        assert reopened.get(a)['status'] == 'PAUSED'
        assert reopened.get(b)['status'] == 'CANCELLED'
        assert reopened.claim_next('new-run') is None
    finally:
        reopened.close()


def test_stop_flushes_progress_even_inside_throttle_window(api):
    client, store, _ = api
    uid = upload(client)
    task = store.claim_next('worker')
    root = Path(task['output_dir'])
    root.mkdir(parents=True)
    (root/'summary.json').write_text(json.dumps({'result_status': {'ok': 1}}))
    store.update_stage(task['id'], 'extracting', chunks_total=3, run_id='worker')
    store.request_stop(uid, 'pause')
    scheduler = object.__new__(BatchScheduler)
    scheduler.store = store
    scheduler._finish_stop(task)
    assert store.get(uid)['chunks_done'] == 1
    assert store.get(uid)['status'] == 'PAUSED'
