"""Real-PDF task-control acceptance; synchronization/fault hooks live only here.

Run api, worker, and verify against the same --root. The worker uses the real
pipeline with one task/chunk at a time; only the retry case injects a failure.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PDF_ROOT = Path('/data/patents')
SAMPLES = {
    'pause': PDF_ROOT/'chemrex_batch/api/uploads/6eff27f9dd7b4f45a29391cabf8e4cfb/acscatal.8b02844.pdf',
    'cancel': PDF_ROOT/'chemrex_batch/api/uploads/53c22e82913a484bbf830b07207e2443/US8426586.pdf',
    'delete': PDF_ROOT/'chemrex_batch/api/uploads/fc1e0bace8af4749b354d5e8f7dd438f/anie201805732.pdf',
    'retry': PDF_ROOT/'FDA_RLD/fda_server_bundle/output/pdf/ACETAMINOPHEN; IBUPROFEN SODIUM/US11446266.pdf',
}


def worker(root):
    from batch import config
    from batch.cli import main, build_parser, apply_llm_env
    argv = ['--db', str(root/'tasks.db'), 'run', '--daemon', '--task-workers','1',
            '--chunk-workers','1', '--poll-interval','0.2', '--no-auto-retry']
    # Match the production CLI's provider setup before importing pipeline.config.
    apply_llm_env(build_parser().parse_args(argv))
    from batch.scheduler import BatchScheduler
    from batch.document_processor import DocumentProcessor
    from batch.mineru_client import MineruClient
    from batch.image_client import ImageWorkerClient
    config.BATCH_LOG_DIR = root/'logs'
    config.BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ['BATCH_DB_PATH'] = str(root/'tasks.db')
    os.environ['CHEMREX_REUSE_TASK_DATABASES'] = ''
    os.environ.pop('CHEMREX_REUSE_RESULTS_ROOT', None)
    os.environ['CHEMREX_CALL_METRICS'] = str(root/'logs/model-calls.jsonl')
    local = threading.local()
    event_lock = threading.Lock()
    def event(kind, task, **extra):
        with event_lock, (root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), kind=kind, task=task, **extra))+'\n')
    def hold(name):
        (root/name).touch()
        deadline = time.monotonic()+120
        while not (root/(name+'.release')).exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Acceptance driver did not release '+name)
            time.sleep(.05)
    for stage in ('parse', 'mining', 'extracting'):
        original = getattr(BatchScheduler, '_stage_'+stage)
        def wrap(self, task, *args, _stage=stage, _original=original):
            local.task = task['uuid']
            event(_stage+'_start', task['uuid'])
            result = _original(self, task, *args)
            event(_stage+'_done', task['uuid'])
            return result
        setattr(BatchScheduler, '_stage_'+stage, wrap)
    for cls, method, kind in [(MineruClient, 'parse', 'parser_request'),
                              (ImageWorkerClient, 'process', 'image_request')]:
        original = getattr(cls, method)
        def called(self, *args, _kind=kind, _original=original, **kwargs):
            event(_kind, getattr(local, 'task', 'unknown'))
            return _original(self, *args, **kwargs)
        setattr(cls, method, called)
    original_extract = DocumentProcessor.extract
    def extract(self, prepared, output_dir, *args, **kwargs):
        local.task = Path(output_dir).name
        return original_extract(self, prepared, output_dir, *args, **kwargs)
    DocumentProcessor.extract = extract
    original_append = DocumentProcessor._append
    def append(stream, result):
        original_append(stream, result)
        uid = Path(stream.name).parent.name
        event('chunk_saved', uid, chunk=result['id'], status=result['status'])
        if uid == 'api_acceptance_pause' and not (root/'pause-held').exists():
            hold('pause-held')
        if uid == 'api_acceptance_retry' and not (root/'failure-injected').exists():
            (root/'failure-injected').touch()
            event('failure_injected', uid)
            raise RuntimeError('Acceptance-only failure after first durable real chunk')
    DocumentProcessor._append = staticmethod(append)
    return main(argv)


def verify(root, port):
    import requests
    session = requests.Session()
    base = f'http://127.0.0.1:{port}'
    history = []
    def request(method, path, **kwargs):
        response = session.request(method, base+path, timeout=30, **kwargs)
        response.raise_for_status()
        body = response.json() if response.content else None
        history.append(dict(time=time.time(), method=method, path=path, http=response.status_code,
                            status=(body or {}).get('status'), finished=(body or {}).get('finished')))
        return response, body
    def poll(uid, wanted=None, predicate=None, timeout=3600):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            r = session.get(base+f'/v1/tasks/{uid}', timeout=30)
            if wanted == 'ABSENT' and r.status_code == 404:
                return None
            r.raise_for_status()
            body = r.json()
            history.append(dict(time=time.time(), task=uid, status=body['status'], stage=body['stage'], chunks_done=body['chunks_done']))
            if body['status'] == wanted or predicate and predicate(body):
                return body
            if body['status'] == 'FAILED' and wanted != 'FAILED':
                raise RuntimeError(json.dumps(body))
            time.sleep(.1 if predicate else .5)
        raise TimeoutError((uid, wanted))
    def submit(case):
        with SAMPLES[case].open('rb') as stream:
            _, body = request('POST','/v1/pdf?wait=false', headers={'Idempotency-Key':'acceptance_'+case},
                              files={'file':(SAMPLES[case].name,stream,'application/pdf')}, data={'language':'en'})
        return body['task_id']
    def artifacts(uid, allow_empty=False):
        _, body = request('GET',f'/v1/tasks/{uid}/result')
        result = body['result']
        for kind in ['reactions','routes']:
            assert result['downloads'][kind], (uid,kind)
            r = session.get(result['downloads'][kind],timeout=30)
            r.raise_for_status()
            rows = [json.loads(line) for line in r.text.splitlines()]
            assert rows == result[kind]
        assert result['document']['status'] != 'partial'
        assert result['delivery_manifest']['processing_status'] == 'complete'
        assert result['reactions'] or allow_empty, uid
        assert all(r.get('source') for r in result['reactions'])
        assert result['reaction_sources'] or allow_empty
        return dict(reactions=len(result['reactions']), routes=len(result['routes']),
                    delivery_status=result['delivery_manifest']['processing_status'])
    report = dict(samples={k:dict(pdf=str(v),bytes=v.stat().st_size) for k,v in SAMPLES.items()}, cases={})
    try:
        uid = submit('pause')
        poll(uid,predicate=lambda _: (root/'pause-held').exists())
        _, state = request('POST',f'/v1/tasks/{uid}/pause')
        assert state['status'] == 'PAUSING'
        (root/'pause-held.release').touch()
        poll(uid,'PAUSED')
        saved = (root/'results'/uid/'results.jsonl').read_text().splitlines()
        assert len(saved) == 1
        r, body = request('GET',f'/v1/tasks/{uid}/result?wait=true')
        assert r.status_code == 202 and body['result']['document']['status'] == 'partial'
        assert body['result']['delivery_manifest']['processing_status'] == 'partial'
        r = session.get(body['result']['downloads']['reactions'],timeout=30)
        assert r.status_code == 200
        request('POST',f'/v1/tasks/{uid}/resume')
        poll(uid,'DONE')
        final = (root/'results'/uid/'results.jsonl').read_text().splitlines()
        assert final[:1] == saved and len({json.loads(r)['id'] for r in final}) == len(final)
        report['cases']['pause'] = artifacts(uid)

        uid = submit('cancel')
        poll(uid,predicate=lambda s:s['status']=='RUNNING' and s['stage']=='parsing')
        request('POST',f'/v1/tasks/{uid}/cancel')
        poll(uid,'CANCELLED')
        assert (root/'results'/uid/'pdf_prepared/parse_result.json').exists()
        request('POST',f'/v1/tasks/{uid}/resume')
        poll(uid,'DONE')
        report['cases']['cancel'] = artifacts(uid)

        uid = submit('delete')
        poll(uid,predicate=lambda s:s['status']=='RUNNING' and s['stage']=='parsing')
        upload_dirs = list((root/'uploads').iterdir())
        request('DELETE',f'/v1/tasks/{uid}')
        poll(uid,'ABSENT')
        assert not (root/'results'/uid).exists()
        assert session.get(base+f'/v1/tasks/{uid}/files/delivery/reactions.jsonl',timeout=30).status_code == 404
        assert session.delete(base+f'/v1/tasks/{uid}',timeout=30).status_code == 204
        assert len(list((root/'uploads').iterdir())) == len(upload_dirs)-1
        report['cases']['delete'] = dict(task_absent=True,files_absent=True)

        uid = submit('retry')
        state = poll(uid,'FAILED')
        assert not state['retry_pending'] and state['finished']
        saved = (root/'results'/uid/'results.jsonl').read_text().splitlines()
        assert len(saved) == 1 and (root/'failure-injected').exists()
        request('POST',f'/v1/tasks/{uid}/resume')
        poll(uid,'DONE')
        final = (root/'results'/uid/'results.jsonl').read_text().splitlines()
        assert final[:1] == saved and len(final)>len(saved)
        assert len({json.loads(r)['id'] for r in final}) == len(final)
        # This formulation patent legitimately has no synthetic reaction;
        # verify the complete empty delivery rather than inventing chemistry.
        report['cases']['retry'] = artifacts(uid, allow_empty=True)
        events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        for case in ['pause','cancel','retry']:
            observed = [e for e in events if e['task']=='api_acceptance_'+case]
            assert sum(e['kind']=='parser_request' for e in observed) == 1, (case, observed)
            saved_ids = [e['chunk'] for e in observed if e['kind']=='chunk_saved']
            assert len(saved_ids) == len(set(saved_ids)), (case, saved_ids)
            report['cases'][case]['events'] = observed
        report['passed'] = True
    finally:
        report['history'] = history
        (root/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['history','samples']},ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['api','worker','verify'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,default=9253)
    args = parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    os.environ['CHEMREX_API_DATA_ROOT'] = str(args.root)
    if args.mode == 'api':
        import uvicorn
        from batch.api import create_app
        uvicorn.run(create_app(), host='127.0.0.1',port=args.port)
    elif args.mode == 'worker':
        return worker(args.root)
    else:
        verify(args.root,args.port)


if __name__ == '__main__':
    main()
