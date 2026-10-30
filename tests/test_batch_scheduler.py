"""Integration tests for batch/scheduler.py — task lifecycle with faked services."""

import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from batch.scheduler import BatchScheduler
from batch.store import TaskStore


class FakeMineru:
    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.parse_calls = 0

    def wait_healthy(self, attempts=1):
        return {"status": "ok"}

    def parse(self, pdf_path, output_dir, backend, language="en",
              first_page=1, last_page=None, server_url=None):
        self.parse_calls += 1
        if self.parse_calls <= self.fail_times:
            raise RuntimeError("mineru exploded")
        return {"status": "ok", "content_list_file": str(Path(output_dir) / "cl.json"),
                "image_base": str(Path(output_dir) / "images"),
                "parse_dir": output_dir, "images": 3}


class FakeImageClient:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.stopped = False

    def ensure_started(self):
        pass

    def process(self, image_base, return_per_image=True, result_path=None):
        self.calls += 1
        return {"dict_result": {"1": "CCO"}, "image_result": [],
                "per_image_detections": []}

    def stop(self):
        self.stopped = True


class FakePrepared:
    def __init__(self):
        self.chunks = []
        self.matcher = None
        self.manifest = {"with_procedure": 2, "total_chunks": 2}


class FakeProcessor:
    calls = []

    def __init__(self, language="en", chunk_workers=1, skip_final_product=False,
                 global_dict_enabled=True, no_delivery=False, preset=None):
        self.language = language
        self.preset = preset
        FakeProcessor.calls.append(self)

    def prepare(self, content_list_file, output_dir):
        return FakePrepared()

    def extract(self, prepared, output_dir, image_data, retry_errors=False,
                progress_cb=None, stop_requested=None):
        self.image_data = image_data
        self.retry_errors = retry_errors
        path = Path(output_dir) / "results.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": "example_1", "status": "ok"}) + "\n")
        if progress_cb:
            progress_cb(1, 2)
        return {"result_status": {"ok": 1}}


class BatchSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "tasks.db"
        self.store = TaskStore(self.db)
        self.out_root = self.root / "out"
        FakeProcessor.calls = []
        # must stay active for the whole test: the scheduler resolves
        # DocumentProcessor lazily inside run()
        self._patcher = mock.patch("batch.scheduler.DocumentProcessor", FakeProcessor)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.store.close()
        self.tmp.cleanup()

    def _add(self, uuid="doc", language="en"):
        self.store.add_task(uuid, f"/pdfs/{uuid}.pdf",
                            str(self.out_root / uuid), language=language)

    def _scheduler(self, mineru, image_client, **kwargs):
        defaults = dict(task_workers=1, chunk_workers=2, poll_interval=0.05,
                        max_attempts=3, lock_expire_seconds=60, auto_retry=True,
                        retry_backoff=0)
        defaults.update(kwargs)
        return BatchScheduler(store=self.store, mineru=mineru,
                              image_client=image_client, **defaults)

    def test_happy_path_done(self):
        self._add()
        mineru = FakeMineru()
        image_client = FakeImageClient()
        scheduler = self._scheduler(mineru, image_client)
        claimed = scheduler.run()
        self.assertEqual(claimed, 1)
        task = self.store.get("doc")
        self.assertEqual(task["status"], "DONE")
        self.assertEqual(mineru.parse_calls, 1)
        self.assertEqual(image_client.calls, 1)
        out = self.out_root / "doc"
        self.assertTrue((out / "pdf_prepared" / "parse_result.json").is_file())
        self.assertTrue((out / "image_result.json").is_file())
        self.assertTrue((out / "results.jsonl").is_file())
        processor = FakeProcessor.calls[0]
        self.assertEqual(processor.image_data["dict_result"], {"1": "CCO"})
        self.assertFalse(processor.retry_errors)
        self.assertTrue(image_client.stopped)

    def test_parse_failure_auto_retries_then_succeeds(self):
        self._add()
        mineru = FakeMineru(fail_times=1)
        scheduler = self._scheduler(mineru, FakeImageClient())
        claimed = scheduler.run()
        self.assertEqual(claimed, 2)  # claimed twice (retry within the run)
        self.assertEqual(self.store.get("doc")["status"], "DONE")
        self.assertEqual(mineru.parse_calls, 2)
        # the retried task passes retry_errors=True to the processor
        self.assertTrue(FakeProcessor.calls[-1].retry_errors)

    def test_failure_without_auto_retry_marks_failed(self):
        self._add()
        mineru = FakeMineru(fail_times=99)
        scheduler = self._scheduler(mineru, FakeImageClient(), auto_retry=False)
        scheduler.run()
        task = self.store.get("doc")
        self.assertEqual(task["status"], "FAILED")
        self.assertIn("mineru exploded", task["error"])

    def test_attempts_exhausted_marks_failed(self):
        self._add()
        mineru = FakeMineru(fail_times=99)
        scheduler = self._scheduler(mineru, FakeImageClient(),
                                    auto_retry=True, max_attempts=2)
        scheduler.run()
        task = self.store.get("doc")
        self.assertEqual(task["status"], "FAILED")
        self.assertEqual(task["attempt_count"], 2)

    def test_resume_skips_parsing_and_mining(self):
        self._add()
        out = self.out_root / "doc"
        (out / "pdf_prepared").mkdir(parents=True)
        (out / "cl.json").write_text("[]", encoding="utf-8")
        (out / "images").mkdir()
        fingerprint = "md5:" + hashlib.md5(b"[]").hexdigest()
        (out / "pdf_prepared" / "parse_result.json").write_text(json.dumps({
            "content_list_file": str(out / "cl.json"),
            "image_base": str(out / "images"),
            "content_list_fingerprint": fingerprint}), encoding="utf-8")
        (out / "image_result.json").write_text(json.dumps(
            {"dict_result": {}, "image_result": [],
             "parse_fingerprint": fingerprint,
             "image_pipeline_version": __import__("pipeline.versions", fromlist=["IMAGE_PIPELINE_VERSION"]).IMAGE_PIPELINE_VERSION}), encoding="utf-8")
        mineru = FakeMineru()
        image_client = FakeImageClient()
        scheduler = self._scheduler(mineru, image_client)
        scheduler.run()
        self.assertEqual(mineru.parse_calls, 0)
        self.assertEqual(image_client.calls, 0)
        self.assertEqual(self.store.get("doc")["status"], "DONE")

    def test_no_image_skips_mining(self):
        self._add()
        mineru = FakeMineru()
        image_client = FakeImageClient()
        scheduler = self._scheduler(mineru, image_client, no_image=True)
        scheduler.run()
        self.assertEqual(image_client.calls, 0)
        self.assertFalse((self.out_root / "doc" / "image_result.json").exists())
        self.assertEqual(self.store.get("doc")["status"], "DONE")

    def test_recover_stuck_on_start(self):
        self._add()
        task = self.store.claim_next()
        self.store._execute("UPDATE pdf_task SET locked_at=? WHERE id=?",
                            (0.0, task["id"]))
        scheduler = self._scheduler(FakeMineru(), FakeImageClient())
        scheduler.run()
        # the stuck RUNNING task was requeued and processed
        self.assertEqual(self.store.get("doc")["status"], "DONE")

    def test_two_documents_concurrently(self):
        self._add("doc1")
        self._add("doc2")
        mineru = FakeMineru()
        scheduler = self._scheduler(mineru, FakeImageClient(), task_workers=2)
        claimed = scheduler.run()
        self.assertEqual(claimed, 2)
        self.assertEqual(self.store.counts()["DONE"], 2)

    def test_recover_fresh_lock_from_previous_run(self):
        # H2: a crashed run left RUNNING with a fresh lock (no expiry) —
        # the new run must requeue it immediately via the run_id mismatch.
        self._add()
        self.store.claim_next(run_id="crashed-run")
        scheduler = self._scheduler(FakeMineru(), FakeImageClient())
        claimed = scheduler.run()
        self.assertEqual(claimed, 1)
        self.assertEqual(self.store.get("doc")["status"], "DONE")

    def test_stale_parse_marker_triggers_reparse(self):
        # M3: marker exists but points at a deleted content list
        self._add()
        out = self.out_root / "doc"
        (out / "pdf_prepared").mkdir(parents=True)
        (out / "pdf_prepared" / "parse_result.json").write_text(json.dumps({
            "content_list_file": str(out / "missing_cl.json"),
            "image_base": None}), encoding="utf-8")
        mineru = FakeMineru()
        scheduler = self._scheduler(mineru, FakeImageClient())
        scheduler.run()
        self.assertEqual(mineru.parse_calls, 1)
        self.assertEqual(self.store.get("doc")["status"], "DONE")

    def test_chunk_errors_trigger_retry_then_done(self):
        # H1: an error summary must raise → auto-retry (with retry_errors),
        # not a silent DONE.
        self._add()
        calls = []

        class FlakyProcessor(FakeProcessor):
            def extract(self, prepared, output_dir, image_data,
                        retry_errors=False, progress_cb=None, stop_requested=None):
                calls.append(retry_errors)
                if len(calls) == 1:
                    status = {"error": 1}
                else:
                    status = {"ok": 1}
                path = Path(output_dir) / "results.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    stream.write(json.dumps(
                        {"id": "example_1", "status": next(iter(status))}) + "\n")
                if progress_cb:
                    progress_cb(1, 1)
                return {"result_status": status}

        with mock.patch("batch.scheduler.DocumentProcessor", FlakyProcessor):
            scheduler = self._scheduler(FakeMineru(), FakeImageClient())
            claimed = scheduler.run()
        self.assertEqual(claimed, 2)
        self.assertEqual(self.store.get("doc")["status"], "DONE")
        self.assertEqual(calls, [False, True])

    def test_persistent_chunk_errors_end_failed(self):
        # H1: when attempts are exhausted the task must stay FAILED (retriable),
        # never DONE with missing results.
        self._add()

        class BrokenProcessor(FakeProcessor):
            def extract(self, prepared, output_dir, image_data,
                        retry_errors=False, progress_cb=None, stop_requested=None):
                path = Path(output_dir) / "results.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    stream.write(json.dumps(
                        {"id": "example_1", "status": "error"}) + "\n")
                return {"result_status": {"error": 1}}

        with mock.patch("batch.scheduler.DocumentProcessor", BrokenProcessor):
            scheduler = self._scheduler(FakeMineru(), FakeImageClient(),
                                        max_attempts=2)
            claimed = scheduler.run()
        self.assertEqual(claimed, 2)
        task = self.store.get("doc")
        self.assertEqual(task["status"], "FAILED")
        self.assertIn("failed during extraction", task["error"])

    # ── regression tests: doc/batch-code-review-round2.md ──

    def test_h1_stale_chunk_error_does_not_fail_task(self):
        # H1: an error record for a chunk id that no longer exists (a
        # re-parse changed the segmentation) must not keep the task FAILED.
        self._add()

        class StaleRecordProcessor(FakeProcessor):
            def extract(self, prepared, output_dir, image_data,
                        retry_errors=False, progress_cb=None, stop_requested=None):
                path = Path(output_dir) / "results.jsonl"
                records = [{"id": "example_1", "status": "ok"},
                           {"id": "example_2", "status": "ok"},
                           {"id": "removed_example_9", "status": "error"}]
                with path.open("w", encoding="utf-8") as stream:
                    for record in records:
                        stream.write(json.dumps(record) + "\n")
                return {"result_status": {"ok": 2}}

        with mock.patch("batch.scheduler.DocumentProcessor", StaleRecordProcessor):
            scheduler = self._scheduler(FakeMineru(), FakeImageClient())
            scheduler.run()
        self.assertEqual(self.store.get("doc")["status"], "DONE")

    def test_h1_unknown_status_fails_task(self):
        # H1 (附带): a latest status that is neither ok nor error must not
        # slip through as DONE.
        self._add()

        class WeirdProcessor(FakeProcessor):
            def extract(self, prepared, output_dir, image_data,
                        retry_errors=False, progress_cb=None, stop_requested=None):
                path = Path(output_dir) / "results.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    stream.write(json.dumps(
                        {"id": "example_1", "status": "weird"}) + "\n")
                return {"result_status": {"weird": 1}}

        with mock.patch("batch.scheduler.DocumentProcessor", WeirdProcessor):
            scheduler = self._scheduler(FakeMineru(), FakeImageClient(),
                                        auto_retry=False)
            scheduler.run()
        task = self.store.get("doc")
        self.assertEqual(task["status"], "FAILED")
        self.assertIn("failed during extraction", task["error"])

    def test_h2_reparse_marks_downstream_caches_stale(self):
        # H2: a re-parse must retire the previous parse's image_result.json
        # and results.jsonl instead of letting later stages reuse them.
        self._add()
        out = self.out_root / "doc"
        (out / "pdf_prepared").mkdir(parents=True)
        (out / "pdf_prepared" / "parse_result.json").write_text(json.dumps({
            "content_list_file": str(out / "missing_cl.json"),
            "image_base": None}), encoding="utf-8")
        (out / "image_result.json").write_text(
            json.dumps({"dict_result": {"1": "OLD"}}), encoding="utf-8")
        (out / "results.jsonl").write_text(
            '{"id": "example_1", "status": "ok"}\n', encoding="utf-8")
        mineru = FakeMineru()
        image_client = FakeImageClient()
        scheduler = self._scheduler(mineru, image_client)
        scheduler.run()
        self.assertEqual(mineru.parse_calls, 1)
        self.assertTrue((out / "image_result.json.stale").is_file())
        self.assertTrue((out / "results.jsonl.stale").is_file())
        # the stale caches were not consumed: mining re-ran
        self.assertEqual(image_client.calls, 1)
        self.assertEqual(self.store.get("doc")["status"], "DONE")

    def test_h2_fingerprint_mismatch_triggers_reparse(self):
        # H2: a marker whose recorded fingerprint no longer matches the
        # content_list file is stale → re-parse.
        self._add()
        out = self.out_root / "doc"
        (out / "pdf_prepared").mkdir(parents=True)
        (out / "cl.json").write_text("[]", encoding="utf-8")
        (out / "images").mkdir()
        (out / "pdf_prepared" / "parse_result.json").write_text(json.dumps({
            "content_list_file": str(out / "cl.json"),
            "image_base": str(out / "images"),
            "content_list_fingerprint": "md5:deadbeef"}), encoding="utf-8")
        mineru = FakeMineru()
        scheduler = self._scheduler(mineru, FakeImageClient())
        scheduler.run()
        self.assertEqual(mineru.parse_calls, 1)
        self.assertEqual(self.store.get("doc")["status"], "DONE")

    def test_h2_mining_reused_only_for_same_parse(self):
        # H2: image_result.json without a matching parse fingerprint is
        # re-mined even when the parse itself was reused.
        self._add()
        out = self.out_root / "doc"
        (out / "pdf_prepared").mkdir(parents=True)
        (out / "cl.json").write_text("[]", encoding="utf-8")
        (out / "images").mkdir()
        fingerprint = "md5:" + hashlib.md5(b"[]").hexdigest()
        (out / "pdf_prepared" / "parse_result.json").write_text(json.dumps({
            "content_list_file": str(out / "cl.json"),
            "image_base": str(out / "images"),
            "content_list_fingerprint": fingerprint}), encoding="utf-8")
        (out / "image_result.json").write_text(
            json.dumps({"dict_result": {"1": "OLD"}}), encoding="utf-8")
        mineru = FakeMineru()
        image_client = FakeImageClient()
        scheduler = self._scheduler(mineru, image_client)
        scheduler.run()
        self.assertEqual(mineru.parse_calls, 0)   # parse reused
        self.assertEqual(image_client.calls, 1)   # mining re-run

    def test_h3_final_product_failure_fails_task(self):
        # H3: a final-product error must not end as DONE; the task fails and
        # can be retried.
        self._add()

        class FPProcessor(FakeProcessor):
            def extract(self, prepared, output_dir, image_data,
                        retry_errors=False, progress_cb=None, stop_requested=None):
                path = Path(output_dir) / "results.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    stream.write(json.dumps(
                        {"id": "example_1", "status": "ok"}) + "\n")
                return {"result_status": {"ok": 1},
                        "final_product": {"error": "RuntimeError: target LLM 429"}}

        with mock.patch("batch.scheduler.DocumentProcessor", FPProcessor):
            scheduler = self._scheduler(FakeMineru(), FakeImageClient(),
                                        auto_retry=False)
            scheduler.run()
        task = self.store.get("doc")
        self.assertEqual(task["status"], "FAILED")
        self.assertIn("final product", task["error"])

    def test_h5_failed_task_with_attempts_requeued_on_startup(self):
        # H5: a task that failed during the in-memory backoff window (stop or
        # crash) is FAILED with attempts remaining — the next run must pick
        # it up again instead of stranding it.
        self._add()
        task = self.store.claim_next(run_id="crashed-run")
        self.store.mark_failed(task["id"], "RuntimeError: mineru exploded",
                               run_id="crashed-run", retry_pending=True)
        self.assertEqual(self.store.get("doc")["status"], "FAILED")
        scheduler = self._scheduler(FakeMineru(), FakeImageClient())
        claimed = scheduler.run()
        self.assertEqual(claimed, 1)
        self.assertEqual(self.store.get("doc")["status"], "DONE")

    def test_h6_idle_recovery_never_reenters_current_run(self):
        # H6: an expired lock must not let idle recovery re-claim a task this
        # run is still executing (duplicate mining / concurrent writes).
        self._add("slow")
        self._add("fast")
        calls = {}

        class SlowFastImageClient(FakeImageClient):
            def process(self, image_base, return_per_image=True, result_path=None):
                calls[image_base] = calls.get(image_base, 0) + 1
                if "slow" in image_base:
                    time.sleep(1.5)  # longer than lock_expire_seconds
                return {"dict_result": {"1": "CCO"}, "image_result": [],
                        "per_image_detections": []}

        scheduler = self._scheduler(FakeMineru(), SlowFastImageClient(),
                                    task_workers=2, lock_expire_seconds=1,
                                    poll_interval=0.05, daemon=True)
        thread = threading.Thread(target=scheduler.run)
        thread.start()
        deadline = time.time() + 20
        while time.time() < deadline and self.store.counts()["DONE"] < 2:
            time.sleep(0.05)
        scheduler._stop.set()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.store.counts()["DONE"], 2)
        slow_calls = [n for base, n in calls.items() if "slow" in base]
        self.assertEqual(slow_calls, [1])
        slow = self.store.get("slow")
        self.assertEqual(slow["status"], "DONE")
        self.assertEqual(slow["attempt_count"], 1)

    def test_batch_keeps_results_per_document_without_summary(self):
        self._add()
        scheduler = self._scheduler(FakeMineru(), FakeImageClient())
        claimed = scheduler.run()
        self.assertEqual(claimed, 1)
        self.assertEqual(self.store.get("doc")["status"], "DONE")
        self.assertFalse((self.root / "delivery_summary").exists())
        task = self.store.get("doc")
        identity = json.loads((Path(task["output_dir"]) / "source_identity.json").read_text())
        self.assertEqual(identity["document_id"], Path(task["pdf_path"]).name)

    def test_duplicate_pdf_names_are_qualified_even_for_new_tasks(self):
        self.store.add_task("first", "/left/shared.pdf", str(self.out_root / "first"))
        scheduler = self._scheduler(FakeMineru(), FakeImageClient())
        self.store.add_task("second", "/right/shared.pdf", str(self.out_root / "second"))
        self.assertEqual(scheduler.run(), 2)
        first = json.loads((self.out_root / "first/source_identity.json").read_text())
        second = json.loads((self.out_root / "second/source_identity.json").read_text())
        self.assertEqual(first["document_id"], "shared.pdf")
        self.assertEqual(second["document_id"], "second/shared.pdf")
        self.assertFalse((self.root / "delivery_summary").exists())

    def test_h6_late_requeue_does_not_overwrite_done(self):
        # H6: a stale attempt's requeue is fenced by run_id and cannot pull a
        # DONE task back to PENDING.
        self._add()
        scheduler = self._scheduler(FakeMineru(), FakeImageClient())
        scheduler.run()
        self.assertEqual(self.store.get("doc")["status"], "DONE")
        task_id = self.store.get("doc")["id"]
        scheduler._requeue(task_id, run_id=scheduler.run_id)
        self.assertEqual(self.store.get("doc")["status"], "DONE")


if __name__ == "__main__":
    unittest.main()
