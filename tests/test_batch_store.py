"""Tests for batch/store.py — SQLite task lifecycle."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from batch.store import TaskStore, build_uuid


class TaskStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "tasks.db"
        self.store = TaskStore(self.db)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_add_is_idempotent(self):
        first, created1 = self.store.add_task("a", "/x/a.pdf", "/out/a", "en")
        second, created2 = self.store.add_task("a", "/x/a.pdf", "/out/a", "en")
        self.assertTrue(created1)
        self.assertFalse(created2)
        self.assertEqual(first, second)
        self.assertEqual(self.store.counts()["PENDING"], 1)

    def test_claim_next_orders_and_increments_attempts(self):
        self.store.add_task("a", "/x/a.pdf", "/out/a")
        time.sleep(0.01)
        self.store.add_task("b", "/x/b.pdf", "/out/b")
        first = self.store.claim_next()
        second = self.store.claim_next()
        self.assertEqual(first["uuid"], "a")
        self.assertEqual(second["uuid"], "b")
        self.assertEqual(first["status"], "RUNNING")
        self.assertEqual(first["attempt_count"], 1)
        self.assertEqual(second["attempt_count"], 1)
        self.assertIsNone(self.store.claim_next())

    def test_stage_progress_done(self):
        task_id, _ = self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next()
        self.store.update_stage(task_id, "extracting", chunks_total=10)
        self.store.update_progress(task_id, 3)
        task = self.store.list_tasks(status="RUNNING")[0]
        self.assertEqual(task["stage"], "extracting")
        self.assertEqual(task["chunks_total"], 10)
        self.assertEqual(task["chunks_done"], 3)
        self.store.mark_done(task_id, 12.5)
        task = self.store.get("a")
        self.assertEqual(task["status"], "DONE")
        self.assertEqual(task["elapsed_seconds"], 12.5)
        self.assertEqual(task["stage"], "")

    def test_mark_failed_and_retry(self):
        task_id, _ = self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next()
        self.store.mark_failed(task_id, "boom")
        task = self.store.get("a")
        self.assertEqual(task["status"], "FAILED")
        self.assertEqual(task["error"], "boom")
        moved = self.store.retry(all_failed=True)
        self.assertEqual(moved, 1)
        self.assertEqual(self.store.get("a")["status"], "PENDING")
        # retry of a PENDING task is a no-op
        self.assertEqual(self.store.retry(uuid="a"), 0)

    def test_recover_stuck_requeues_expired(self):
        task_id, _ = self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next()
        # simulate an expired lock
        self.store._execute("UPDATE pdf_task SET locked_at=? WHERE id=?",
                            (time.time() - 99999, task_id))
        result = self.store.recover_stuck(lock_expire_seconds=60, max_attempts=3)
        self.assertEqual(result["requeued"], 1)
        self.assertEqual(self.store.get("a")["status"], "PENDING")

    def test_recover_stuck_requeues_other_run_with_fresh_lock(self):
        task_id, _ = self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next(run_id="crashed-run")
        # fresh lock: only the run_id mismatch can identify it as stuck
        result = self.store.recover_stuck(lock_expire_seconds=99999,
                                          max_attempts=3,
                                          current_run_id="current-run")
        self.assertEqual(result["requeued"], 1)
        task = self.store.get("a")
        self.assertEqual(task["status"], "PENDING")
        self.assertIsNone(task["run_id"])

    def test_recover_stuck_keeps_current_run_fresh_lock(self):
        self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next(run_id="current-run")
        result = self.store.recover_stuck(lock_expire_seconds=99999,
                                          max_attempts=3,
                                          current_run_id="current-run")
        self.assertEqual(result, {"requeued": 0, "failed": 0})
        self.assertEqual(self.store.get("a")["status"], "RUNNING")

    def test_retry_done_with_errors(self):
        out = Path(self.tmp.name) / "done_out"
        out.mkdir()
        (out / "results.jsonl").write_text(
            '{"id": "a", "status": "error"}\n'
            '{"id": "a", "status": "ok", "reactions": [{"status": "ok"}]}\n'
            '{"id": "b", "status": "ok", "reactions": [{"status": "ok"}]}\n', encoding="utf-8")
        task_id, _ = self.store.add_task("done-ok", "/x/a.pdf", str(out))
        self.store.claim_next()
        self.store.mark_done(task_id, 1.0)
        # retried-and-fixed records (latest status ok) must not be requeued
        self.assertEqual(self.store.retry_done_with_errors(), 0)
        (out / "results.jsonl").write_text(
            '{"id": "a", "status": "ok"}\n'
            '{"id": "b", "status": "error"}\n', encoding="utf-8")
        self.assertEqual(self.store.retry_done_with_errors(), 1)
        self.assertEqual(self.store.get("done-ok")["status"], "PENDING")

    def test_retry_done_without_procedures(self):
        out = Path(self.tmp.name) / "empty_out"
        out.mkdir()
        (out / "summary.json").write_text('{"with_procedure": 0}', encoding="utf-8")
        task_id, _ = self.store.add_task("empty", "/x/empty.pdf", str(out))
        self.store.claim_next()
        self.store.mark_done(task_id, 1.0)
        self.assertEqual(self.store.retry_done_without_procedures(True), 1)
        task = self.store.get("empty")
        self.assertEqual(task["status"], "PENDING")
        self.assertEqual(task["attempt_count"], 0)

    def test_retry_done_outdated_respects_limit(self):
        for name, version in (("old", "old-version"),
                              ("current", "v2"), ("old2", "old-version")):
            out = Path(self.tmp.name) / name
            out.mkdir()
            (out / "summary.json").write_text(
                json.dumps({"segmentation_version": version}), encoding="utf-8")
            task_id, _ = self.store.add_task(name, f"/x/{name}.pdf", str(out))
            self.store.claim_next()
            self.store.mark_done(task_id, 1.0)
        self.assertEqual(self.store.retry_done_outdated("v2", limit=1), 1)
        self.assertEqual(self.store.get("old")["status"], "PENDING")
        self.assertEqual(self.store.get("current")["status"], "DONE")
        self.assertEqual(self.store.retry_done_outdated("v2"), 1)
        self.assertEqual(self.store.get("old2")["status"], "PENDING")

    def test_recover_stuck_fails_when_attempts_exhausted(self):
        task_id, _ = self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next()
        self.store.claim_next()  # not claimable: RUNNING; bump attempts directly
        self.store._execute("UPDATE pdf_task SET attempt_count=3, locked_at=? WHERE id=?",
                            (time.time() - 99999, task_id))
        result = self.store.recover_stuck(lock_expire_seconds=60, max_attempts=3)
        self.assertEqual(result["failed"], 1)
        task = self.store.get("a")
        self.assertEqual(task["status"], "FAILED")
        self.assertIn("attempts exhausted", task["error"])

    def test_recover_stuck_ignores_expired_lock_of_current_run(self):
        # review round-2 H6: lock expiry alone must not requeue a task the
        # current run is still executing (long stages do not heartbeat).
        self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next(run_id="current-run")
        self.store._execute("UPDATE pdf_task SET locked_at=? WHERE uuid=?",
                            (0.0, "a"))
        result = self.store.recover_stuck(lock_expire_seconds=60, max_attempts=3,
                                          current_run_id="current-run")
        self.assertEqual(result, {"requeued": 0, "failed": 0})
        self.assertEqual(self.store.get("a")["status"], "RUNNING")

    def test_state_writes_are_fenced_by_run_id(self):
        # review round-2 H6: a stale attempt cannot overwrite the state of a
        # task that was reset or re-claimed by another run.
        task_id, _ = self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next(run_id="run-1")
        self.store.mark_done(task_id, 1.0, run_id="run-2")       # stale writer
        self.assertEqual(self.store.get("a")["status"], "RUNNING")
        self.store.mark_failed(task_id, "boom", run_id="run-2")  # stale writer
        self.assertEqual(self.store.get("a")["status"], "RUNNING")
        self.store.update_stage(task_id, "mining", run_id="run-2")
        self.assertNotEqual(self.store.get("a")["stage"], "mining")
        self.store.mark_done(task_id, 1.0, run_id="run-1")       # owner
        self.assertEqual(self.store.get("a")["status"], "DONE")

    def test_requeue_failed_with_attempts(self):
        # review round-2 H5: FAILED tasks with attempts left are requeued at
        # scheduler startup; exhausted ones stay FAILED.
        task_id, _ = self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next()
        self.store.mark_failed(task_id, "boom", retry_pending=True)
        self.assertEqual(self.store.requeue_failed_with_attempts(3), 1)
        task = self.store.get("a")
        self.assertEqual(task["status"], "PENDING")
        self.assertEqual(task["error"], "boom")  # kept as a note
        self.store.claim_next()
        self.store.mark_failed(task_id, "boom2", retry_pending=True)
        self.store._execute("UPDATE pdf_task SET attempt_count=3 WHERE id=?",
                            (task_id,))
        self.assertEqual(self.store.requeue_failed_with_attempts(3), 0)
        self.assertEqual(self.store.get("a")["status"], "FAILED")

    def test_results_have_errors_ignores_stale_ids(self):
        # review round-2 H1: records for chunk ids that no longer exist must
        # not count; any non-ok latest status of a current chunk must count.
        out = Path(self.tmp.name) / "out_stale"
        out.mkdir()
        (out / "chunks.jsonl").write_text(
            '{"id": "example_1", "has_procedure": true}\n', encoding="utf-8")
        (out / "results.jsonl").write_text(
            '{"id": "example_1", "status": "ok", "reactions": [{"status": "ok"}]}\n'
            '{"id": "gone_2", "status": "error"}\n', encoding="utf-8")
        self.assertFalse(TaskStore._results_have_errors(str(out)))
        (out / "results.jsonl").write_text(
            '{"id": "example_1", "status": "ok", "reactions": [{"status": "ok"}]}\n'
            '{"id": "example_1", "status": "weird"}\n', encoding="utf-8")
        self.assertTrue(TaskStore._results_have_errors(str(out)))

    def test_retry_done_missing_directory_and_legacy_result(self):
        missing = Path(self.tmp.name) / "missing"
        legacy = Path(self.tmp.name) / "legacy"
        legacy.mkdir()
        (legacy / "results.jsonl").write_text(json.dumps({
            "id": "example_3", "status": "ok", "validation": {
                "rxn_validation": {"best_expression": "CCO>>CC=O"}}}) + "\n")
        for name, out in (("missing", missing), ("legacy", legacy)):
            task_id, _ = self.store.add_task(name, f"/x/{name}.pdf", str(out))
            self.store.claim_next()
            self.store.mark_done(task_id, 1.0)
        self.assertEqual(self.store.retry_done_with_errors(), 2)
        self.assertTrue(all(self.store.get(name)["status"] == "PENDING"
                            for name in ("missing", "legacy")))

    def test_classified_non_reaction_remains_complete(self):
        out = Path(self.tmp.name) / "assay"
        out.mkdir()
        (out / "results.jsonl").write_text(json.dumps({
            "id": "a", "status": "non_reaction", "reason": "assay", "reactions": []}) + "\n")
        self.assertFalse(TaskStore._results_have_errors(str(out)))

    def test_retry_done_with_final_product_error(self):
        # review round-2 H3: a DONE task whose summary records a final-product
        # error is requeued by `retry --errors`.
        out = Path(self.tmp.name) / "out_fp"
        out.mkdir()
        (out / "summary.json").write_text(
            json.dumps({"result_status": {"ok": 1},
                        "final_product": {"error": "RuntimeError: target LLM 429"}}),
            encoding="utf-8")
        task_id, _ = self.store.add_task("done-fp", "/x/a.pdf", str(out))
        self.store.claim_next()
        self.store.mark_done(task_id, 1.0)
        self.assertEqual(self.store.retry_done_with_errors(), 1)
        self.assertEqual(self.store.get("done-fp")["status"], "PENDING")

    def test_reset_stuck(self):
        self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.claim_next()
        moved = self.store.reset_stuck()
        self.assertEqual(moved, 1)
        self.assertEqual(self.store.get("a")["status"], "PENDING")

    def test_counts_and_list_filter(self):
        self.store.add_task("a", "/x/a.pdf", "/out/a")
        self.store.add_task("b", "/x/b.pdf", "/out/b")
        self.store.claim_next()
        counts = self.store.counts()
        self.assertEqual(counts["PENDING"], 1)
        self.assertEqual(counts["RUNNING"], 1)
        self.assertEqual(counts["TOTAL"], 2)
        running = self.store.list_tasks(status="RUNNING")
        self.assertEqual([t["uuid"] for t in running], ["a"])

    def test_build_uuid_stable_and_path_sensitive(self):
        pdf = Path("/data/Some Patent.pdf")
        self.assertEqual(build_uuid(pdf), build_uuid(pdf))
        self.assertNotEqual(build_uuid(pdf), build_uuid(Path("/data2/Some Patent.pdf")))
        self.assertTrue(build_uuid(pdf).startswith("Some Patent_"))


if __name__ == "__main__":
    unittest.main()
