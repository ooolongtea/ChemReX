"""Tests for batch/cli.py — add / list / status / retry / LLM env defaults."""

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from batch import config as batch_config
from batch.cli import apply_llm_env, main
from batch.store import TaskStore


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "tasks.db"
        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir()
        for name in ("a.pdf", "b.pdf"):
            (self.pdf_dir / name).write_bytes(b"%PDF-1.4 fake")
        (self.pdf_dir / "notes.txt").write_text("not a pdf")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *argv):
        return main(["--db", str(self.db)] + list(argv))

    def test_add_scan_and_idempotency(self):
        code = self._run("add", str(self.pdf_dir))
        self.assertEqual(code, 0)
        store = TaskStore(self.db)
        self.assertEqual(store.counts()["PENDING"], 2)
        store.close()
        # re-adding the same directory adds nothing
        code = self._run("add", str(self.pdf_dir))
        self.assertEqual(code, 0)
        store = TaskStore(self.db)
        self.assertEqual(store.counts()["PENDING"], 2)
        store.close()

    def test_add_single_file_and_limit(self):
        self.assertEqual(self._run("add", str(self.pdf_dir / "a.pdf")), 0)
        # limit=1 scans sorted [a.pdf, b.pdf] → a.pdf only, already present
        self.assertEqual(self._run("add", str(self.pdf_dir), "--limit", "1"), 0)
        store = TaskStore(self.db)
        self.assertEqual(store.counts()["PENDING"], 1)
        store.close()

    def test_list_and_status(self):
        self._run("add", str(self.pdf_dir))
        self.assertEqual(self._run("list", "--status", "PENDING"), 0)
        self.assertEqual(self._run("status"), 0)

    def test_retry_failed(self):
        self._run("add", str(self.pdf_dir))
        store = TaskStore(self.db)
        task = store.list_tasks()[0]
        store.claim_next()
        store.mark_failed(task["id"], "boom")
        store.close()
        self.assertEqual(self._run("retry", "--failed"), 0)
        store = TaskStore(self.db)
        self.assertEqual(store.counts()["PENDING"], 2)
        self.assertEqual(store.counts()["FAILED"], 0)
        store.close()

    def test_retry_done_with_error_records(self):
        out = self.root / "task_out"
        out.mkdir()
        (out / "results.jsonl").write_text(
            '{"id": "x", "status": "error"}\n', encoding="utf-8")
        store = TaskStore(self.db)
        task_id, _ = store.add_task("doc", "/x/doc.pdf", str(out))
        store.claim_next()
        store.mark_done(task_id, 1.0)
        store.close()
        self.assertEqual(self._run("retry", "--errors"), 0)
        store = TaskStore(self.db)
        self.assertEqual(store.get("doc")["status"], "PENDING")
        store.close()

    def test_retry_done_uuid_reports_error(self):
        store = TaskStore(self.db)
        task_id, _ = store.add_task("doc", "/x/doc.pdf", str(self.root / "out"))
        store.claim_next()
        store.mark_done(task_id, 1.0)
        store.close()
        self.assertEqual(self._run("retry", "doc"), 1)
        self.assertEqual(self._run("retry", "missing"), 1)

    def test_add_no_pdfs_fails(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(self._run("add", str(empty)), 1)

    def test_add_language_change_warns(self):
        # review round-2 M1: re-adding with a different language must not be
        # a silent no-op.
        import contextlib
        import io

        self._run("add", str(self.pdf_dir / "a.pdf"), "--language", "en")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = self._run("add", str(self.pdf_dir / "a.pdf"), "--language", "zh")
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("warning", output)
        self.assertIn("language=en", output)
        store = TaskStore(self.db)
        task = store.list_tasks()[0]
        self.assertEqual(task["language"], "en")  # original kept
        store.close()

    def test_reset_stuck_refuses_running_scheduler(self):
        # review round-2 M3: reset-stuck must refuse while a scheduler holds
        # the instance lock (stale-writer interlock).
        import fcntl

        lock_path = Path(str(self.db)).with_suffix(".lock")
        lock = open(lock_path, "a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            self.assertEqual(self._run("reset-stuck"), 1)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        # with the scheduler stopped, reset-stuck works again
        self.assertEqual(self._run("reset-stuck"), 0)

    def test_run_refuses_second_scheduler(self):
        import fcntl

        lock_path = Path(str(self.db)).with_suffix(".lock")
        lock = open(lock_path, "a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.addCleanup(lock.close)
        saved = {key: os.environ.get(key) for key in
                 ("EXTRACTION_PRESET", "STANDARDIZE_PROVIDER", "COREF_PROVIDER",
                  "STANDARDIZE_MODEL", "COREF_MODEL")}
        try:
            self.assertEqual(self._run("run", "--no-image"), 1)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class ApplyLlmEnvTest(unittest.TestCase):
    """Batch defaults to the MaaS preset; flags and BATCH_* win over it."""

    def _args(self, **kwargs):
        defaults = dict(preset=None, preset_zh=None, preset_en=None,
                        llm_provider=None, llm_model=None,
                        offline_standardize=False)
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_defaults_pin_maas(self):
        saved = {k: os.environ.get(k) for k in
                 ("EXTRACTION_PRESET", "STANDARDIZE_PROVIDER", "COREF_PROVIDER",
                  "STANDARDIZE_MODEL", "COREF_MODEL")}
        try:
            apply_llm_env(self._args())
            self.assertEqual(os.environ["EXTRACTION_PRESET"],
                             batch_config.BATCH_EXTRACTION_PRESET)
            self.assertEqual(os.environ["STANDARDIZE_PROVIDER"],
                             batch_config.BATCH_LLM_PROVIDER)
            self.assertEqual(os.environ["COREF_MODEL"], batch_config.BATCH_LLM_MODEL)
            self.assertEqual(os.environ["STANDARDIZE_MODEL"], batch_config.BATCH_LLM_MODEL)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_per_language_presets_override_routing(self):
        saved = dict(batch_config.BATCH_EXTRACTION_PRESETS)
        try:
            apply_llm_env(self._args(preset_zh="chatgpt", preset_en="deepseek-v4-pro"))
            self.assertEqual(batch_config.BATCH_EXTRACTION_PRESETS["zh"], "chatgpt")
            self.assertEqual(batch_config.BATCH_EXTRACTION_PRESETS["en"], "deepseek-v4-pro")
        finally:
            batch_config.BATCH_EXTRACTION_PRESETS.clear()
            batch_config.BATCH_EXTRACTION_PRESETS.update(saved)

    def test_flags_override_defaults(self):
        saved = {k: os.environ.get(k) for k in
                 ("EXTRACTION_PRESET", "STANDARDIZE_PROVIDER", "COREF_PROVIDER",
                  "STANDARDIZE_MODEL", "COREF_MODEL", "STANDARDIZE_OFFLINE")}
        try:
            apply_llm_env(self._args(preset="chatgpt", llm_provider="openai",
                                     llm_model="gpt-4o", offline_standardize=True))
            self.assertEqual(os.environ["EXTRACTION_PRESET"], "chatgpt")
            self.assertEqual(os.environ["STANDARDIZE_PROVIDER"], "openai")
            self.assertEqual(os.environ["COREF_MODEL"], "gpt-4o")
            self.assertEqual(os.environ["STANDARDIZE_OFFLINE"], "1")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
