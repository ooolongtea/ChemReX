"""Tests for batch/image_client.py — subprocess supervision and line protocol."""

import json
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from batch.image_client import ImageWorkerClient, ImageWorkerError

ECHO_WORKER = textwrap.dedent("""
    import json, sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        resp = {"request_id": req["request_id"], "status": "ok",
                "result": {"echo": req["image_base"]}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
""")

CRASH_AFTER_ONE = textwrap.dedent("""
    import json, sys
    answered = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if answered:
            sys.exit(3)
        answered = True
        resp = {"request_id": req["request_id"], "status": "ok", "result": {}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
""")

SLEEPING_WORKER = textwrap.dedent("""
    import sys, time
    for line in sys.stdin:
        time.sleep(30)
""")

STALE_THEN_REAL = textwrap.dedent("""
    import json, sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        stale = {"request_id": "stale-id", "status": "ok", "result": {"stale": True}}
        sys.stdout.write(json.dumps(stale) + "\\n")
        sys.stdout.flush()
        fresh = {"request_id": req["request_id"], "status": "ok",
                 "result": {"fresh": True}}
        sys.stdout.write(json.dumps(fresh) + "\\n")
        sys.stdout.flush()
""")

BAD_UTF8_WORKER = textwrap.dedent("""
    import sys, time
    sys.stdout.buffer.write(b'\\xff\\xfe not utf8\\n')
    sys.stdout.flush()
    time.sleep(30)
""")

FILE_RESULT_WORKER = textwrap.dedent("""
    import json, sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        path = req["result_path"]
        payload = {"echo": req["image_base"], "payload": "x" * 100000}
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        resp = {"request_id": req["request_id"], "status": "ok",
                "result_file": path}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
""")

MISSING_FILE_WORKER = textwrap.dedent("""
    import json, sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        resp = {"request_id": req["request_id"], "status": "ok",
                "result_file": req["result_path"]}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
""")


class ImageWorkerClientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _script(self, source: str) -> Path:
        path = self.root / "worker.py"
        path.write_text(source, encoding="utf-8")
        return path

    def _client(self, script: Path, timeout: int = 20) -> ImageWorkerClient:
        return ImageWorkerClient(
            command=[sys.executable, "-u", str(script)], timeout=timeout)

    def test_round_trip(self):
        client = self._client(self._script(ECHO_WORKER))
        try:
            result = client.process("/images/dir")
            self.assertEqual(result, {"echo": "/images/dir"})
        finally:
            client.stop()

    def test_restart_after_crash(self):
        client = self._client(self._script(CRASH_AFTER_ONE))
        try:
            first = client.process("/a")
            self.assertEqual(first, {})
            # worker exits after the first response; next call restarts it
            second = client.process("/b")
            self.assertEqual(second, {})
        finally:
            client.stop()

    def test_timeout_kills_worker(self):
        client = self._client(self._script(SLEEPING_WORKER), timeout=1)
        try:
            with self.assertRaises(ImageWorkerError) as ctx:
                client.process("/a")
            self.assertIn("timed out", str(ctx.exception))
            # client killed the process; a fresh request spawns a new worker
            self.assertIsNone(client._proc)
        finally:
            client.stop()

    def test_stale_response_is_discarded(self):
        # M1: a reply with a foreign request_id (e.g. left over from a killed
        # worker) is discarded; the current request still gets its answer.
        client = self._client(self._script(STALE_THEN_REAL))
        try:
            result = client.process("/images/dir")
            self.assertEqual(result, {"fresh": True})
        finally:
            client.stop()

    def test_restart_uses_fresh_response_queue(self):
        # A new worker process gets its own queue, so old responses can never
        # be read as the new request's answer.
        client = self._client(self._script(CRASH_AFTER_ONE))
        try:
            self.assertEqual(client.process("/a"), {})
            first_queue = client._responses
            self.assertEqual(client.process("/b"), {})
            self.assertIsNot(client._responses, first_queue)
        finally:
            client.stop()

    def test_result_file_round_trip_and_cleanup(self):
        client = self._client(self._script(FILE_RESULT_WORKER))
        path = self.root / "result.json"
        try:
            result = client.process("/images/dir", result_path=path)
            self.assertEqual(result["echo"], "/images/dir")
            self.assertEqual(len(result["payload"]), 100000)
            self.assertFalse(path.exists())
        finally:
            client.stop()

    def test_default_result_path_is_beside_document(self):
        client = self._client(self._script(FILE_RESULT_WORKER))
        image_base = self.root / "out" / "pdf_prepared" / "stem" / "images"
        image_base.mkdir(parents=True)
        try:
            result = client.process(str(image_base))
            self.assertEqual(result["echo"], str(image_base))
            self.assertFalse((self.root / "out" / ".image_worker_result.json").exists())
        finally:
            client.stop()

    def test_missing_result_file_raises(self):
        client = self._client(self._script(MISSING_FILE_WORKER))
        try:
            with self.assertRaises(ImageWorkerError):
                client.process("/a", result_path=self.root / "missing.json")
        finally:
            client.stop()

    def test_reader_failure_fails_fast(self):
        # review round-2 M4: a reader thread killed by undecodable output
        # must fail the request immediately instead of silently blocking
        # until the full per-request timeout.
        client = self._client(self._script(BAD_UTF8_WORKER), timeout=10)
        try:
            started = time.monotonic()
            with self.assertRaises(ImageWorkerError) as ctx:
                client.process("/a")
            self.assertIn("reader thread failed", str(ctx.exception))
            self.assertLess(time.monotonic() - started, 8)
            self.assertIsNone(client._proc)
        finally:
            client.stop()

    def test_restarts_do_not_leak_log_handles(self):
        # review round-2 M2: each restart must close the previous worker's
        # log file handle instead of opening one more fd per crash.
        import os
        if not os.path.isdir("/proc/self/fd"):
            self.skipTest("/proc/self/fd not available")
        log_path = self.root / "worker.log"
        client = ImageWorkerClient(
            command=[sys.executable, "-u", str(self._script(CRASH_AFTER_ONE))],
            timeout=20, log_path=log_path)

        def open_fds_for(path):
            count = 0
            for fd in os.listdir("/proc/self/fd"):
                try:
                    if os.readlink(f"/proc/self/fd/{fd}") == str(path):
                        count += 1
                except OSError:
                    pass
            return count

        try:
            for _ in range(3):
                self.assertEqual(client.process("/a"), {})  # restarts the worker
            self.assertEqual(open_fds_for(log_path), 1)
        finally:
            client.stop()


if __name__ == "__main__":
    unittest.main()
