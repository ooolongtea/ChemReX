"""Scheduler-side supervisor for the resident image worker subprocess.

Spawns batch/image_worker.py in the imgparser conda env, feeds it one request
at a time over stdin (a lock serialises concurrent task threads), and reads
JSON responses from a persistent reader thread with a per-request timeout.
A hung or crashed worker is killed and respawned; the affected task fails
and can be retried.
"""

import json
import queue
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class ImageWorkerError(RuntimeError):
    pass


class ImageWorkerClient:
    def __init__(self, command: list, timeout: int = 3600,
                 log_path: Optional[Path] = None):
        self.command = command
        self.timeout = timeout
        self.log_path = log_path
        self._lock = threading.Lock()      # one request in flight at a time
        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None
        self._responses: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._reader_state: Dict[str, bool] = {"failed": False}

    # ── process lifecycle ──
    def _start(self) -> subprocess.Popen:
        # Without a log file, discard stderr: a pipe nobody reads would fill
        # up (64 KB) and block the worker on its next log line.
        stderr = subprocess.DEVNULL
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Close the previous worker's handle first: repeated crashes and
            # restarts in a long daemon would otherwise leak one fd per
            # restart (review round-2 M2).
            if self._log_file and not self._log_file.closed:
                self._log_file.close()
            self._log_file = open(self.log_path, "ab")
            stderr = self._log_file
        logger.info("starting image worker: {}", " ".join(self.command))
        self._proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, encoding="utf-8", bufsize=1)
        # A fresh queue per process: responses from a killed worker can never
        # be mistaken for the current worker's reply.
        self._responses = queue.Queue()
        # Per-process reader state: a late failure flag from a killed
        # worker's reader must not fail the next worker's requests.
        self._reader_state = {"failed": False}
        self._reader = threading.Thread(
            target=self._read_loop, args=(self._proc, self._responses,
                                          self._reader_state),
            daemon=True, name="image-worker-reader")
        self._reader.start()
        return self._proc

    def _read_loop(self, proc: subprocess.Popen,
                   responses: "queue.Queue[Dict[str, Any]]",
                   state: Dict[str, bool]) -> None:
        # Any uncaught reader failure must wake the waiter immediately: a
        # silent dead reader plus a worker blocked on a full stdout pipe was
        # observed as a 30-minute mining stall on a 1,548-image document.
        try:
            while True:
                try:
                    line = proc.stdout.readline()
                except (OSError, ValueError) as exc:
                    # UnicodeDecodeError/OSError from a broken pipe must not
                    # kill the reader silently (review round-2 M4).
                    state["failed"] = True
                    logger.error("image worker reader thread failed: {}", exc)
                    break
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    responses.put(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("image worker sent a non-JSON line: {}", line[:200])
        except Exception as exc:
            state["failed"] = True
            logger.exception("image worker reader thread crashed: {}", exc)
        finally:
            logger.info("image worker stdout closed")

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            if self._proc is not None:
                logger.warning("image worker exited (code {}); restarting",
                               self._proc.returncode)
            self._start()
        return self._proc

    def ensure_started(self) -> None:
        """Public prewarm hook (used by the scheduler at startup)."""
        with self._lock:
            self._ensure_started()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is not None:
                try:
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.close()
                    proc.wait(timeout=30)
                except Exception:
                    proc.kill()
            if self._log_file:
                self._log_file.close()
                self._log_file = None
        logger.info("image worker stopped")

    # ── request protocol ──
    @staticmethod
    def _default_result_path(image_base: str) -> Path:
        """Place the result beside the document, not in the pipe.

        ``image_base`` is <output>/pdf_prepared/<stem>/images, so parents[2]
        is <output>. Fall back to the batch temp dir for unusual layouts.
        """
        base = Path(image_base)
        try:
            return base.parents[2] / ".image_worker_result.json"
        except IndexError:
            return Path(tempfile.gettempdir()) / ".image_worker_result.json"

    def process(self, image_base: str, return_per_image: bool = True,
                result_path: Optional[Path] = None) -> Dict[str, Any]:
        """Run image mining for one document; blocks until reply or timeout.

        The worker writes the per-image result to *result_path* and returns
        only a small ``result_file`` pointer. This keeps multi-megabyte JSON
        off the stdout pipe, where it could fill the pipe buffer and stall
        both processes while the reader is busy or starved.

        A worker that dies mid-request is restarted once and the request
        resent; a second death (or a real timeout) raises ImageWorkerError.
        """
        with self._lock:
            request_id = uuid.uuid4().hex[:12]
            if result_path is None:
                result_path = self._default_result_path(image_base)
            request = {"request_id": request_id, "image_base": image_base,
                       "return_per_image": return_per_image,
                       "result_path": str(result_path)}
            last_error = "no attempt made"
            for attempt in range(2):
                proc = self._ensure_started()
                responses = self._responses
                reader_state = self._reader_state
                try:
                    proc.stdin.write(json.dumps(request) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, ValueError, AttributeError) as exc:
                    last_error = f"image worker pipe broken: {exc}"
                    self._kill()
                    continue

                response = None
                deadline = time.monotonic() + self.timeout
                while True:
                    if reader_state["failed"]:
                        # The reader thread died (broken/undecodable pipe):
                        # fail now instead of waiting out the whole timeout.
                        self._kill()
                        raise ImageWorkerError(
                            "image worker reader thread failed"
                            " (worker killed; task will be retried)")
                    try:
                        candidate = responses.get(timeout=1.0)
                    except queue.Empty:
                        candidate = None
                    if candidate is not None:
                        if candidate.get("request_id") != request_id:
                            # Stale/late reply (e.g. from a killed worker):
                            # discard it and keep waiting for ours.
                            logger.warning(
                                "discarding stale image worker response {}",
                                candidate.get("request_id"))
                            if time.monotonic() >= deadline:
                                self._kill()
                                raise ImageWorkerError(
                                    f"image worker timed out after {self.timeout}s "
                                    "(worker killed; task will be retried)")
                            continue
                        response = candidate
                        break
                    if proc.poll() is not None:
                        # worker died without answering this request
                        break
                    if time.monotonic() >= deadline:
                        self._kill()
                        raise ImageWorkerError(
                            f"image worker timed out after {self.timeout}s "
                            "(worker killed; task will be retried)")
                if response is None:
                    last_error = (f"image worker exited (code {proc.poll()}) "
                                  "before responding")
                    self._kill()
                    continue
                if response.get("status") != "ok":
                    raise ImageWorkerError(response.get("error", "unknown worker error"))
                if response.get("result_file"):
                    path = Path(response["result_file"])
                    try:
                        result = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        raise ImageWorkerError(
                            f"image worker result file unreadable: {exc}") from exc
                    finally:
                        try:
                            path.unlink()
                        except OSError:
                            pass
                    return result
                if "result" not in response:
                    raise ImageWorkerError(
                        "image worker response has neither result nor result_file")
                return response["result"]
            raise ImageWorkerError(f"{last_error}; giving up after restart")
