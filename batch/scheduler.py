"""Batch scheduler — claims tasks from the store and drives them to DONE.

Concurrency model
─────────────────
- ``task_workers`` documents run at once (each in its own thread);
- the resident MinerU service serialises GPU parsing server-side;
- the resident image worker serialises OCR/OCSR mining via a client lock;
- each document runs ``chunk_workers`` concurrent LLM calls (M1→M3→M4).

Resume
──────
Stage-level, file-driven (mirrors run_document.py), and tied to the *parse
identity* — an md5 of the parsed content_list recorded in
``parse_result.json``:
- parsing   — pdf_prepared/parse_result.json exists, still points at an
  existing content_list/image dir AND the recorded fingerprint matches the
  current file → skip MinerU; otherwise re-parse and rename the previous
  ``image_result.json`` / ``results.jsonl`` / ``final_product.json`` to
  ``*.stale`` so no stage consumes results of a superseded parse;
- mining    — image_result.json exists, parses AND carries the same parse
  fingerprint → skip the image worker;
- extracting— results.jsonl append-only, ok records skipped; error records
  retried when the task itself is being retried (attempt_count > 1, or the
  file already holds error records). A task whose chunks ended in error —
  or whose final product failed — is never marked DONE: it raises so the
  task is auto-retried and, once attempts are exhausted, stored as FAILED.

Lifecycle
─────────
- crashed scheduler: RUNNING tasks claimed by another run are requeued
  immediately on start and, in daemon mode, whenever the queue is idle
  (store.recover_stuck). Lock expiry alone never requeues a task of the
  current run (long stages do not heartbeat; re-entering a live task would
  duplicate work and corrupt its output files);
- FAILED tasks with attempts remaining are requeued at startup when
  auto-retry is on: the retry plan (backoff → requeue) lives in memory and
  a stop/crash inside the window used to strand them (review round-2 H5);
- every state write carries the claiming run_id as a fencing token, so a
  late write from a stale attempt cannot overwrite a reset/re-claimed
  task's state (review round-2 H6);
- failed tasks: stored as FAILED with the error; auto-requeued while
  attempts remain (``--no-auto-retry`` to disable), or manually via
  ``run_batch.py retry`` (``retry --errors`` requeues DONE tasks whose
  results are incomplete).
"""

import hashlib
import json
import os
import signal
import threading
import time
import uuid
import faulthandler
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from . import config as batch_config
from .document_processor import DocumentProcessor
from .image_client import ImageWorkerClient
from .mineru_client import MineruClient
from .store import TaskStore, STOPPING


class TaskStopped(Exception):
    """Cooperative task control, never an extraction failure."""


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _content_fingerprint(path: Optional[str]) -> Optional[str]:
    """Content hash of the parsed content_list — the parse identity that the
    stage caches (image_result.json, results.jsonl) are tied to."""
    if not path:
        return None
    digest = hashlib.md5()
    try:
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return f"md5:{digest.hexdigest()}"


class BatchScheduler:
    def __init__(
        self,
        store: TaskStore,
        mineru: MineruClient,
        image_client: Optional[ImageWorkerClient],
        task_workers: int = 2,
        chunk_workers: int = 16,
        poll_interval: float = 10.0,
        daemon: bool = False,
        max_tasks: Optional[int] = None,
        mineru_backend: str = "vlm-auto-engine",
        max_attempts: int = 3,
        lock_expire_seconds: int = 7200,
        auto_retry: bool = True,
        no_image: bool = False,
        skip_final_product: bool = False,
        no_global_dict: bool = False,
        retry_backoff: float = 5.0,
        no_delivery: bool = False,
        task_uuids=None,
    ):
        self.store = store
        self.mineru = mineru
        self.image_client = image_client
        self.task_workers = max(1, task_workers)
        self.chunk_workers = max(1, chunk_workers)
        self.poll_interval = poll_interval
        self.daemon = daemon
        self.max_tasks = max_tasks
        self.mineru_backend = mineru_backend
        self.max_attempts = max_attempts
        self.lock_expire_seconds = lock_expire_seconds
        self.auto_retry = auto_retry
        self.no_image = no_image
        self.skip_final_product = skip_final_product
        self.no_global_dict = no_global_dict
        self.retry_backoff = retry_backoff
        self.no_delivery = no_delivery
        self.task_uuids = list(dict.fromkeys(task_uuids)) if task_uuids is not None else None
        from pipeline.document.business import filename_ids
        self._document_ids = filename_ids([
            (Path(task["output_dir"]).name, Path(task["pdf_path"]).name)
            for task in store.list_tasks()])
        self.run_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._stop = threading.Event()
        self._processors: Dict[str, DocumentProcessor] = {}
        self._processors_lock = threading.Lock()
        self._progress_at: Dict[int, float] = {}
        self._stall_reported: Dict[int, float] = {}

    # ── setup ──
    def _processor(self, language: str) -> DocumentProcessor:
        with self._processors_lock:
            processor = self._processors.get(language)
            if processor is None:
                preset = batch_config.BATCH_EXTRACTION_PRESETS.get(
                    language, batch_config.BATCH_EXTRACTION_PRESET)
                processor = DocumentProcessor(
                    language=language,
                    chunk_workers=self.chunk_workers,
                    skip_final_product=self.skip_final_product,
                    global_dict_enabled=not self.no_global_dict,
                    no_delivery=self.no_delivery,
                    preset=preset,
                )
                self._processors[language] = processor
            return processor

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            # Keep the handler async-signal-safe: set the flag first and avoid
            # logger (non-reentrant lock; a signal may interrupt a log call).
            if self._stop.is_set():
                os.write(2, b"second signal - exiting immediately\n")
                os._exit(130)
            self._stop.set()
            os.write(2, f"signal {signum} - finishing in-flight tasks, then "
                        "exiting (send again to force)\n".encode())

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass  # not the main thread (tests)

    # ── main loop ──
    def _wait_mineru_healthy(self) -> bool:
        """Block until the MinerU service answers.

        Daemon mode waits forever (logging each retry) so the scheduler
        survives a service restart; one-shot mode fails fast. Returns False
        when a stop was requested while still waiting (review M7)."""
        if not self.daemon:
            health = self.mineru.wait_healthy(attempts=1)
            logger.info("MinerU service ready: {}", health)
            return True
        while not self._stop.is_set():
            try:
                health = self.mineru.wait_healthy(attempts=1)
                logger.info("MinerU service ready: {}", health)
                return True
            except Exception as exc:
                logger.warning("MinerU service not ready ({}); retrying", exc)
                self._stop.wait(5.0)
        return False

    def _check_lock_config(self) -> None:
        """Warn when a single stage can outlive the lock (review H6)."""
        mineru_timeout = getattr(self.mineru, "timeout", 0) or 0
        image_timeout = (getattr(self.image_client, "timeout", 0) or 0) \
            if self.image_client is not None else 0
        floor = max(mineru_timeout, 2 * image_timeout + 300)
        if floor and self.lock_expire_seconds <= floor:
            logger.warning(
                "BATCH_LOCK_EXPIRE ({}s) is not larger than the longest single"
                " stage ({}s = max(mineru timeout, 2×image timeout + margin));"
                " expiry-based recovery tooling may consider slow tasks stuck",
                self.lock_expire_seconds, floor)

    def run(self) -> int:
        if not self._wait_mineru_healthy():
            logger.info("stop requested before MinerU became healthy; exiting")
            return 0
        self.store.recover_controls(self.run_id)
        if self.auto_retry:
            # The retry plan (backoff → requeue) lives in memory; a stop or
            # crash inside the window used to strand FAILED tasks that still
            # had attempts left. Pick them up again on startup (review H5).
            requeued = (self.store.requeue_failed_with_attempts(self.max_attempts, uuids=self.task_uuids)
                        if self.task_uuids is not None else self.store.requeue_failed_with_attempts(self.max_attempts))
            if requeued:
                logger.info("requeued {} FAILED task(s) with attempts remaining",
                            requeued)
        self._recover_stuck("startup")
        self._check_lock_config()
        if self.image_client is not None:
            self.image_client.ensure_started()

        self._install_signal_handlers()
        claimed = 0
        inflight = []
        try:
            with ThreadPoolExecutor(max_workers=self.task_workers,
                                    thread_name_prefix="task") as pool:
                while not self._stop.is_set():
                    self._report_stalled_extraction()
                    completed = [f for f in inflight if f.done()]
                    for future in completed:
                        future.result()
                    inflight = [f for f in inflight if f not in completed]
                    if not inflight:
                        faulthandler.cancel_dump_traceback_later()
                    if self.max_tasks is not None and claimed >= self.max_tasks:
                        logger.info("reached --max-tasks {}", self.max_tasks)
                        break
                    if len(inflight) >= self.task_workers:
                        time.sleep(0.5)
                        continue
                    task = (self.store.claim_next(run_id=self.run_id, uuids=self.task_uuids)
                            if self.task_uuids is not None else self.store.claim_next(run_id=self.run_id))
                    if task is None:
                        if not self.daemon and not inflight:
                            break
                        if self.daemon:
                            # A crashed run may have left RUNNING tasks behind;
                            # recover them without waiting for a restart.
                            self._recover_stuck("idle")
                        time.sleep(self.poll_interval)
                        continue
                    claimed += 1
                    if not inflight:
                        self._arm_stall_diagnostics()
                    inflight.append(pool.submit(self._process_task, task))
                for future in inflight:
                    future.result()
        finally:
            faulthandler.cancel_dump_traceback_later()
            if self.image_client is not None:
                self.image_client.stop()
        counts = self.store.counts()
        logger.info("scheduler finished: {} tasks claimed, store: {}", claimed, counts)
        return claimed

    def _recover_stuck(self, context: str) -> None:
        recovered = self.store.recover_stuck(
            self.lock_expire_seconds, self.max_attempts,
            current_run_id=self.run_id, **({'uuids': self.task_uuids} if self.task_uuids is not None else {}))
        if recovered["requeued"] or recovered["failed"]:
            logger.info("recovered stuck tasks ({}): {}", context, recovered)

    def _report_stalled_extraction(self) -> None:
        """Expose Python stacks during a stall without racing an active writer."""
        now = time.time()
        limit = float(os.getenv('BATCH_EXTRACTION_STALL_SECONDS', '900'))
        for task_id, last in list(self._progress_at.items()):
            if now - last < limit or now - self._stall_reported.get(task_id, 0) < limit:
                continue
            self._stall_reported[task_id] = now
            logger.error('task {} has made no extraction progress for {:.0f}s; dumping thread stacks',
                         task_id, now - last)
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    @staticmethod
    def _arm_stall_diagnostics():
        # This timer runs in C even if Python's GIL is held by a stuck native
        # extension. Progress resets it; idle and shutdown cancel it.
        seconds = float(os.getenv('BATCH_EXTRACTION_STALL_SECONDS', '900'))
        faulthandler.dump_traceback_later(seconds, repeat=True, file=sys.stderr)

    # ── per-task pipeline ──
    def _process_task(self, task: Dict[str, Any]) -> None:
        task_id = task["id"]
        uuid = task["uuid"]
        run_id = task["run_id"]
        started = time.time()
        try:
            self._check_stop(task)
            self._resolve_language(task)
            self._check_stop(task)
            content_list_file, image_base, parse_fp = self._stage_parse(task)
            self._check_stop(task)
            self._resolve_language(task, content_list_file)
            self._check_stop(task)
            image_data = self._stage_mining(task, image_base, parse_fp)
            self._check_stop(task)
            self._stage_extracting(task, content_list_file, image_data)
            self._check_stop(task)
            elapsed = round(time.time() - started, 1)
            if not self.store.mark_done(task_id, elapsed, run_id=run_id):
                self._finish_stop(task)
            else:
                logger.info("task {} DONE in {}s", uuid, elapsed)
        except TaskStopped:
            self._finish_stop(task)
            logger.info("task {} safely stopped", uuid)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            elapsed = round(time.time() - started, 1)
            if (self.auto_retry and task["attempt_count"] < self.max_attempts
                    and not self._stop.is_set()):
                if not self.store.mark_failed(task_id, error, run_id=run_id, retry_pending=True):
                    self._finish_stop(task)
                    return
                delay = self._retry_delay(task["attempt_count"])
                logger.warning("task {} failed after {}s (attempt {}/{}) — retrying in "
                               "{:.0f}s: {}",
                               uuid, elapsed, task["attempt_count"], self.max_attempts,
                               delay, error[:300])
                # Leave the task FAILED (visible) during the backoff, then
                # requeue it; a stop request cancels the requeue.
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not self._stop.is_set():
                    if self._stop_requested(task):
                        self._finish_stop(task)
                        return
                    self._stop.wait(min(0.1, max(0, deadline - time.monotonic())))
                if not self._stop.is_set():
                    if not self._requeue(task_id, run_id=run_id):
                        self._finish_stop(task)
                elif self._stop_requested(task):
                    self._finish_stop(task)
                else:
                    self.store._execute(
                        "UPDATE pdf_task SET execution_active=0, run_id=NULL WHERE id=? AND run_id=? AND status='FAILED'",
                        (task_id, run_id))
            else:
                if not self.store.mark_failed(task_id, error, run_id=run_id):
                    self._finish_stop(task)
                logger.error("task {} FAILED after {}s: {}", uuid, elapsed, error[:500])

    def _stop_requested(self, task):
        current = self.store.get(task['uuid'])
        return (current is None or current['id'] != task['id']
                or current['status'] in STOPPING)

    def _check_stop(self, task):
        if self._stop_requested(task):
            raise TaskStopped()

    def _resolve_language(self, task, content_list_file=None):
        if task['language'] != 'auto':
            return
        from .language import detect_pdf_language, detect_parsed_language
        if content_list_file is None:
            language = detect_pdf_language(task['pdf_path'])
            source = 'pdf_text'
            if language is None:
                return
        else:
            language = detect_parsed_language(content_list_file)
            source = 'parsed_text' if language else 'fallback_en'
            language = language or 'en'
        if self.store.resolve_language(task['id'], language, source, task['run_id']):
            task.update(language=language, language_source=source)
            logger.info('task {} auto language={} source={}', task['uuid'], language, source)

    def _finish_stop(self, task):
        current = self.store.get(task['uuid'])
        if current and current['id'] == task['id'] and current['status'] in ('PAUSING', 'CANCELLING'):
            try:
                out = Path(task['output_dir'])
                out.mkdir(parents=True, exist_ok=True)
                document = self._read_json(out / 'document_result.json') or {}
                document.update(status='partial', reason='Safely stopped; resume to complete the document')
                _write_json(out / 'document_result.json', document)
                summary = self._read_json(out / 'summary.json') or {}
                summary['extraction_outcome'] = 'partial'
                counts = summary.get('result_status') or {}
                if counts:
                    self.store.update_progress(task['id'], sum(int(counts.get(s, 0))
                        for s in ('ok', 'needs_review', 'non_reaction')), run_id=task['run_id'])
                if (out / 'results.jsonl').is_file() and not self.no_delivery:
                    from pipeline.document.delivery import try_export_document
                    summary['delivery'] = try_export_document(out)
                _write_json(out / 'summary.json', summary)
            except Exception as exc:
                logger.exception('task {} partial export failed', task['uuid'])
                self.store._execute('UPDATE pdf_task SET error=? WHERE id=? AND run_id=?',
                                    (f'Partial result export failed: {exc}', task['id'], task['run_id']))
        self.store.settle_stop(task['id'], task['run_id'])

    def _retry_delay(self, attempt_count: int) -> float:
        """Exponential backoff before an automatic requeue (0 disables it)."""
        if self.retry_backoff <= 0:
            return 0.0
        return min(self.retry_backoff * (2 ** max(0, attempt_count - 1)), 60.0)

    def _requeue(self, task_id: int, run_id: Optional[str] = None) -> bool:
        sql = ("UPDATE pdf_task SET status='PENDING', stage='', run_id=NULL, execution_active=0, retry_pending=0,"
               " error=NULL, chunks_done=0, chunks_total=NULL, elapsed_seconds=NULL,"
               " updated_at=? WHERE id=?")
        params: List[Any] = [time.time(), task_id]
        if run_id is not None:
            # Fenced: a late requeue from a stale attempt must not pull a
            # task back to PENDING after it was reset or re-claimed (review H6).
            sql += " AND status='FAILED' AND retry_pending=1 AND run_id=?"
            params.append(run_id)
        return self.store._execute(sql, tuple(params)).rowcount == 1

    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _mark_stale(out: Path, *names: str) -> None:
        """Rename superseded stage caches to ``*.stale`` (kept for
        traceability) instead of letting later stages consume results that
        belong to a previous parse (review H2)."""
        for name in names:
            path = out / name
            if not path.exists():
                continue
            stale = path.with_name(name + ".stale")
            try:
                path.replace(stale)
                logger.info("marked {} stale (superseded by a re-parse)", stale.name)
            except OSError as exc:
                logger.warning("could not mark {} stale: {}", path, exc)

    def _stage_parse(self, task: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
        """MinerU parsing (skipped when a valid parse_result.json exists).
        Returns (content_list_file, image_base, parse_fingerprint)."""
        out = Path(task["output_dir"])
        prepared_dir = out / "pdf_prepared"
        marker = prepared_dir / "parse_result.json"
        if marker.is_file():
            payload = self._read_json(marker)
            content_list = payload.get("content_list_file") if isinstance(payload, dict) else None
            image_base = payload.get("image_base") if isinstance(payload, dict) else None
            from pipeline.document.source_paths import resolve_source
            try:
                content_list = str(resolve_source(content_list)) if content_list else None
                image_base = str(resolve_source(image_base)) if image_base else None
            except FileNotFoundError:
                pass
            content_ok = bool(content_list) and Path(content_list).is_file()
            images_ok = not image_base or Path(image_base).is_dir()
            # Parse identity: the recorded fingerprint must match the current
            # content_list, otherwise the file was rewritten by a re-parse
            # (or the marker predates fingerprints) and every downstream
            # cache is suspect (review H2).
            fingerprint_ok = (content_ok and payload.get("content_list_fingerprint")
                              == _content_fingerprint(content_list))
            if content_ok and images_ok and fingerprint_ok:
                logger.info("task {} parsing: reused {}", task["uuid"], marker.name)
                return content_list, image_base, payload.get("content_list_fingerprint")
            logger.warning("task {} parsing: stale {} (content_list={}, image_base={},"
                           " fingerprint={}); re-parsing", task["uuid"], marker.name,
                           content_ok, images_ok, fingerprint_ok)
        prepared_dir.mkdir(parents=True, exist_ok=True)
        self.store.update_stage(task["id"], "parsing", run_id=task["run_id"])
        payload = self.mineru.parse(
            task["pdf_path"], str(prepared_dir), backend=self.mineru_backend,
            # MinerU's ch OCR supports Chinese and English. Unknown scans
            # must not be forced through the English-only OCR configuration.
            language='zh' if task['language'] == 'auto' else task['language'])
        # Downstream caches belong to the previous parse (different text or
        # image set); retire them before the new marker makes the parse
        # reusable, so a crash between the two still re-parses (review H2).
        self._mark_stale(out, "image_result.json", "results.jsonl",
                         "final_product.json")
        payload["content_list_fingerprint"] = _content_fingerprint(
            payload["content_list_file"])
        _write_json(marker, payload)
        return (payload["content_list_file"], payload.get("image_base"),
                payload["content_list_fingerprint"])

    def _stage_mining(self, task: Dict[str, Any], image_base: Optional[str],
                      parse_fingerprint: Optional[str]) -> Dict[str, Any]:
        out = Path(task["output_dir"])
        result_path = out / "image_result.json"
        if result_path.is_file():
            from pipeline.versions import IMAGE_PIPELINE_VERSION
            payload = self._read_json(result_path)
            if isinstance(payload, dict):
                from pipeline.document.image_reuse import known_image_defects, reject_confirmed_candidates
                defects = known_image_defects(task.get('uuid'))
                _, rejected = reject_confirmed_candidates(payload, defects)
                # Reuse only results of the SAME parse: a cached dictionary
                # from an older parse can misassign SMILES to the new
                # chunks (review H2).
                if (parse_fingerprint is not None
                        and payload.get("parse_fingerprint") == parse_fingerprint
                        and payload.get("image_pipeline_version") == IMAGE_PIPELINE_VERSION and not rejected):
                    logger.info("task {} mining: reused {}", task["uuid"], result_path.name)
                    return payload
                if (parse_fingerprint is not None and payload.get('parse_fingerprint') == parse_fingerprint
                        and self.image_client is not None and not self.no_image and image_base):
                    from pipeline.document.image_reuse import repair_missing_molecules
                    repaired = repair_missing_molecules(payload, image_base, self.image_client.process,
                                                        known_defects=defects)
                    if repaired is not None:
                        _write_json(result_path, repaired)
                        return repaired
                logger.warning("task {} mining: {} does not match the current parse;"
                               " re-running", task["uuid"], result_path.name)
            else:
                logger.warning("task {} mining: unreadable {}; re-running",
                               task["uuid"], result_path.name)
        if self.no_image or self.image_client is None or not image_base:
            return {}
        out.mkdir(parents=True, exist_ok=True)
        self.store.update_stage(task["id"], "mining", run_id=task["run_id"])
        image_data = self.image_client.process(
            image_base, return_per_image=True,
            result_path=out / ".image_worker_result.json")
        from pipeline.document.image_reuse import known_image_defects, reject_confirmed_candidates
        image_data, _ = reject_confirmed_candidates(image_data, known_image_defects(task.get('uuid')))
        from pipeline.versions import IMAGE_PIPELINE_VERSION
        image_data["parse_fingerprint"] = parse_fingerprint
        image_data["image_pipeline_version"] = IMAGE_PIPELINE_VERSION
        if image_data.get("failed_images"):
            logger.warning("task {} image mining had {} failed image stages",
                           task["uuid"], len(image_data["failed_images"]))
        _write_json(result_path, image_data)
        logger.info("task {} mining: dict={} entries", task["uuid"],
                    len(image_data.get("dict_result") or {}))
        return image_data

    def _stage_extracting(self, task: Dict[str, Any], content_list_file: str,
                          image_data: Dict[str, Any]) -> None:
        out = Path(task["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        processor = self._processor(task["language"])
        # Daemon mode may receive additional tasks after startup. Qualify new
        # duplicate names without renaming identities already selected.
        with self._processors_lock:
            if out.name not in self._document_ids:
                from pipeline.document.business import filename_ids
                names = filename_ids([(Path(t["output_dir"]).name, Path(t["pdf_path"]).name)
                                      for t in self.store.list_tasks()])
                for label, document_id in names.items():
                    self._document_ids.setdefault(label, document_id)
        # Persist the original PDF for offline delivery rebuilds as well as
        # default exports. The business ID is based on bytes, not task UUID.
        _write_json(out / "source_identity.json", {
            "pdf_path": task["pdf_path"],
            "document_id": self._document_ids[out.name],
        })
        prepared = processor.prepare(content_list_file, out)
        self.store.update_stage(task["id"], "extracting",
                                chunks_total=prepared.manifest["with_procedure"],
                                run_id=task["run_id"])
        task_id = task["id"]

        self._progress_at[task_id] = time.time()

        def progress(done: int, total: int) -> None:
            now = time.time()
            if done >= total or now - self._progress_at.get(task_id, 0) >= 5.0:
                self._progress_at[task_id] = now
                self._arm_stall_diagnostics()
                self.store.update_progress(task_id, done, run_id=task["run_id"])

        # Retry error records not only on attempt ≥ 2 but also whenever the
        # results file already holds non-ok records — e.g. after
        # `retry --errors --reset-attempts`, whose first claim would
        # otherwise skip the error chunks and waste an attempt.
        retry_errors = (task["attempt_count"] > 1
                        or TaskStore._results_have_errors(task["output_dir"]))
        try:
            summary = processor.extract(prepared, out, image_data,
                                        retry_errors=retry_errors, progress_cb=progress,
                                        stop_requested=lambda: self._stop_requested(task))
        finally:
            self._progress_at.pop(task_id, None)
            self._stall_reported.pop(task_id, None)
        status_counts = summary.get("result_status") or {}
        completed = sum(int(status_counts.get(key, 0))
                        for key in ("ok", "needs_review", "non_reaction"))
        failed = sum(int(n) for n in status_counts.values()) - completed
        expected = int(summary.get("with_procedure") or 0)
        failed += max(0, expected - completed - failed)
        logger.info("task {} extracting: {}", task["uuid"], status_counts)
        if failed:
            # A per-chunk failure must not end as a silent DONE: raising here
            # triggers auto-retry (retry_errors on the next attempt) and, once
            # attempts are exhausted, a visible FAILED task.
            raise RuntimeError(
                f"{failed} of {expected or '?'} procedure chunk(s) "
                "failed during extraction; see results.jsonl")
        final_product = summary.get("final_product")
        if isinstance(final_product, dict) and final_product.get("error"):
            # Same contract for the document-level product: a transient
            # TargetExtractor failure must fail the task so the auto-retry
            # regenerates it, instead of a DONE task with no
            # final_product.json (review H3).
            raise RuntimeError(
                f"final product extraction failed: {final_product['error']}")
