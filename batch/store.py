"""SQLite-backed task store for large-batch PDF extraction.

Status machine::

    PENDING ──claim──▶ RUNNING (stage: parsing → mining → extracting) ──▶ DONE
        ▲                                                    │
        └────────────── recover_stuck / retry ◀──────────────┴──▶ FAILED

- ``claim_next`` is atomic (BEGIN IMMEDIATE + status guard), safe for a
  single scheduler process; the WAL journal lets CLI processes (add / list /
  retry) access the same database concurrently.
- The connection is shared across scheduler worker threads, so every
  statement (and the whole ``claim_next`` transaction) is serialised by an
  RLock — no other thread can join an open transaction.
- ``recover_stuck`` requeues tasks left RUNNING by a crashed scheduler. When
  called with the current ``run_id`` (a live scheduler), only tasks claimed by
  a *different* run are requeued — immediately, regardless of lock age. Lock
  expiry alone never requeues a task of the current run: long parse/mining
  stages do not heartbeat, and re-entering them would run two attempts on the
  same output dir. Tasks that exhausted MAX_ATTEMPTS go to FAILED instead.
- State writes accept the claiming ``run_id`` as a fencing token: an UPDATE
  scoped with ``status='RUNNING' AND run_id=?`` can no longer be applied by a
  stale attempt after the task was reset or re-claimed.
- Stage-level resume is file-driven (content_list.json / image_result.json /
  results.jsonl under the task output dir), mirroring run_document.py.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STOPPING = ("PAUSING", "CANCELLING", "DELETING")
STATUSES = ("PENDING", "RUNNING", "DONE", "FAILED", "DEFERRED", "PAUSED", "CANCELLED", *STOPPING)
STAGES = ("parsing", "mining", "extracting")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pdf_task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    pdf_path TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    requested_language TEXT NOT NULL DEFAULT 'en',
    language_source TEXT NOT NULL DEFAULT 'explicit',
    status TEXT NOT NULL DEFAULT 'PENDING',
    stage TEXT NOT NULL DEFAULT '',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    locked_at REAL,
    error TEXT,
    chunks_total INTEGER,
    chunks_done INTEGER NOT NULL DEFAULT 0,
    elapsed_seconds REAL,
    run_id TEXT,
    execution_active INTEGER NOT NULL DEFAULT 0,
    retry_pending INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pdf_task_status ON pdf_task(status, created_at);
"""


def build_uuid(pdf_path: Path, digest_len: int = 8) -> str:
    """Stable task id: pdf stem + short hash of the resolved path (idempotent add).

    *digest_len* is raised by callers on the (rare) collision between two
    different PDFs sharing a stem — 8 hex chars is only 32 bits."""
    import hashlib

    resolved = str(pdf_path.expanduser().resolve())
    digest = hashlib.md5(resolved.encode("utf-8")).hexdigest()[:digest_len]
    return f"{pdf_path.stem}_{digest}"


class TaskStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0,
                                     isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the initial schema (in-place upgrade).

        Runs inside BEGIN IMMEDIATE: two processes starting on the same legacy
        database would otherwise race between PRAGMA table_info and ALTER
        TABLE and one would hit a duplicate-column error."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                columns = {row["name"] for row in
                           self._conn.execute("PRAGMA table_info(pdf_task)").fetchall()}
                if "run_id" not in columns:
                    self._conn.execute("ALTER TABLE pdf_task ADD COLUMN run_id TEXT")
                if 'requested_language' not in columns:
                    self._conn.execute("ALTER TABLE pdf_task ADD COLUMN requested_language TEXT NOT NULL DEFAULT 'en'")
                    self._conn.execute('UPDATE pdf_task SET requested_language=language')
                if 'language_source' not in columns:
                    self._conn.execute("ALTER TABLE pdf_task ADD COLUMN language_source TEXT NOT NULL DEFAULT 'explicit'")
                if "execution_active" not in columns:
                    self._conn.execute("ALTER TABLE pdf_task ADD COLUMN execution_active INTEGER NOT NULL DEFAULT 0")
                    self._conn.execute("UPDATE pdf_task SET execution_active=1 WHERE status='RUNNING'")
                if "retry_pending" not in columns:
                    self._conn.execute("ALTER TABLE pdf_task ADD COLUMN retry_pending INTEGER NOT NULL DEFAULT 0")
                    # Preserve legacy restart recovery once. New failures carry
                    # the scheduler's explicit decision rather than inferring it.
                    from .config import MAX_ATTEMPTS
                    self._conn.execute(
                        "UPDATE pdf_task SET retry_pending=1 WHERE status='FAILED' AND attempt_count < ?",
                        (MAX_ATTEMPTS,))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── internal helpers ──
    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    def _execute(self, sql: str, params: Tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            for attempt in range(3):
                try:
                    return self._conn.execute(sql, params)
                except sqlite3.OperationalError as exc:
                    if "locked" in str(exc).lower() and attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    raise

    # ── task creation / query ──
    def add_task(self, uuid: str, pdf_path: str, output_dir: str,
                 language: str = "en") -> Tuple[Optional[int], bool]:
        """Insert a task; returns (task_id, created). Existing uuid → (id, False)."""
        now = time.time()
        cursor = self._execute(
            "INSERT OR IGNORE INTO pdf_task (uuid, pdf_path, output_dir, language,"
            " requested_language, language_source, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (uuid, pdf_path, output_dir, language, language,
             'pending' if language == 'auto' else 'explicit', "PENDING", now, now),
        )
        row = self._execute("SELECT id FROM pdf_task WHERE uuid=?", (uuid,)).fetchone()
        if row is None:
            return None, False
        return row["id"], cursor.rowcount == 1

    def get(self, uuid: str) -> Optional[Dict[str, Any]]:
        row = self._execute("SELECT * FROM pdf_task WHERE uuid=?", (uuid,)).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(self, status: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM pdf_task"
        params: Tuple = ()
        if status:
            if status not in STATUSES:
                raise ValueError(f"status must be one of {STATUSES}")
            sql += " WHERE status=?"
            params = (status,)
        sql += " ORDER BY created_at, id"
        if limit:
            sql += " LIMIT ?"
            params = params + (limit,)
        return [self._row_to_task(r) for r in self._execute(sql, params).fetchall()]

    def counts(self) -> Dict[str, int]:
        rows = self._execute(
            "SELECT status, COUNT(*) AS n FROM pdf_task GROUP BY status").fetchall()
        result = {status: 0 for status in STATUSES}
        for row in rows:
            result[row["status"]] = row["n"]
        result["TOTAL"] = sum(result.values())
        return result

    # ── scheduler state transitions ──
    @staticmethod
    def _fence(sql: str, params: List[Any],
               run_id: Optional[str], progress: bool = False) -> Tuple[str, List[Any]]:
        """Scope an UPDATE to the claim that owns the task (stale-writer
        fencing): a late write from a previous attempt/run cannot overwrite
        the state of a task that was reset or re-claimed in the meantime."""
        if run_id is not None:
            sql += (" AND status IN ('RUNNING','PAUSING','CANCELLING','DELETING') AND run_id=?"
                    if progress else " AND status='RUNNING' AND run_id=?")
            params = params + [run_id]
        return sql, params

    def claim_next(self, run_id: Optional[str] = None, uuids=None) -> Optional[Dict[str, Any]]:
        """Atomically move the oldest PENDING task to RUNNING.

        The whole SELECT/UPDATE/COMMIT sequence runs under the connection
        lock so worker-thread statements cannot join the open transaction.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if uuids is not None and not uuids:
                    self._conn.execute('COMMIT')
                    return None
                scope = ' AND uuid IN (' + ','.join('?' for _ in uuids) + ')' if uuids is not None else ''
                row = self._conn.execute(
                    "SELECT * FROM pdf_task WHERE status='PENDING'" + scope +
                    " ORDER BY created_at, id LIMIT 1", tuple(uuids or ())).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None
                now = time.time()
                self._conn.execute(
                    "UPDATE pdf_task SET status='RUNNING', stage='', run_id=?, execution_active=1, retry_pending=0,"
                    " attempt_count=attempt_count+1, locked_at=?, updated_at=?"
                    " WHERE id=? AND status='PENDING'",
                    (run_id, now, now, row["id"]))
                task = self._row_to_task(row)
                task["status"] = "RUNNING"
                task["run_id"] = run_id
                task["execution_active"] = 1
                task["retry_pending"] = 0
                task["attempt_count"] = row["attempt_count"] + 1
                task["locked_at"] = now
                self._conn.execute("COMMIT")
                return task
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def update_stage(self, task_id: int, stage: str,
                     chunks_total: Optional[int] = None,
                     run_id: Optional[str] = None) -> None:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        sets = ["stage=?", "updated_at=?", "locked_at=?"]
        params: List[Any] = [stage, time.time(), time.time()]
        if chunks_total is not None:
            sets.append("chunks_total=?")
            params.append(chunks_total)
        sql = f"UPDATE pdf_task SET {', '.join(sets)} WHERE id=?"
        params.append(task_id)
        sql, params = self._fence(sql, params, run_id, progress=True)
        self._execute(sql, tuple(params))

    def update_progress(self, task_id: int, chunks_done: int,
                        run_id: Optional[str] = None) -> None:
        sql = ("UPDATE pdf_task SET chunks_done=?, updated_at=?, locked_at=?"
               " WHERE id=?")
        params: List[Any] = [chunks_done, time.time(), time.time(), task_id]
        sql, params = self._fence(sql, params, run_id, progress=True)
        self._execute(sql, tuple(params))

    def resolve_language(self, task_id: int, language: str, source: str, run_id: str) -> bool:
        if language not in ('en', 'zh'):
            raise ValueError('Resolved language must be en or zh')
        sql = ("UPDATE pdf_task SET language=?, language_source=?, updated_at=? "
               "WHERE id=? AND requested_language='auto' AND language='auto'")
        sql, params = self._fence(sql, [language, source, time.time(), task_id], run_id, progress=True)
        return self._execute(sql, tuple(params)).rowcount == 1

    def mark_done(self, task_id: int, elapsed_seconds: float,
                  run_id: Optional[str] = None) -> bool:
        sql = ("UPDATE pdf_task SET status='DONE', stage='', error=NULL, execution_active=0, retry_pending=0,"
               " elapsed_seconds=?, updated_at=? WHERE id=?")
        params: List[Any] = [elapsed_seconds, time.time(), task_id]
        sql, params = self._fence(sql, params, run_id)
        return self._execute(sql, tuple(params)).rowcount == 1

    def mark_failed(self, task_id: int, error: str,
                    run_id: Optional[str] = None, retry_pending: bool = False) -> bool:
        sql = ("UPDATE pdf_task SET status='FAILED', stage='', error=?, retry_pending=?, execution_active=?,"
               " updated_at=? WHERE id=?")
        params: List[Any] = [error[:4000], int(retry_pending), int(retry_pending), time.time(), task_id]
        sql, params = self._fence(sql, params, run_id)
        return self._execute(sql, tuple(params)).rowcount == 1

    def request_stop(self, uuid: str, action: str) -> Dict[str, Any]:
        """Serialize controls with claims and recovery; active work acknowledges later."""
        target = {"pause": "PAUSING", "cancel": "CANCELLING", "delete": "DELETING"}[action]
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT * FROM pdf_task WHERE uuid=?", (uuid,)).fetchone()
                if row is None:
                    raise KeyError(uuid)
                status = row['status']
                if (status == 'DELETING' and action != 'delete'
                        or status in ('CANCELLING', 'CANCELLED') and action == 'pause'
                        or status == 'DONE' and action != 'delete'):
                    raise ValueError(f"Cannot {action} a {status} task")
                if not row['execution_active']:
                    target = {'pause': 'PAUSED', 'cancel': 'CANCELLED', 'delete': 'DELETING'}[action]
                self._conn.execute(
                    "UPDATE pdf_task SET status=?, retry_pending=0, updated_at=? WHERE id=?",
                    (target, time.time(), row['id']))
                result = dict(self._conn.execute("SELECT * FROM pdf_task WHERE id=?", (row['id'],)).fetchone())
                self._conn.execute("COMMIT")
                return result
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def settle_stop(self, task_id: int, run_id: str) -> None:
        """Called only after all work and file writes for this execution have drained."""
        self._execute(
            "UPDATE pdf_task SET status=CASE status WHEN 'PAUSING' THEN 'PAUSED' "
            "WHEN 'CANCELLING' THEN 'CANCELLED' ELSE status END, execution_active=0,"
            " retry_pending=0, run_id=NULL, updated_at=? WHERE id=? AND run_id=? "
            "AND status IN ('PAUSING','CANCELLING','DELETING')",
            (time.time(), task_id, run_id))

    def recover_controls(self, current_run_id: str) -> None:
        """The caller owns the scheduler lock; previous scheduler work has exited."""
        self._execute(
            "UPDATE pdf_task SET status=CASE status WHEN 'PAUSING' THEN 'PAUSED' "
            "WHEN 'CANCELLING' THEN 'CANCELLED' ELSE status END, execution_active=0,"
            " retry_pending=0, run_id=NULL, updated_at=? "
            "WHERE status IN ('PAUSING','CANCELLING','DELETING') "
            "AND execution_active=1 AND (run_id IS NULL OR run_id != ?)",
            (time.time(), current_run_id))

    def resume(self, uuid: str) -> Dict[str, Any]:
        with self._lock:
            self._conn.execute('BEGIN IMMEDIATE')
            try:
                row = self._conn.execute('SELECT * FROM pdf_task WHERE uuid=?', (uuid,)).fetchone()
                if row is None:
                    raise KeyError(uuid)
                if row['status'] != 'PENDING':
                    if row['execution_active'] or row['status'] not in ('PAUSED','CANCELLED','FAILED'):
                        raise ValueError(f"Cannot resume a {row['status']} task")
                    if not Path(row['pdf_path']).is_file():
                        raise ValueError('Original PDF is missing; upload it again')
                    self._conn.execute(
                        f"UPDATE pdf_task SET {self._REQUEUE_SET}, attempt_count=0, updated_at=? WHERE id=?",
                        (time.time(), row['id']))
                result = dict(self._conn.execute('SELECT * FROM pdf_task WHERE uuid=?', (uuid,)).fetchone())
                self._conn.execute('COMMIT')
                return result
            except Exception:
                self._conn.execute('ROLLBACK')
                raise

    # ── recovery / retry ──
    # Common SET clause for every requeue: stale progress must not survive a
    # move back to PENDING (dump_task would show it as if it were current).
    _REQUEUE_SET = ("status='PENDING', stage='', run_id=NULL, error=NULL, execution_active=0, retry_pending=0,"
                    " chunks_done=0, chunks_total=NULL, elapsed_seconds=NULL")

    def recover_stuck(self, lock_expire_seconds: int, max_attempts: int,
                      current_run_id: Optional[str] = None, uuids=None) -> Dict[str, int]:
        """Requeue RUNNING tasks left behind by a crashed scheduler.

        With *current_run_id* (a live scheduler), only tasks claimed by a
        different run are recovered — immediately, regardless of lock age.
        Lock expiry alone must not requeue a task of the current run: the
        long parse/mining stages do not heartbeat, and re-entering a task
        that is still executing would run two attempts on the same output
        dir (review H6). Expiry-based recovery remains for callers without
        a run context (tooling/tests).
        """
        if current_run_id is not None:
            rows = self._execute(
                "SELECT id, uuid, attempt_count, run_id FROM pdf_task WHERE status='RUNNING'"
                " AND (run_id IS NULL OR run_id != ?)",
                (current_run_id,)).fetchall()
        else:
            cutoff = time.time() - lock_expire_seconds
            rows = self._execute(
                "SELECT id, uuid, attempt_count, run_id FROM pdf_task WHERE status='RUNNING'"
                " AND (locked_at IS NULL OR locked_at < ?)",
                (cutoff,)).fetchall()
        requeued = failed = 0
        for row in rows:
            if uuids is not None and row['uuid'] not in uuids:
                continue
            if row["attempt_count"] >= max_attempts:
                failed += self._execute(
                    "UPDATE pdf_task SET status='FAILED', error='recovered as stuck and attempts exhausted',"
                    " execution_active=0, retry_pending=0, updated_at=? "
                    "WHERE id=? AND status='RUNNING' AND run_id IS ?",
                    (time.time(), row['id'], row['run_id'])).rowcount
            else:
                requeued += self._execute(
                    f"UPDATE pdf_task SET {self._REQUEUE_SET}, updated_at=?"
                    " WHERE id=? AND status='RUNNING' AND run_id IS ?",
                    (time.time(), row["id"], row['run_id'])).rowcount
        return {"requeued": requeued, "failed": failed}

    def requeue_failed_with_attempts(self, max_attempts: int, uuids=None) -> int:
        """Requeue FAILED tasks that still have attempts left.

        The scheduler's automatic retry plan lives in memory (backoff, then
        requeue); a stop or crash inside that window used to strand such
        tasks as FAILED forever. Called at scheduler startup so a new run
        picks them up again (review H5). The last error is kept as a note
        until the task completes or fails again."""
        if uuids is not None and not uuids:
            return 0
        scope = ' AND uuid IN (' + ','.join('?' for _ in uuids) + ')' if uuids is not None else ''
        cursor = self._execute(
            "UPDATE pdf_task SET status='PENDING', stage='', run_id=NULL, execution_active=0, retry_pending=0,"
            " chunks_done=0, chunks_total=NULL, elapsed_seconds=NULL, updated_at=?"
            " WHERE status='FAILED' AND retry_pending=1 AND attempt_count < ?" + scope,
            (time.time(), max_attempts) + tuple(uuids or ()))
        return cursor.rowcount

    def retry_selected(self, uuids, reset_attempts=False):
        """Requeue exactly the supplied completed/failed tasks atomically."""
        uuids = list(dict.fromkeys(uuids))
        with self._lock:
            self._conn.execute('BEGIN IMMEDIATE')
            try:
                rows = []
                for uuid in uuids:
                    row = self._conn.execute('SELECT id, status, execution_active FROM pdf_task WHERE uuid=?', (uuid,)).fetchone()
                    if row is None:
                        raise KeyError('unknown uuid: ' + uuid)
                    if row['status'] == 'RUNNING' or row['execution_active']:
                        raise ValueError('cannot requeue running task: ' + uuid)
                    rows.append(row)
                moved = 0
                for row in rows:
                    if row['status'] in ('DONE', 'FAILED'):
                        moved += self._conn.execute(
                            f"UPDATE pdf_task SET {self._REQUEUE_SET},"
                            f"{'attempt_count=0,' if reset_attempts else ''} updated_at=?"
                            " WHERE id=? AND status IN ('DONE','FAILED')", (time.time(), row['id'])).rowcount
                self._conn.execute('COMMIT')
                return moved
            except Exception:
                self._conn.execute('ROLLBACK')
                raise

    def retry(self, uuid: Optional[str] = None, all_failed: bool = False,
              reset_attempts: bool = False) -> int:
        """Move FAILED (or a specific uuid) tasks back to PENDING."""
        now = time.time()
        if uuid:
            row = self._execute("SELECT id, status, execution_active FROM pdf_task WHERE uuid=?",
                                (uuid,)).fetchone()
            if row is None:
                raise KeyError(f"unknown uuid: {uuid}")
            if row["status"] not in ("FAILED", "PENDING"):
                raise ValueError(
                    f"task {uuid} is {row['status']}; only FAILED tasks retry "
                    "(use `retry --errors` to requeue DONE tasks with error records)")
            if row["status"] == "PENDING":
                return 0
            if row['execution_active']:
                raise ValueError('Task still has an active retry; pause or cancel it first')
            cursor = self._execute(
                f"UPDATE pdf_task SET {self._REQUEUE_SET},"
                f"{'attempt_count=0,' if reset_attempts else ''} updated_at=?"
                " WHERE id=? AND status='FAILED' AND execution_active=0",
                (now, row["id"]))
            return cursor.rowcount
        if not all_failed:
            raise ValueError("retry requires a uuid, --failed or --errors")
        cursor = self._execute(
            f"UPDATE pdf_task SET {self._REQUEUE_SET},"
            f"{'attempt_count=0,' if reset_attempts else ''} updated_at=?"
            " WHERE status='FAILED' AND execution_active=0", (now,))
        return cursor.rowcount

    def retry_done_with_errors(self, reset_attempts: bool = False) -> int:
        """Requeue DONE tasks whose results are incomplete.

        Catches a chunk whose latest record is incomplete and a recorded
        final-product error — scheduler-side guards against results silently
        lost before those failures were propagated (see doc/batch-code-review.md
        H1 and round-2 H3); a no-op for healthy tasks.
        """
        rows = self._execute(
            "SELECT id, output_dir FROM pdf_task WHERE status='DONE'").fetchall()
        moved = 0
        now = time.time()
        for row in rows:
            if not (self._results_have_errors(row["output_dir"])
                    or self._final_product_failed(row["output_dir"])):
                continue
            self._execute(
                f"UPDATE pdf_task SET {self._REQUEUE_SET},"
                f"{'attempt_count=0,' if reset_attempts else ''} updated_at=?"
                " WHERE id=?",
                (now, row["id"]))
            moved += 1
        return moved

    def retry_done_without_procedures(self, reset_attempts: bool = False) -> int:
        """Requeue legacy DONE documents whose saved segmentation found no procedure.

        The parsed PDF and image mining cache are retained. A restarted
        scheduler can re-segment them with the current rules at low GPU cost.
        """
        rows = self._execute(
            "SELECT id, output_dir FROM pdf_task WHERE status='DONE'").fetchall()
        moved = 0
        now = time.time()
        for row in rows:
            path = Path(row["output_dir"]) / "summary.json"
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(summary, dict) or summary.get("with_procedure") != 0:
                continue
            cursor = self._execute(
                f"UPDATE pdf_task SET {self._REQUEUE_SET},"
                f"{'attempt_count=0,' if reset_attempts else ''} updated_at=?"
                " WHERE id=? AND status='DONE'", (now, row["id"]))
            moved += cursor.rowcount
        return moved

    def retry_done_outdated(self, segmentation_version: str,
                            limit: Optional[int] = None,
                            reset_attempts: bool = False) -> int:
        """Requeue DONE PDFs segmented under an older rule version."""
        if limit is not None and limit <= 0:
            return 0
        rows = self._execute(
            "SELECT id, output_dir FROM pdf_task WHERE status='DONE' ORDER BY id").fetchall()
        moved = 0
        now = time.time()
        for row in rows:
            path = Path(row["output_dir"]) / "summary.json"
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
            if isinstance(summary, dict) and summary.get("segmentation_version") == segmentation_version:
                continue
            cursor = self._execute(
                f"UPDATE pdf_task SET {self._REQUEUE_SET},"
                f"{'attempt_count=0,' if reset_attempts else ''} updated_at=?"
                " WHERE id=? AND status='DONE'", (now, row["id"]))
            moved += cursor.rowcount
            if limit is not None and moved >= limit:
                break
        return moved

    @staticmethod
    def _current_chunk_ids(output_dir: str) -> Optional[dict]:
        """Current chunk id → fingerprint (None when segmentation is unknown)."""
        path = Path(output_dir) / "chunks.jsonl"
        if not path.is_file():
            return None
        ids = {}
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk, dict) and chunk.get("id"):
                        ids[chunk["id"]] = chunk.get("chunk_fingerprint")
        except OSError:
            return None
        return ids

    @classmethod
    def _results_have_errors(cls, output_dir: str) -> bool:
        """True when the latest record of any current chunk failed extraction.

        Records whose id no longer exists in chunks.jsonl (a re-parse changed
        the segmentation) are ignored: their errors can never be fixed by
        re-running and must not keep the task FAILED forever (review H1).
        Missing output directories and legacy completed records without reaction
        output are incomplete too. Current needs_review/non_reaction records
        remain completed, including explicitly classified non-reactions."""
        from pipeline.document.results import legacy_completed_record

        if not Path(output_dir).is_dir():
            return True
        path = Path(output_dir) / "results.jsonl"
        current = cls._current_chunk_ids(output_dir)
        if not path.is_file():
            return bool(current)
        latest: Dict[str, Dict[str, Any]] = {}
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    chunk_id = record.get("id")
                    if chunk_id and (current is None or chunk_id in current):
                        expected = current.get(chunk_id) if current is not None else None
                        if expected and record.get("chunk_fingerprint") != expected:
                            continue
                        latest[chunk_id] = record
        except OSError:
            return False
        return (any(record.get("status") not in ("ok", "needs_review", "non_reaction")
                    or legacy_completed_record(record) for record in latest.values())
                or (current is not None and any(cid not in latest for cid in current)))

    @staticmethod
    def _final_product_failed(output_dir: str) -> bool:
        """True when summary.json records a final-product error (review H3)."""
        path = Path(output_dir) / "summary.json"
        if not path.is_file():
            return False
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        fp = summary.get("final_product") if isinstance(summary, dict) else None
        return isinstance(fp, dict) and bool(fp.get("error"))

    def reset_stuck(self, lock_expire_seconds: int = 0) -> int:
        """Force-requeue every RUNNING task (manual recovery, no expiry check)."""
        cursor = self._execute(
            f"UPDATE pdf_task SET {self._REQUEUE_SET}, updated_at=?"
            " WHERE status='RUNNING'", (time.time(),))
        return cursor.rowcount


def dump_task(task: Dict[str, Any]) -> str:
    """One-line human-readable rendering for `list` output."""
    progress = ""
    if task.get("chunks_total"):
        progress = f" {task.get('chunks_done', 0)}/{task['chunks_total']}"
    error = f" | {task['error'][:80]}" if task.get("error") else ""
    elapsed = f" | {task['elapsed_seconds']:.0f}s" if task.get("elapsed_seconds") else ""
    return (f"{task['uuid']}\t{task['status']}\t{task.get('stage') or '-'}{progress}"
            f"\ttry {task['attempt_count']}{elapsed}{error}")


def dump_tasks_json(tasks: List[Dict[str, Any]]) -> str:
    return json.dumps(tasks, ensure_ascii=False, indent=2)
