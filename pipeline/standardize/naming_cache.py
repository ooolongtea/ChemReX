"""Persistent compound-name standardization cache shared by batch workers.

Rows are keyed by ``(name, language, config_version)``. A row with a SMILES is
a positive hit; a row whose SMILES is NULL is a negative hit, so names that
cannot be resolved are not re-sent to the LLM and the CIR/PubChem/OPSIN
cascade on every occurrence.

SQLite in WAL mode keeps one scheduler's chunk threads and separate scheduler
processes from blocking each other. Cache failures are logged and degrade to a
no-op; they never fail document processing.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from .names import string_names

DEFAULT_PATH = Path(
    "/data/patents/chemrex_batch/naming-cache.sqlite3")
_COLUMNS = ("standard_name", "smiles", "cxsmiles", "stereochemistry",
            "resolution_status", "resolution_trace", "naming_raw")
_CHUNK = 500
# Both language processors use this module. A per-instance lock is not enough:
# the stalled scheduler had one thread in sqlite3.connect and another in
# Connection.close, while even sleeping/network threads could not resume.
# Serialize these short DB sessions, including connection destruction.
_CONNECTION_LOCK = threading.RLock()


def _dump(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _load(value: Optional[str], default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class NamingCache:
    """SQLite-backed name cache; disabled mode is a no-op."""

    def __init__(self, path: Optional[Path] = None, enabled: bool = True):
        self.path = Path(path) if path else DEFAULT_PATH
        self.enabled = enabled
        self._ready = False
        self._lock = threading.Lock()
        self._warned = False

    @classmethod
    def from_env(cls) -> "NamingCache":
        raw = os.getenv("CHEMREX_NAMING_CACHE", "1").strip().lower()
        enabled = raw not in ("0", "false", "no", "off")
        path = os.getenv("CHEMREX_NAMING_CACHE_PATH") or DEFAULT_PATH
        return cls(path=path, enabled=enabled)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
        except Exception:
            conn.close()
            raise
        return conn

    @contextmanager
    def _connection(self):
        with _CONNECTION_LOCK:
            conn = self._connect()
            try:
                yield conn
            finally:
                conn.close()

    def _ensure_ready(self) -> bool:
        if not self.enabled:
            return False
        if self._ready:
            return True
        with self._lock:
            if self._ready:
                return True
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self._connection() as conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS naming_cache ("
                        " name TEXT NOT NULL,"
                        " language TEXT NOT NULL DEFAULT '',"
                        " config_version TEXT NOT NULL,"
                        " standard_name TEXT,"
                        " smiles TEXT,"
                        " cxsmiles TEXT,"
                        " stereochemistry TEXT,"
                        " resolution_status TEXT,"
                        " resolution_trace TEXT,"
                        " naming_raw TEXT,"
                        " updated_at REAL NOT NULL,"
                        " PRIMARY KEY (name, language, config_version))")
                    conn.commit()
                self._ready = True
                logger.debug("naming cache ready: {}", self.path)
                return True
            except (OSError, sqlite3.Error) as exc:
                self.enabled = False
                if not self._warned:
                    logger.warning("naming cache disabled ({}: {})", type(exc).__name__, exc)
                    self._warned = True
                return False

    def get_many(self, names: Iterable[str], language: str,
                 config_version: str) -> Dict[str, Dict[str, Any]]:
        unique = string_names(names if isinstance(names, (str, list, tuple)) else list(names))
        if not unique or not self._ensure_ready():
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        try:
            with self._connection() as conn:
                for start in range(0, len(unique), _CHUNK):
                    batch = unique[start:start + _CHUNK]
                    placeholders = ",".join("?" * len(batch))
                    rows = conn.execute(
                        "SELECT name, standard_name, smiles, cxsmiles, stereochemistry,"
                        " resolution_status, resolution_trace, naming_raw"
                        " FROM naming_cache"
                        f" WHERE config_version=? AND language=? AND name IN ({placeholders})",
                        (config_version, language or "", *batch))
                    for row in rows:
                        result[row[0]] = {
                            "standard_name": row[1],
                            "smiles": row[2],
                            "cxsmiles": row[3],
                            "stereochemistry": _load(row[4], None),
                            "resolution_status": row[5],
                            "resolution_trace": _load(row[6], []),
                            "naming_raw": row[7],
                        }
        except (OSError, sqlite3.Error) as exc:
            if not self._warned:
                logger.warning("naming cache read failed ({}: {})", type(exc).__name__, exc)
                self._warned = True
            return {}
        return result

    def put_many(self, values: Dict[str, Dict[str, Any]], language: str,
                 config_version: str) -> None:
        if not values or not self._ensure_ready():
            return
        rows = []
        now = time.time()
        for name, value in values.items():
            rows.append((
                name, language or "", config_version,
                value.get("standard_name"), value.get("smiles"), value.get("cxsmiles"),
                _dump(value.get("stereochemistry")),
                value.get("resolution_status"), _dump(value.get("resolution_trace")),
                value.get("naming_raw"), now))
        try:
            with self._connection() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO naming_cache"
                    " (name, language, config_version, standard_name, smiles, cxsmiles,"
                    "  stereochemistry, resolution_status, resolution_trace, naming_raw, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
                conn.commit()
        except (OSError, sqlite3.Error) as exc:
            if not self._warned:
                logger.warning("naming cache write failed ({}: {})", type(exc).__name__, exc)
                self._warned = True
