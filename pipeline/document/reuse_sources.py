"""Find reusable document outputs in existing historical and live task stores.

Task databases and append-only result files are the reuse store: new completed
API tasks become sources automatically, including after worker restarts.
"""
import json
import os
from pathlib import Path
import sqlite3

from loguru import logger

from .extraction_reuse import ExtractionReuse
from .source_paths import resolve_source


def same_pdf(left, right):
    """Compare actual bytes, avoiding filename collisions and stat-only hits."""
    left, right = Path(left), Path(right)
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            block = a.read(1024 * 1024)
            if block != b.read(1024 * 1024):
                return False
            if not block:
                return True


def load_document_reuse(output, language):
    databases = os.getenv('CHEMREX_REUSE_TASK_DATABASES', '')
    if not databases:
        return None
    reuse = ExtractionReuse([])
    try:
        identity = json.loads((output / 'source_identity.json').read_text(encoding='utf-8'))
        pdf = resolve_source(identity['pdf_path'])
    except (OSError, ValueError, KeyError) as exc:
        logger.warning('document reuse unavailable for {}: {}', output.name, exc)
        return None
    seen = {output.resolve()}
    loaded = 0
    for database in dict.fromkeys(databases.split(os.pathsep)):
        path = Path(database).expanduser().resolve()
        if not path.is_file():
            continue
        try:
            conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=10)
            try:
                # The original filename narrows candidates. Bytes below establish
                # identity even when different patents share the same filename.
                rows = conn.execute(
                    "SELECT pdf_path, output_dir FROM pdf_task WHERE status='DONE' "
                    "AND language=? AND substr(pdf_path, -?)=? ORDER BY updated_at ASC, id ASC",
                    (language, len('/' + pdf.name), '/' + pdf.name)).fetchall()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning('cannot read reuse task database {}: {}', path, exc)
            continue
        for source_pdf, folder in rows:
            try:
                folder = resolve_source(folder)
                if folder in seen:
                    continue
                seen.add(folder)
                if not same_pdf(pdf, resolve_source(source_pdf)):
                    continue
                source = ExtractionReuse.load(folder)
                # Same resume policy as a single output folder: newest saved
                # extraction wins, while all normalization candidates survive.
                reuse.overlay_recent(source)
                loaded += bool(source.inputs)
            except (OSError, ValueError, KeyError) as exc:
                # Bad/moved history never prevents a fresh extraction.
                logger.warning('cannot reuse document {}: {}', folder, exc)
    logger.info('document {}: loaded {} historical/API reuse source(s)', output.name, loaded)
    return reuse if reuse.inputs else None
