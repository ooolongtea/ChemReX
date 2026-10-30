"""Read append-only chunk results using the current source fingerprint."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def legacy_completed_record(record: Dict[str, Any]) -> bool:
    """Old runners saved successful module output without step reactions.

    Such records must be extracted again, even when the source fingerprint
    still matches. An empty non_reaction list remains a valid classification.
    """
    if record.get("status") not in ("ok", "needs_review", "non_reaction"):
        return False
    reactions = record.get("reactions")
    return not isinstance(reactions, list) or (record.get("status") != "non_reaction" and not reactions)


def latest_by_id(path: Path, fingerprints: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    path = Path(path)
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not record.get("id"):
                continue
            chunk_id = record["id"]
            if fingerprints is not None and (
                chunk_id not in fingerprints
                or record.get("chunk_fingerprint") != fingerprints[chunk_id]
            ):
                continue
            latest[chunk_id] = record
    return latest
