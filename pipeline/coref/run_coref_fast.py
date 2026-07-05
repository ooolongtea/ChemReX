"""Fast M3 coref — direct identifier matching (no LLM, no standardization).

For OpenExp eval data, text identifiers (e.g. "1a", "2a") directly match
M2 dict_result keys. This script skips LLM standardization and LLM coref
mapping, doing direct string matching instead.

Usage:
    python -m pipeline.coref.run_coref_fast \
        --ner-result data/outputs/m1/m1_predictions.jsonl \
        --dict-result data/outputs/m2/m2_image_result.json \
        --output data/outputs/m3/m3_coref.jsonl \
        --max-workers 1
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from loguru import logger

from .coref_resolver import (
    _COMPOUND_FIELDS,
    _compound_name,
    _is_identifier_compound,
    _looks_like_identifier,
)


_PLACEHOLDER_RE = re.compile(r"^\d+[a-z]?$", re.IGNORECASE)


def _is_placeholder_identifier(name: str) -> bool:
    """Recognize placeholder codes like '1a', '2b', '3', '10c'."""
    s = (name or "").strip()
    if not s:
        return False
    if _PLACEHOLDER_RE.match(s):
        return True
    return _looks_like_identifier(s)


def _is_identifier_compound_fast(compound: Any) -> bool:
    if isinstance(compound, dict):
        std = (compound.get("standard_name") or "").strip()
        if std:
            return std.upper() == "ID"
        return _is_placeholder_identifier(_normalize_id(str(compound.get("name") or "")))
    return _is_placeholder_identifier(_normalize_id(str(compound)))


def _extract_index_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"openexp_(\d+)", fname)
    return int(m.group(1)) if m else None


def _load_dict_per_record(path: str) -> Dict[int, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[int, Dict[str, str]] = {}
    if isinstance(data, dict) and "per_image_detections" in data:
        for item in data["per_image_detections"]:
            fname = item.get("image_file_name", "")
            idx = _extract_index_from_filename(fname)
            if idx is not None:
                mapping[idx] = item.get("dict_result") or {}
    elif isinstance(data, dict):
        for k, v in data.items():
            try:
                idx = int(k)
                if isinstance(v, dict):
                    mapping[idx] = {str(kk): str(vv) for kk, vv in v.items()}
            except (ValueError, TypeError):
                continue
    return mapping


def _load_ner_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _normalize_ner_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "ner" in raw:
        return raw
    pred = raw.get("_prediction")
    if isinstance(pred, dict) and "ner" in pred:
        record = dict(pred)
        for k in ("openexp_index", "reference_style", "language"):
            if k in raw:
                record[k] = raw[k]
        return record
    return raw


def _collect_identifiers(ner: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    seen = set()
    for field in _COMPOUND_FIELDS:
        raw = ner.get(field, []) or []
        if isinstance(raw, str):
            raw = [raw] if raw.strip() else []
        for comp in (raw if isinstance(raw, list) else [raw]):
            if _is_identifier_compound_fast(comp):
                name = _compound_name(comp)
                if name and name not in seen:
                    seen.add(name)
                    ids.append(name)
    return ids


_ID_PREFIXES = ("compound", "cmpd", "intermediate", "reactant", "product", "example")


def _normalize_id(s: str) -> str:
    s = (s or "").lower().strip()
    changed = True
    while changed:
        changed = False
        for prefix in _ID_PREFIXES:
            new = re.sub(rf"^{re.escape(prefix)}[\s.:_-]*", "", s)
            if new != s:
                s = new
                changed = True
    s = re.sub(r"[.:_\-\s<>\[\](){}]+", "", s)
    return s.strip()


def _direct_match(
    text_identifiers: List[str],
    dict_result: Dict[str, str],
) -> Dict[str, str]:
    """Direct matching of text identifiers to dict_result keys, with normalization."""
    mapping: Dict[str, str] = {}
    norm_to_dict = {_normalize_id(k): k for k in dict_result.keys()}
    for tid in text_identifiers:
        norm_tid = _normalize_id(tid)
        if norm_tid and norm_tid in norm_to_dict:
            mapping[tid] = norm_to_dict[norm_tid]
    return mapping


def _run_one_fast(
    ner_record: Dict[str, Any],
    dict_result: Dict[str, str],
) -> Dict[str, Any]:
    ner = ner_record.get("ner", {}) or {}
    action = ner_record.get("action", [])
    language = ner_record.get("language")
    task = ner_record.get("task", "both")

    if not dict_result:
        return {
            "ner": ner,
            "action": action,
            "language": language,
            "task": task,
            "id_smiles_mapping": {},
            "unresolved_identifiers": _collect_identifiers(ner),
            "standardize_status": "skipped",
            "summary": {"total_identifiers": 0, "resolved": 0, "unresolved": 0, "dict_entries": 0},
        }

    identifiers = _collect_identifiers(ner)
    mapping = _direct_match(identifiers, dict_result)

    id_smiles_mapping = {
        tid: dict_result[dk]
        for tid, dk in mapping.items()
        if dk in dict_result and dict_result[dk]
    }
    unresolved = [tid for tid in identifiers if tid not in id_smiles_mapping]

    return {
        "ner": ner,
        "action": action,
        "language": language,
        "task": task,
        "id_smiles_mapping": id_smiles_mapping,
        "unresolved_identifiers": unresolved,
        "standardize_status": "direct_match",
        "summary": {
            "total_identifiers": len(identifiers),
            "resolved": len(id_smiles_mapping),
            "unresolved": len(unresolved),
            "dict_entries": len(dict_result),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner-result", type=str, required=True)
    parser.add_argument("--dict-result", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    records = _load_ner_records(args.ner_result)
    if not records:
        parser.error("no NER records loaded")
    logger.info(f"Loaded {len(records)} NER records")

    dict_per_record = _load_dict_per_record(args.dict_result)
    logger.info(f"Per-record dict_result: {len(dict_per_record)} entries")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, raw_rec in enumerate(records):
            rec = _normalize_ner_record(raw_rec)
            idx = raw_rec.get("openexp_index")
            dr = dict_per_record.get(int(idx), {}) if idx is not None else {}
            result = _run_one_fast(rec, dr)
            if "openexp_index" not in result and "openexp_index" in raw_rec:
                result["openexp_index"] = raw_rec["openexp_index"]
            if "reference_style" not in result and "reference_style" in raw_rec:
                result["reference_style"] = raw_rec["reference_style"]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(records)}")

    logger.info(f"Saved {written} records -> {args.output}")


if __name__ == "__main__":
    main()
