"""CLI for coreference resolution: compound standardization + ID→SMILES mapping.

Usage:
    # Single record (backward compatible)
    python -m pipeline.coref.run_coref --ner-result ner.json --dict-result dict.json --output coref.json

    # Batch with per-record dict_result from M2 output (JSONL input/output)
    python -m pipeline.coref.run_coref \
        --ner-result data/outputs/m1/m1_predictions.jsonl \
        --dict-result data/outputs/m2/m2_image_result.json \
        --output data/outputs/m3/m3_coref.jsonl \
        --max-workers 8

Inputs:
    --ner-result : JSON (dict/list) or JSONL. Each record should have a 'ner' field
                   (or '_prediction.ner' from M1 output). Records with 'openexp_index'
                   are matched to per-record dict_result.
    --dict-result: One of:
                   1. M2 output JSON with 'per_image_detections' (per-record dict)
                   2. Flat dict {id_key: smiles} (global, applied to all records)
                   3. Mapping {openexp_index_str: {id_key: smiles}}
    --max-workers: Concurrency for batch processing (default: 1)
Missing dict_result → standardize only and skip ID mapping.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from loguru import logger

from pipeline.chemical_quality import image_label_evidence

from .coref_resolver import CorefResolver


def _run_one(resolver: CorefResolver, ner_record: Dict[str, Any], dict_result: Dict[str, str]) -> Dict[str, Any]:
    if not isinstance(ner_record, dict):
        logger.error(f"expected a record dict, got {type(ner_record).__name__}; skipping")
        return ner_record
    if "ner" not in ner_record:
        logger.warning("ner-result record has no 'ner' field")
    try:
        payload = dict(ner_record)
        payload["dict_result"] = dict_result
        return resolver.resolve(payload)
    except Exception as exc:
        logger.error(f"coref failed: {exc}")
        return {
            "id_smiles_mapping": {},
            "unresolved_identifiers": [],
            "ner": ner_record.get("ner", {}),
            "summary": {"total_identifiers": 0, "resolved": 0, "unresolved": 0, "dict_entries": len(dict_result)},
            "error": str(exc),
        }


def _normalize_ner_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw record from M1 output to have 'ner' at top level.

    M1 output format: {text, language, openexp_index, _prediction: {ner, action, raw}, ...}
    Coref expects: {ner, action, openexp_index, ...}
    """
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


def _load_ner_records(path: str) -> List[Dict[str, Any]]:
    """Load NER records from JSON (dict/list) or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    is_jsonl = path.endswith(".jsonl") or _looks_like_jsonl(content)
    if is_jsonl:
        records = []
        for lineno, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSONL line {lineno}: {e}")
        return records

    data = json.loads(content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unexpected NER data type: {type(data).__name__}")


def _looks_like_jsonl(content: str) -> bool:
    first = content.lstrip()
    if not first:
        return False
    try:
        json.loads(first.split("\n")[0])
        return True
    except (json.JSONDecodeError, IndexError):
        return False


def _extract_index_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"openexp_(\d+)", fname)
    return int(m.group(1)) if m else None


def _load_dict_per_record(path: str) -> Dict[int, Dict[str, str]]:
    """Build {openexp_index: {id: smiles}} from dict-result file.

    Supports:
    1. M2 output JSON with 'per_image_detections' list
    2. Mapping {openexp_index_str: {id: smiles}}
    3. Flat dict {id: smiles} → returns {} (caller falls back to global)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "per_image_detections" in data:
        mapping: Dict[int, Dict[str, str]] = {}
        images_by_index = {}
        for item in data["per_image_detections"]:
            fname = item.get("image_file_name", "")
            idx = _extract_index_from_filename(fname)
            if idx is not None:
                images_by_index.setdefault(idx, []).append(item)
        for idx, items in images_by_index.items():
            mapping[idx] = image_label_evidence(items)['dict_result']
        return mapping

    if isinstance(data, dict):
        mapping = {}
        for k, v in data.items():
            try:
                idx = int(k)
                if isinstance(v, dict):
                    mapping[idx] = {str(kk): str(vv) for kk, vv in v.items()}
            except (ValueError, TypeError):
                continue
        if mapping:
            return mapping

    return {}


def _load_global_dict(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "per_image_detections" in data:
            return {}
        if "dict_result" in data and isinstance(data["dict_result"], dict):
            return {str(k): str(v) for k, v in data["dict_result"].items()}
        return {str(k): str(v) for k, v in data.items() if not isinstance(v, (dict, list))}
    return {}


def main():
    parser = argparse.ArgumentParser(description="ChemReX Module 4 — Coreference resolution")
    parser.add_argument("--ner-result", type=str, required=True,
                        help="NER+Action record JSON/JSONL (dict, list, or .jsonl)")
    parser.add_argument("--dict-result", type=str, required=True,
                        help="Image dictionary JSON (M2 output, flat {id:smiles}, or {index: {id:smiles}})")
    parser.add_argument("--output", type=str, default=None, help="Output JSON or JSONL file")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Concurrency for batch processing (default: 1)")
    args = parser.parse_args()

    records = _load_ner_records(args.ner_result)
    if not records:
        parser.error("no NER records loaded")
    logger.info(f"Loaded {len(records)} NER records")

    dict_per_record = _load_dict_per_record(args.dict_result)
    global_dict = _load_global_dict(args.dict_result) if not dict_per_record else {}
    if dict_per_record:
        logger.info(f"Per-record dict_result: {len(dict_per_record)} entries")
    elif global_dict:
        logger.info(f"Global dict_result: {len(global_dict)} entries")
    else:
        logger.warning("empty dict_result — coref will standardize only")

    resolver = CorefResolver()

    def _get_dict_for_record(rec: Dict[str, Any]) -> Dict[str, str]:
        idx = rec.get("openexp_index")
        if idx is not None and dict_per_record:
            return dict_per_record.get(int(idx), {})
        return global_dict

    is_jsonl_output = args.output and args.output.endswith(".jsonl")

    if args.max_workers <= 1:
        results: List[Dict[str, Any]] = []
        for i, raw_rec in enumerate(records):
            rec = _normalize_ner_record(raw_rec)
            dr = _get_dict_for_record(rec)
            result = _run_one(resolver, rec, dr)
            if "openexp_index" not in result and "openexp_index" in raw_rec:
                result["openexp_index"] = raw_rec["openexp_index"]
            results.append(result)
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(records)}")
    else:
        results = []
        completed = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_to_raw = {}
            for raw_rec in records:
                rec = _normalize_ner_record(raw_rec)
                dr = _get_dict_for_record(rec)
                fut = pool.submit(_run_one, resolver, rec, dr)
                future_to_raw[fut] = raw_rec
            for fut in as_completed(future_to_raw):
                raw_rec = future_to_raw[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    logger.error(f"coref failed: {e}")
                    result = {"ner": {}, "error": str(e)}
                if "openexp_index" not in result and "openexp_index" in raw_rec:
                    result["openexp_index"] = raw_rec["openexp_index"]
                results.append(result)
                completed += 1
                if completed % 50 == 0:
                    logger.info(f"Progress: {completed}/{len(records)}")

    if is_jsonl_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(results)} records -> {args.output}")
    else:
        if len(results) == 1:
            out = results[0]
        else:
            out = results
        payload = json.dumps(out, ensure_ascii=False, indent=2)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(payload)
            logger.info(f"Saved: {args.output}")
        else:
            print(payload)


if __name__ == "__main__":
    main()
