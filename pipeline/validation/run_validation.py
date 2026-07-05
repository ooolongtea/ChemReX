"""CLI for Module 5 — Reaction consistency validation.

Usage:
    # validate a single standardized record (output of M3 standardize)
    python -m pipeline.validation.run_validation --ner-result standardized.json --output validated.json

    # read on stdin
    cat standardized.json | python -m pipeline.validation.run_validation --output validated.json

Input: a standardized NER+Action record (dict) or a JSON list of such records.
Output: the record(s) with an added ``rxn_validation`` block (and ``reactant_side_participants``
when a conservative patch is accepted). Temp/Time/Yield/Action are never modified.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Union

from loguru import logger

from .validator import ReactionValidator


def _looks_like_jsonl(content: str) -> bool:
    first = content.lstrip()
    if not first:
        return False
    try:
        json.loads(first.split("\n")[0])
        return "\n" in content.strip()
    except (json.JSONDecodeError, IndexError):
        return False


def _run_one(validator: ReactionValidator, record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        logger.error(f"expected a record dict, got {type(record).__name__}; skipping")
        return record
    if "ner" not in record:
        logger.warning("record has no 'ner' field — validation may be uninformative")
    try:
        return validator.validate_record(record)
    except Exception as exc:
        logger.error(f"validate_record failed: {exc}")
        record = dict(record)
        record["rxn_validation"] = {
            "status": "needs_review",
            "diagnosis": "validation_error",
            "needs_human_review": True,
            "reason": str(exc),
        }
        return record


def main():
    parser = argparse.ArgumentParser(description="ChemReX Module 5 — Reaction validation")
    parser.add_argument("--ner-result", type=str, default=None,
                        help="Standardized NER+Action record JSON (dict or list of dicts)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.ner_result:
        with open(args.ner_result, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if args.ner_result.endswith(".jsonl") or _looks_like_jsonl(content):
            data = []
            for lineno, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSONL line {lineno}: {e}")
        else:
            data = json.loads(content)
    else:
        data = json.load(sys.stdin)

    validator = ReactionValidator()

    if isinstance(data, list):
        results: List[Any] = [_run_one(validator, r) for r in data]
        out = results
    elif isinstance(data, dict):
        out = _run_one(validator, data)
    else:
        parser.error(f"input must be a JSON object or array, got {type(data).__name__}")

    payload = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if args.output.endswith(".jsonl") and isinstance(out, list):
            with open(args.output, "w", encoding="utf-8") as f:
                for rec in out:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(payload)
        logger.info(f"Saved: {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
