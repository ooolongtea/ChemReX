"""CLI for Module 3 — Data standardization.

Usage:
    # from extraction result JSON file
    python -m pipeline.standardize.run_standardize --input extraction.json --output standardized.json

    # inline (read extraction output JSON on stdin)
    cat extraction.json | python -m pipeline.standardize.run_standardize --output standardized.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

from loguru import logger

from .standardizer import Standardizer


def main():
    parser = argparse.ArgumentParser(description="ChemReX Module 3 — Data standardization")
    parser.add_argument("--input", type=str, default=None, help="Extraction result JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            extraction_result = json.load(f)
    else:
        extraction_result = json.load(sys.stdin)

    std = Standardizer()
    result = std.standardize(extraction_result)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
