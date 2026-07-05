#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper to run ChemReX eval.py on a specific ReactionSeek predictions.jsonl.
"""

import argparse
import sys
from pathlib import Path

# Add ChemReX/eval to path so we can import eval.py
EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))

import eval as eval_module  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ReactionSeek predictions.jsonl")
    parser.add_argument("--pred", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--task_filter", default="ner", choices=["all", "ner", "action"])
    parser.add_argument("--language_filter", default="all", choices=["all", "zh", "en"])
    args = parser.parse_args()

    eval_module.CONFIG["data_dir"] = None
    eval_module.CONFIG["data"] = [args.pred]
    eval_module.CONFIG["task_filter"] = args.task_filter
    eval_module.CONFIG["language_filter"] = args.language_filter
    eval_module.CONFIG["split_by_language"] = True
    if args.language_filter in ("zh", "en"):
        eval_module.CONFIG["out_name"] = f"eval_metrics_language_{args.language_filter}.json"

    eval_module.main()


if __name__ == "__main__":
    main()
