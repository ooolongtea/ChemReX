#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-evaluate BERTScore (and other Action metrics) on existing predictions.jsonl
using the new multilingual backbone FacebookAI/xlm-roberta-large.

Only re-runs directories that contain ACTION samples. NER-only dirs (reactionseek)
are skipped because they have no BERTScore.

Usage:
    python reeval_bertscore.py
    python reeval_bertscore.py --only external_Qwen3-8B external_qwen3-lora-08
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import eval as eval_module  # noqa: E402

RESULTS_DIR = (EVAL_DIR.parent / "results").resolve()

# Directories containing ACTION samples (NER+ACTION, 8298 lines each).
ACTION_DIRS = [
    "external_Qwen3-8B_20260630-115257",
    "external_qwen3-lora-08_20260630-121627",
    "external_gpt-4o_20260131-205535",
    "external_qwen-plus_20260201-145935",
    "external_deepseek-v3_20260201-155435",
    "base_chemdfm-v1.5-8b_20260128-154402",
    "base_ChemLLM-7B-SFT_20260130-155158",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only re-eval these dir names (subset of ACTION_DIRS)")
    args = parser.parse_args()

    targets = args.only if args.only else ACTION_DIRS

    print("=" * 90)
    print("Re-evaluate BERTScore with FacebookAI/xlm-roberta-large")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Targets ({len(targets)}):")
    for t in targets:
        print(f"  - {t}")
    print("=" * 90)

    # Configure eval module: disable dir traversal, we feed files one-by-one.
    eval_module.CONFIG["data_dir"] = None
    eval_module.CONFIG["task_filter"] = "all"
    eval_module.CONFIG["language_filter"] = "all"
    eval_module.CONFIG["split_by_language"] = True
    # Keep default out_name so it overwrites eval_metrics_language.json in-place.

    succeeded = []
    failed = []

    for name in targets:
        pred_path = RESULTS_DIR / name / "predictions.jsonl"
        if not pred_path.exists():
            print(f"\n[SKIP] {name}: predictions.jsonl not found at {pred_path}")
            failed.append((name, "no predictions.jsonl"))
            continue

        print(f"\n{'#' * 90}")
        print(f"# START: {name}")
        print(f"# pred: {pred_path}  ({pred_path.stat().st_size/1e6:.1f} MB)")
        print(f"# time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 90}")

        eval_module.CONFIG["data"] = [str(pred_path)]

        try:
            eval_module.main()
            succeeded.append(name)
            print(f"[OK] {name} done @ {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed.append((name, str(e)))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Succeeded ({len(succeeded)}):")
    for s in succeeded:
        print(f"  [OK] {s}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name, err in failed:
            print(f"  [FAIL] {name}: {err}")
    else:
        print("All succeeded.")
    print("DONE_REEVAL_BERTSCORE")


if __name__ == "__main__":
    main()
