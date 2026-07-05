#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recompute eval_metrics.json for all predictions.jsonl under ChemReX/results
using the canonical eval.py logic.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Use HF mirror (modelscope-backed) to download BERTScore models when HuggingFace
# is not directly reachable.
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add ChemReX/eval to path so we can import eval.py
EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))

import eval as eval_module  # noqa: E402


def identify_task_type(sample: Dict[str, Any]) -> str:
    """Mirrors evaluate.py logic: prefer _task_type_eval, then task_type."""
    if "_task_type_eval" in sample and sample["_task_type_eval"]:
        v = str(sample["_task_type_eval"]).strip().lower()
        if v in ("ner", "action"):
            return v
    if "task_type" in sample:
        v = sample["task_type"]
        if isinstance(v, int):
            return "ner" if v == 0 else "action"
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("0", "ner"):
                return "ner"
            if vv in ("1", "action"):
                return "action"
    return "ner"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] skip line {line_no}: {e}")
    return rows


def recompute_one(pred_path: Path) -> Dict[str, Any]:
    rows = load_jsonl(str(pred_path))
    if not rows:
        raise ValueError(f"Empty predictions file: {pred_path}")

    # Group by task type
    task_data = {"ner": {"predictions": [], "references": []},
                 "action": {"predictions": [], "references": []}}
    for r in rows:
        tt = identify_task_type(r)
        task_data[tt]["predictions"].append((r.get("_prediction") or "").strip())
        task_data[tt]["references"].append((r.get("output") or "").strip())

    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(pred_path),
        "total_samples": len(rows),
        "counts": {k: len(v["predictions"]) for k, v in task_data.items()},
        "ner": {},
        "action": {},
    }

    if task_data["ner"]["predictions"]:
        ner_fm = eval_module.compute_field_level_ner_metrics(
            task_data["ner"]["predictions"],
            task_data["ner"]["references"],
            logger=None,
        )
        metrics["ner"]["field_metrics"] = ner_fm
        metrics["ner"]["avg"] = ner_fm["AVERAGE"]

    if task_data["action"]["predictions"]:
        action_m = eval_module.compute_action_metrics(
            task_data["action"]["predictions"],
            task_data["action"]["references"],
            logger=None,
        )
        metrics["action"] = action_m

    return metrics


def main():
    results_dir = Path(__file__).resolve().parents[2] / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    updated = []
    skipped = []

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        pred_path = subdir / "predictions.jsonl"
        if not pred_path.exists():
            skipped.append(str(subdir.name))
            continue

        metrics_path = subdir / "eval_metrics.json"
        old_metrics = None
        if metrics_path.exists():
            try:
                old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        print(f"\n[PROCESS] {subdir.name}")
        print(f"  predictions: {pred_path}")
        metrics = recompute_one(pred_path)

        # Write/overwrite eval_metrics.json
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"  saved: {metrics_path}")
        print(f"  counts: {metrics['counts']}")
        if metrics.get("ner") and metrics["ner"].get("avg"):
            print(f"  NER avg F1: {metrics['ner']['avg']['f1']:.4f}")
        if metrics.get("action") and metrics["action"].get("bleu_4") is not None:
            print(f"  Action BLEU-4: {metrics['action']['bleu_4']:.4f}  EM: {metrics['action']['em']:.4f}")

        updated.append({
            "name": subdir.name,
            "path": str(metrics_path),
            "old_ner_f1": old_metrics.get("ner", {}).get("avg", {}).get("f1") if old_metrics else None,
            "new_ner_f1": metrics.get("ner", {}).get("avg", {}).get("f1"),
            "old_action_bleu": old_metrics.get("action", {}).get("bleu_4") if old_metrics else None,
            "new_action_bleu": metrics.get("action", {}).get("bleu_4"),
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Updated: {len(updated)} | Skipped (no predictions.jsonl): {len(skipped)}")
    if skipped:
        print(f"Skipped dirs: {', '.join(skipped)}")

    print(f"\n{'Run':<50} {'Old NER F1':>12} {'New NER F1':>12} {'Old BLEU-4':>12} {'New BLEU-4':>12}")
    print("-" * 100)
    for u in updated:
        old_nf = f"{u['old_ner_f1']:.4f}" if u["old_ner_f1"] is not None else "-"
        new_nf = f"{u['new_ner_f1']:.4f}" if u["new_ner_f1"] is not None else "-"
        old_ab = f"{u['old_action_bleu']:.4f}" if u["old_action_bleu"] is not None else "-"
        new_ab = f"{u['new_action_bleu']:.4f}" if u["new_action_bleu"] is not None else "-"
        print(f"{u['name']:<50} {old_nf:>12} {new_nf:>12} {old_ab:>12} {new_ab:>12}")


if __name__ == "__main__":
    main()
