"""Entry script: build gold, parse predictions, run evaluation, output results."""

import argparse
import copy
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataset.build_gold import build_gold
from eval.end2end.parse_predictions import parse_predictions
from eval.end2end.grounding_eval import evaluate_sample, aggregate_results
from eval.end2end.normalize import normalize_identifier

BASE = os.path.dirname(os.path.abspath(__file__))


def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _filter_to_text_ids(gold: dict, pred: dict) -> tuple:
    """Filter gold and pred to only include identifiers found in gold text_mentions.

    Removes gold figure_labels / molecule_boxes / links / grounding for identifiers
    not present in the text.  Also removes the corresponding pred detections so
    that they are not counted as false positives.
    """
    text_ids_norm = {
        normalize_identifier(m["identifier"]) for m in gold["text_mentions"]
    }

    # ── Filter gold ──
    # Keep figure_labels whose normalized label is in text_ids_norm
    kept_label_idx = []
    new_figure_labels = []
    for i, fl in enumerate(gold["figure_labels"]):
        if normalize_identifier(fl["label"]) in text_ids_norm:
            kept_label_idx.append(i)
            new_figure_labels.append(fl)

    # Find mol indices linked to kept labels
    kept_mol_idx = set()
    for lm in gold["label_mol_links"]:
        if lm["label_idx"] in kept_label_idx:
            kept_mol_idx.add(lm["mol_idx"])

    new_molecule_boxes = [
        gold["molecule_boxes"][i] for i in sorted(kept_mol_idx)
    ]

    # Build old→new index maps
    old2new_label = {old: new for new, old in enumerate(kept_label_idx)}
    old2new_mol = {old: new for new, old in enumerate(sorted(kept_mol_idx))}

    new_label_mol_links = []
    for lm in gold["label_mol_links"]:
        if lm["label_idx"] in old2new_label and lm["mol_idx"] in old2new_mol:
            new_label_mol_links.append(
                {
                    "label_idx": old2new_label[lm["label_idx"]],
                    "mol_idx": old2new_mol[lm["mol_idx"]],
                }
            )

    new_id_label_links = []
    for il in gold["id_label_links"]:
        if normalize_identifier(il["identifier"]) in text_ids_norm:
            new_idx = old2new_label.get(il["label_idx"])
            if new_idx is not None:
                new_id_label_links.append(
                    {"identifier": il["identifier"], "label_idx": new_idx}
                )

    new_grounding = [
        g
        for g in gold["grounding"]
        if normalize_identifier(g["identifier"]) in text_ids_norm
    ]

    filtered_gold = dict(gold)
    filtered_gold["figure_labels"] = new_figure_labels
    filtered_gold["molecule_boxes"] = new_molecule_boxes
    filtered_gold["label_mol_links"] = new_label_mol_links
    filtered_gold["id_label_links"] = new_id_label_links
    filtered_gold["grounding"] = new_grounding

    # ── Filter pred ──
    # Keep pred figure_labels whose normalized label matches a kept gold label
    kept_gold_labels_norm = {
        normalize_identifier(fl["label"]) for fl in new_figure_labels
    }

    pred_kept_label_idx = []
    new_pred_labels = []
    for i, fl in enumerate(pred["figure_labels"]):
        if normalize_identifier(fl["label"]) in kept_gold_labels_norm:
            pred_kept_label_idx.append(i)
            new_pred_labels.append(fl)

    # Find pred mol indices linked to kept pred labels
    pred_kept_mol_idx = set()
    for lm in pred["label_mol_links"]:
        if lm["label_idx"] in pred_kept_label_idx:
            pred_kept_mol_idx.add(lm["mol_idx"])

    new_pred_mols = [
        pred["molecule_boxes"][i] for i in sorted(pred_kept_mol_idx)
    ]

    pred_old2new_label = {old: new for new, old in enumerate(pred_kept_label_idx)}
    pred_old2new_mol = {old: new for new, old in enumerate(sorted(pred_kept_mol_idx))}

    new_pred_links = []
    for lm in pred["label_mol_links"]:
        if lm["label_idx"] in pred_old2new_label and lm["mol_idx"] in pred_old2new_mol:
            new_pred_links.append(
                {
                    "label_idx": pred_old2new_label[lm["label_idx"]],
                    "mol_idx": pred_old2new_mol[lm["mol_idx"]],
                }
            )

    new_pred_dict = {
        k: v
        for k, v in pred["dict_result"].items()
        if normalize_identifier(k) in text_ids_norm
    }

    new_pred_id_smi = {
        k: v
        for k, v in pred["id_smiles_mapping"].items()
        if normalize_identifier(k) in text_ids_norm
    }

    filtered_pred = dict(pred)
    filtered_pred["figure_labels"] = new_pred_labels
    filtered_pred["molecule_boxes"] = new_pred_mols
    filtered_pred["label_mol_links"] = new_pred_links
    filtered_pred["dict_result"] = new_pred_dict
    filtered_pred["id_smiles_mapping"] = new_pred_id_smi

    return filtered_gold, filtered_pred


def main():
    ap = argparse.ArgumentParser(description="Run grounding evaluation")
    ap.add_argument("--sidecar", default="data/raw/openexp_500_coref_sidecar.json")
    ap.add_argument("--eval", default="data/raw/openexp_500_eval.json")
    ap.add_argument("--image-dir", default="data/raw/reference_sheets")
    ap.add_argument("--m2", default="data/outputs/m2/per_image_detections.jsonl")
    ap.add_argument("--m1", default="data/outputs/m1/m1_predictions.jsonl")
    ap.add_argument("--m3", default="data/outputs/m3/m3_coref.jsonl")
    ap.add_argument("--gold-output", default="data/gold/gold_annotations.jsonl")
    ap.add_argument("--pred-output", default=os.path.join(BASE, "pred/predictions.jsonl"))
    ap.add_argument("--output-dir", default=os.path.join(BASE, "outputs"))
    ap.add_argument("--dataset", default="openexp_500")
    ap.add_argument("--text-only", action="store_true",
                    help="Only evaluate identifiers that appear in text")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(BASE))
    sidecar_path = os.path.join(repo_root, args.sidecar) if not os.path.isabs(args.sidecar) else args.sidecar
    eval_path = os.path.join(repo_root, args.eval) if not os.path.isabs(args.eval) else args.eval
    m2_path = os.path.join(repo_root, args.m2) if not os.path.isabs(args.m2) else args.m2
    m1_path = os.path.join(repo_root, args.m1) if not os.path.isabs(args.m1) else args.m1
    m3_path = os.path.join(repo_root, args.m3) if not os.path.isabs(args.m3) else args.m3

    image_dir_path = os.path.join(repo_root, args.image_dir) if not os.path.isabs(args.image_dir) else args.image_dir
    gold_output_path = os.path.join(repo_root, args.gold_output) if not os.path.isabs(args.gold_output) else args.gold_output
    pred_output_path = os.path.join(repo_root, args.pred_output) if not os.path.isabs(args.pred_output) else args.pred_output

    print("=== Step 1: Build gold annotations ===")
    build_gold(sidecar_path, eval_path, gold_output_path, image_dir_path)

    print("=== Step 2: Parse predictions ===")
    parse_predictions(m2_path, m1_path, m3_path, pred_output_path)

    print("=== Step 3: Evaluate ===")
    gold_records = _load_jsonl(gold_output_path)
    pred_records = _load_jsonl(pred_output_path)

    pred_by_index = {r["openexp_index"]: r for r in pred_records}

    per_sample = []
    all_errors = []
    for gold in gold_records:
        idx = gold["openexp_index"]
        pred = pred_by_index.get(idx)
        if pred is None:
            print(f"  WARNING: no prediction for openexp_index={idx}, skipping")
            continue
        if args.text_only:
            gold, pred = _filter_to_text_ids(gold, pred)
        result = evaluate_sample(gold, pred)
        per_sample.append(result)
        all_errors.extend(result["errors"])

    print(f"  Evaluated {len(per_sample)} samples")

    print("=== Step 4: Aggregate ===")
    summary = aggregate_results(per_sample, args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    # Summary CSV
    csv_path = os.path.join(args.output_dir, "grounding_eval_summary.csv")
    fieldnames = list(summary.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)
    print(f"  Summary -> {csv_path}")

    # Per-sample JSONL
    per_sample_path = os.path.join(args.output_dir, "grounding_eval_per_sample.jsonl")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        for s in per_sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Per-sample -> {per_sample_path}")

    # Errors JSONL
    errors_path = os.path.join(args.output_dir, "grounding_eval_errors.jsonl")
    with open(errors_path, "w", encoding="utf-8") as f:
        for e in all_errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Errors -> {errors_path} ({len(all_errors)} cases)")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
