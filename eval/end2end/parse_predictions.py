"""Parse M1/M2/M3 outputs into prediction format aligned with gold annotations."""

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.end2end.normalize import is_placeholder_identifier, normalize_identifier

_COMPOUND_FIELDS = (
    "REACTION_PRODUCT",
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
)


def _load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_text_identifiers(ner: dict) -> List[str]:
    ids = []
    seen = set()
    for field in _COMPOUND_FIELDS:
        vals = ner.get(field, [])
        if isinstance(vals, str):
            vals = [vals] if vals.strip() else []
        for v in vals if isinstance(vals, list) else [vals]:
            name = str(v).strip()
            if not name:
                continue
            norm = normalize_identifier(name)
            if norm and is_placeholder_identifier(norm) and norm not in seen:
                seen.add(norm)
                ids.append(norm)
    return ids


def parse_predictions(
    m2_jsonl_path: str,
    m1_jsonl_path: str,
    m3_jsonl_path: str,
    output_path: str,
):
    m2_records = _load_jsonl(m2_jsonl_path)
    m1_records = _load_jsonl(m1_jsonl_path)
    m3_records = _load_jsonl(m3_jsonl_path)

    m1_by_index = {r["openexp_index"]: r for r in m1_records}
    m3_by_index = {r["openexp_index"]: r for r in m3_records}

    pred_records = []
    for m2 in m2_records:
        fname = m2.get("image_file_name", "")
        idx = int(fname.replace("openexp_", "").replace(".png", ""))

        figure_labels = []
        for fl in m2.get("figure_labels", []):
            figure_labels.append(
                {
                    "label": fl.get("text", ""),
                    "bbox": fl.get("bbox_xyxy", []),
                    "confidence": fl.get("confidence", 1.0),
                }
            )

        molecule_boxes = []
        for mb in m2.get("molecule_boxes", []):
            molecule_boxes.append(
                {
                    "bbox": mb.get("bbox_xyxy", []),
                    "smiles": mb.get("smiles", ""),
                    "confidence": mb.get("confidence", 1.0),
                }
            )

        label_mol_links = m2.get("label_mol_links", [])
        dict_result = m2.get("dict_result", {})

        m1 = m1_by_index.get(idx, {})
        m1_ner = m1.get("_prediction", {}).get("ner", {})
        text_mentions = [
            {"identifier": ident} for ident in _extract_text_identifiers(m1_ner)
        ]

        m3 = m3_by_index.get(idx, {})
        id_smiles_mapping = m3.get("id_smiles_mapping", {})

        pred_records.append(
            {
                "openexp_index": idx,
                "image_file": fname,
                "figure_labels": figure_labels,
                "molecule_boxes": molecule_boxes,
                "label_mol_links": label_mol_links,
                "dict_result": dict_result,
                "text_mentions": text_mentions,
                "id_smiles_mapping": id_smiles_mapping,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in pred_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Predictions: {len(pred_records)} -> {output_path}")
    return pred_records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2", required=True, help="M2 per_image_detections JSONL")
    ap.add_argument("--m1", required=True, help="M1 predictions JSONL")
    ap.add_argument("--m3", required=True, help="M3 coref JSONL")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    parse_predictions(args.m2, args.m1, args.m3, args.output)
