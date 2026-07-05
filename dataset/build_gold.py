"""Build gold annotations from sidecar + eval.json for grounding evaluation."""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.end2end.normalize import normalize_identifier

_PLACEHOLDER_RE = re.compile(r"\b(\d+[a-z])\b", re.IGNORECASE)
_ANGLE_RE = re.compile(r"<(\d+[a-z])>", re.IGNORECASE)
_COMPOUND_RE = re.compile(
    r"(?:compound|cmpd|product|intermediate|reactant|example)\s+(\d+[a-z])",
    re.IGNORECASE,
)
_COMPOUND_FIELDS = (
    "REACTION_PRODUCT",
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
)


def _find_text_mentions(
    text: str, identifiers: List[str], reference_style: str
) -> List[dict]:
    mentions = []
    seen_spans = set()
    for ident in identifiers:
        norm = normalize_identifier(ident)
        if reference_style == "angle":
            pattern = re.compile(re.escape(ident), re.IGNORECASE)
        elif reference_style == "compound":
            pattern = re.compile(
                rf"(?:compound|cmpd|product|intermediate|reactant|example)\s+{re.escape(ident)}",
                re.IGNORECASE,
            )
            bare = re.compile(rf"\b{re.escape(ident)}\b", re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{re.escape(ident)}\b", re.IGNORECASE)
            bare = None

        for m in pattern.finditer(text):
            key = (m.start(), m.end())
            if key not in seen_spans:
                seen_spans.add(key)
                mentions.append(
                    {"identifier": ident, "span": [m.start(), m.end()]}
                )

        if reference_style == "compound":
            for m in bare.finditer(text):
                prefix_start = max(0, m.start() - 15)
                prefix_text = text[prefix_start : m.start()].lower()
                if not any(
                    p in prefix_text
                    for p in ("compound", "cmpd", "product", "intermediate", "reactant", "example")
                ):
                    key = (m.start(), m.end())
                    if key not in seen_spans:
                        seen_spans.add(key)
                        mentions.append(
                            {"identifier": ident, "span": [m.start(), m.end()]}
                        )

    mentions.sort(key=lambda x: x["span"][0])
    return mentions


_IMAGE_CACHE: Dict[str, Image.Image] = {}


def _load_image(image_dir: str, image_file: str) -> Optional[Image.Image]:
    if image_file not in _IMAGE_CACHE:
        path = os.path.join(image_dir, image_file)
        if not os.path.exists(path):
            return None
        _IMAGE_CACHE[image_file] = Image.open(path).convert("RGB")
    return _IMAGE_CACHE[image_file]


def _trim_mol_bbox(
    image: Image.Image,
    mol_bbox: List[int],
    label_bbox: Optional[List[int]] = None,
) -> List[int]:
    """Trim gold molecule bbox to exclude label region and white padding.

    The raw mol_bbox from build_openexp_eval.py covers the full tile
    (molecule + RDKit padding + label text).  This function:
      1. Cuts off the label region (everything below label_bbox top)
      2. Trims white borders to get the tight molecule structure extent
    """
    x1, y1, x2, y2 = mol_bbox

    # Step 1: exclude label region
    if label_bbox and len(label_bbox) >= 4:
        cut_y = label_bbox[1] - 2  # 2px gap above label
        if cut_y > y1 + 10:
            y2 = cut_y

    # Step 2: trim white borders
    crop = image.crop((x1, y1, x2, y2))
    arr = np.array(crop)
    # Non-white mask: any channel < 250
    mask = np.any(arr < 250, axis=2)
    if not mask.any():
        return [x1, y1, x2, y2]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [
        int(x1 + cmin),
        int(y1 + rmin),
        int(x1 + cmax + 1),
        int(y1 + rmax + 1),
    ]


def build_gold(sidecar_path: str, eval_path: str, output_path: str, image_dir: str = ""):
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    ner_by_index = {}
    for row in eval_data:
        if row.get("task_type", "").upper() == "NER":
            ner_by_index[row["openexp_index"]] = row

    gold_records = []
    for rec in sidecar:
        idx = rec["openexp_index"]
        ia = rec.get("image_annotation", {})
        participants = ia.get("participants", [])
        dict_result = rec.get("dict_result", {})
        image_file = ia.get("image_file", f"openexp_{idx}.png")

        eval_row = ner_by_index.get(idx, {})
        input_text = eval_row.get("input", "")
        reference_style = rec.get("reference_style", "dollar")

        identifiers = list(dict_result.keys())
        text_mentions = _find_text_mentions(input_text, identifiers, reference_style)

        # Load image for bbox trimming
        ref_image = None
        if image_dir:
            ref_image = _load_image(image_dir, image_file)

        figure_labels = []
        molecule_boxes = []
        label_mol_links = []
        id_label_links = []
        grounding = []

        for i, p in enumerate(participants):
            label = p.get("label", "")
            smiles = p.get("smiles", "")
            mol_bbox = p.get("bbox_xyxy", [])
            label_bbox = p.get("label_bbox_xyxy", [])

            # Trim mol bbox to molecule structure only
            if mol_bbox and ref_image is not None:
                mol_bbox = _trim_mol_bbox(ref_image, mol_bbox, label_bbox)

            mol_idx = len(molecule_boxes)
            label_idx = len(figure_labels)

            if mol_bbox:
                molecule_boxes.append({"bbox": mol_bbox, "smiles": smiles})
            if label_bbox:
                figure_labels.append({"label": label, "bbox": label_bbox})
            if mol_bbox and label_bbox:
                label_mol_links.append({"label_idx": label_idx, "mol_idx": mol_idx})
            if label_bbox:
                id_label_links.append(
                    {"identifier": p.get("ref", label), "label_idx": label_idx}
                )

        for ident, smiles in dict_result.items():
            grounding.append({"identifier": ident, "smiles": smiles})

        gold_records.append(
            {
                "openexp_index": idx,
                "image_file": ia.get("image_file", f"openexp_{idx}.png"),
                "image_size": ia.get("image_size", {}),
                "reference_style": reference_style,
                "text_mentions": text_mentions,
                "figure_labels": figure_labels,
                "molecule_boxes": molecule_boxes,
                "label_mol_links": label_mol_links,
                "id_label_links": id_label_links,
                "grounding": grounding,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in gold_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Gold annotations: {len(gold_records)} -> {output_path}")
    return gold_records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--image-dir", default="")
    args = ap.parse_args()
    build_gold(args.sidecar, args.eval, args.output, args.image_dir)
