"""Build ChemReX-style eval data from OpenExp_500.json.

The output JSON follows data/seed-67/test.json:
instruction, input, output, task_type, language, source.

OpenExp already contains molecule/action placeholders. This script keeps that
signal by rewriting procedure text with deterministic identifier styles and by
emitting sidecar metadata that maps identifiers to SMILES.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


NER_PREFIX_EN = (
    "Below is a description of an organic reaction. Analyze text and extract "
    "the following fields into a Markdown JSON code block:reaction_product, "
    "starting_material, reagent_catalyst, solvent, temperature, time, "
    "yield_percent.**Output ONLY the JSON**\n"
)

ACTION_PREFIX_EN = (
    "Below is a description of an organic reaction. Analyze text and extract "
    "the **atomic action verbs**(e.g., add, purify) in chronological order, "
    "separated by **English commas**.**Output ONLY a single line of verbs**.\n"
)


ACTION_MAP = {
    "ADD": "add",
    "COLLECTLAYER": "collect layer",
    "CONCENTRATE": "concentrate",
    "DEGAS": "degas",
    "DRYSOLID": "dry",
    "DRYSOLUTION": "dry",
    "EXTRACT": "extract",
    "FILTER": "filter",
    "MAKESOLUTION": "make solution",
    "PARTITION": "partition",
    "PH": "adjust pH",
    "PHASESEPARATION": "separate phase",
    "QUENCH": "quench",
    "RECRYSTALLIZE": "recrystallize",
    "REFLUX": "reflux",
    "SETTEMPERATURE": "set temperature",
    "SONICATE": "sonicate",
    "STIR": "stir",
    "TRITURATE": "triturate",
    "WAIT": "wait",
    "WASH": "wash",
    "YIELD": "yield",
}

STYLE_CHOICES = ("dollar", "compound", "angle")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def placeholder_number(token: str) -> str:
    match = re.search(r"\d+", token)
    return match.group(0) if match else token.strip("$#@-")


def placeholder_code(token: str, kind: str = "molecule") -> str:
    number = placeholder_number(token)
    if kind == "product":
        suffix = "b"
    elif kind == "temperature":
        suffix = "t"
    elif kind == "duration":
        suffix = "d"
    else:
        suffix = "a"
    return f"{number}{suffix}"


def styled_placeholder(token: str, style: str, kind: str = "molecule") -> str:
    code = placeholder_code(token, kind)

    if style == "dollar":
        return code
    if style == "compound":
        if kind == "product":
            return f"product {code}"
        if kind == "temperature":
            return f"temperature {code}"
        if kind == "duration":
            return f"duration {code}"
        return f"compound {code}"
    if style == "angle":
        return f"<{code}>"
    raise ValueError(f"unknown reference style: {style}")


def display_ref(ref: str, kind: str = "molecule") -> str:
    return placeholder_code(ref, kind)


def style_for_index(index: int, styles: Sequence[str]) -> str:
    if not styles:
        raise ValueError("at least one style is required")
    return styles[index % len(styles)]


def invert_extracted_molecules(record: Dict[str, Any]) -> Dict[str, str]:
    extracted = record.get("extracted_molecules") or {}
    return {str(smiles): str(ref) for smiles, ref in extracted.items()}


def _is_organic_fragment(mol: Any) -> bool:
    if mol.GetNumBonds() == 0:
        return False

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        if atom.GetTotalNumHs() > 0:
            return True
        if any(neighbor.GetAtomicNum() not in (8,) for neighbor in atom.GetNeighbors()):
            return True
    return False


def organic_fragment_smiles(smiles: str) -> Optional[str]:
    """Keep only drawable organic fragments from a possibly disconnected SMILES."""
    try:
        from rdkit import Chem
    except ImportError:
        return str(smiles)

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    organic = [
        Chem.MolToSmiles(fragment, canonical=True)
        for fragment in fragments
        if _is_organic_fragment(fragment)
    ]
    if not organic:
        return None
    if len(fragments) == 1:
        return str(smiles)
    return ".".join(organic)


def role_refs(record: Dict[str, Any], style: str) -> Dict[str, Dict[str, str]]:
    smiles_to_ref = invert_extracted_molecules(record)
    out: Dict[str, Dict[str, str]] = {}
    for role, kind in [
        ("REACTANT", "molecule"),
        ("CATALYST", "molecule"),
        ("SOLVENT", "molecule"),
        ("PRODUCT", "product"),
    ]:
        out[role] = {}
        for smiles in record.get(role) or []:
            display_smiles = str(smiles)
            if role in ("REACTANT", "PRODUCT"):
                fragment_smiles = organic_fragment_smiles(str(smiles))
                if fragment_smiles is None:
                    continue
                display_smiles = fragment_smiles
            token = smiles_to_ref.get(str(smiles))
            if token:
                out[role][display_smiles] = styled_placeholder(token, style, kind)
            else:
                out[role][display_smiles] = display_smiles
    return out


def name_replacements(record: Dict[str, Any], style: str) -> List[Tuple[str, str]]:
    smiles_to_ref = invert_extracted_molecules(record)
    referenced_smiles = {
        str(x)
        for x in (record.get("REACTANT") or [])
        if organic_fragment_smiles(str(x)) is not None
    }
    referenced_smiles.update(
        str(x)
        for x in (record.get("PRODUCT") or [])
        if organic_fragment_smiles(str(x)) is not None
    )
    replacements: List[Tuple[str, str]] = []
    for name, smiles in (record.get("molecules") or {}).items():
        token = smiles_to_ref.get(str(smiles))
        if token and str(smiles) in referenced_smiles:
            kind = "product" if str(smiles) in set(str(x) for x in (record.get("PRODUCT") or [])) else "molecule"
            replacements.append((str(name), styled_placeholder(token, style, kind)))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    return replacements


def replace_case_insensitive(text: str, old: str, new: str) -> str:
    if not old.strip():
        return text
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(new, text)


def build_referenced_source(record: Dict[str, Any], style: str) -> str:
    text = str(record.get("source") or "")
    for old, new in name_replacements(record, style):
        text = replace_case_insensitive(text, old, new)
    return text


def extract_action_verbs(actions: str) -> List[str]:
    verbs: List[str] = []
    for part in str(actions or "").split(";"):
        part = part.strip()
        if not part:
            continue
        token = part.split(None, 1)[0].strip().upper()
        verbs.append(ACTION_MAP.get(token, token.lower()))
    return verbs


def join_values(values: Iterable[Optional[str]]) -> Optional[str]:
    cleaned = [str(v).strip() for v in values if v is not None and str(v).strip()]
    return ";".join(cleaned) if cleaned else None


def first_or_join(mapping: Dict[str, str]) -> Optional[str]:
    return join_values(mapping.values())


def join_smiles(values: Iterable[Any]) -> Optional[str]:
    return join_values(str(v) for v in values)


def extract_yield(actions: str) -> Optional[str]:
    match = re.search(r"\bYIELD\b[^;]*\(([^)]*(?:%|yield)[^)]*)\)", str(actions or ""), flags=re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"\bYIELD\b[^;]*?(\d+(?:\.\d+)?\s*%)", str(actions or ""), flags=re.I)
    return match.group(1).strip() if match else None


def markdown_json(obj: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


def build_ner_output(record: Dict[str, Any], refs: Dict[str, Dict[str, str]]) -> str:
    temps = list((record.get("extracted_temperature") or {}).keys())
    durations = list((record.get("extracted_duration") or {}).keys())
    output = {
        "reaction_product": first_or_join(refs["PRODUCT"]),
        "starting_material": first_or_join(refs["REACTANT"]),
        "reagent_catalyst": join_smiles(record.get("CATALYST") or []),
        "solvent": join_smiles(record.get("SOLVENT") or []),
        "temperature": join_values(temps),
        "time": join_values(durations),
        "yield_percent": extract_yield(str(record.get("actions") or "")),
    }
    return markdown_json(output)


def reaction_conditions(record: Dict[str, Any]) -> List[str]:
    conditions: List[str] = []
    catalysts = join_smiles(record.get("CATALYST") or [])
    solvents = join_smiles(record.get("SOLVENT") or [])
    temps = join_values((record.get("extracted_temperature") or {}).keys())
    durations = join_values((record.get("extracted_duration") or {}).keys())
    if catalysts:
        conditions.append(f"cat./reagent: {catalysts}")
    if solvents:
        conditions.append(f"solvent: {solvents}")
    if temps:
        conditions.append(f"temp: {temps}")
    if durations:
        conditions.append(f"time: {durations}")
    return conditions


def instruction(prefix: str, source: str) -> str:
    return f"{prefix}###Procedure:{source}"


def build_eval_rows(
    records: Sequence[Dict[str, Any]],
    tasks: str,
    styles: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    sidecar: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        style = style_for_index(i, styles)
        source = build_referenced_source(record, style)
        refs = role_refs(record, style)
        dict_result = {}
        for role_map in (refs["REACTANT"], refs["PRODUCT"]):
            for smiles, ref in role_map.items():
                if ref != smiles:
                    dict_result[ref] = smiles

        base_meta = {
            "openexp_index": record.get("index"),
            "reference_style": style,
            "dict_result": dict_result,
            "reaction_refs": {
                "reactants": refs["REACTANT"],
                "products": refs["PRODUCT"],
                "conditions": reaction_conditions(record),
            },
            "original_source": record.get("source"),
            "original_actions": record.get("actions"),
            "score": record.get("score"),
        }
        sidecar.append(base_meta)

        if tasks in ("action", "both"):
            rows.append(
                {
                    "instruction": instruction(ACTION_PREFIX_EN, source),
                    "input": source,
                    "output": ", ".join(extract_action_verbs(str(record.get("actions") or ""))),
                    "task_type": "Action",
                    "language": "EN",
                    "source": "OpenExp",
                    "openexp_index": record.get("index"),
                    "reference_style": style,
                }
            )
        if tasks in ("ner", "both"):
            rows.append(
                {
                    "instruction": instruction(NER_PREFIX_EN, source),
                    "input": source,
                    "output": build_ner_output(record, refs),
                    "task_type": "NER",
                    "language": "EN",
                    "source": "OpenExp",
                    "openexp_index": record.get("index"),
                    "reference_style": style,
                }
            )
    return rows, sidecar


def _wrap_text(text: str, max_chars: int = 82, max_lines: int = 4) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        nxt = word if not cur else f"{cur} {word}"
        if len(nxt) <= max_chars:
            cur = nxt
            continue
        if cur:
            lines.append(cur)
        cur = word
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and words:
        all_text = " ".join(words)
        shown = " ".join(lines)
        if len(shown) < len(all_text):
            lines[-1] = lines[-1].rstrip(".") + "..."
    return lines or [""]


def _font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []
    ) + [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: Any, text: str, font: Any) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_centered_text(
    draw: Any,
    center: Tuple[int, int],
    text: str,
    font: Any,
    fill: str = "black",
) -> List[int]:
    w, h = _text_size(draw, text, font)
    x = int(center[0] - w / 2)
    y = int(center[1] - h / 2)
    draw.text((x, y), text, font=font, fill=fill)
    return [x, y, x + w, y + h]


def _draw_arrow(draw: Any, start: Tuple[int, int], end: Tuple[int, int]) -> None:
    x1, y1 = start
    x2, y2 = end
    head_len = 18
    head_h = 8
    draw.line((x1, y1, x2, y2), fill="black", width=3)
    draw.polygon(
        [(x2, y2), (x2 - head_len, y2 - head_h), (x2 - head_len, y2 + head_h)],
        fill="black",
    )


def _molecule_tile(
    smiles: str,
    legend: str,
    kind: str,
    size: Tuple[int, int],
) -> Optional[Tuple[Any, List[int]]]:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from PIL import Image, ImageDraw

    fragment_smiles = organic_fragment_smiles(smiles)
    if fragment_smiles is None:
        return None
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return None
    w, h = size
    label_font = _font(20, bold=True)
    label_gap = 6
    label_h = 24
    mol_h = h - label_h - label_gap
    img = Draw.MolToImage(mol, size=(w, mol_h), legend="")
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    tile = Image.new("RGB", (w, h), "white")
    tile.paste(img, (0, 0))
    draw = ImageDraw.Draw(tile)
    label_bbox = _draw_centered_text(
        draw,
        (w // 2, mol_h + label_gap + label_h // 2),
        display_ref(legend, kind),
        label_font,
    )
    return tile, label_bbox


def draw_reference_sheets(sidecar: Sequence[Dict[str, Any]], image_dir: Path) -> int:
    from PIL import Image, ImageDraw

    image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    tile_size = (210, 172)
    gap = 34
    margin_x = 34
    arrow_w = 280
    arrow_gap = 34
    header_h = 34
    canvas_h = 292
    arrow_y = 156
    mol_y = arrow_y - tile_size[1] // 2
    index_font = _font(13)
    plus_font = _font(30, bold=True)
    condition_font = _font(15)
    for item in sidecar:
        reaction_refs = item.get("reaction_refs") or {}
        reactants = [
            (ref, smiles)
            for smiles, ref in (reaction_refs.get("reactants") or {}).items()
        ]
        products = [
            (ref, smiles)
            for smiles, ref in (reaction_refs.get("products") or {}).items()
        ]
        reactants = [
            (ref, smiles)
            for ref, smiles in reactants
            if organic_fragment_smiles(smiles) is not None
        ]
        products = [
            (ref, smiles)
            for ref, smiles in products
            if organic_fragment_smiles(smiles) is not None
        ]
        if not reactants or not products:
            continue

        left_w = len(reactants) * tile_size[0] + max(0, len(reactants) - 1) * gap
        right_w = len(products) * tile_size[0] + max(0, len(products) - 1) * gap
        width = max(960, margin_x * 2 + left_w + right_w + arrow_w + arrow_gap * 2)
        height = max(canvas_h, header_h + tile_size[1] + 44)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (margin_x, 10),
            f"OpenExp index: {item.get('openexp_index')}",
            font=index_font,
            fill="black",
        )
        image_file = f"openexp_{item.get('openexp_index')}.png"
        bboxes: Dict[str, Any] = {
            "image_file": image_file,
            "image_size": {"width": width, "height": height},
            "participants": [],
        }

        left_x = margin_x
        for idx, (ref, smiles) in enumerate(reactants):
            tile_result = _molecule_tile(smiles, ref, "molecule", tile_size)
            if tile_result is None:
                continue
            tile, label_bbox = tile_result
            x = left_x + idx * (tile_size[0] + gap)
            canvas.paste(tile, (x, mol_y))
            bboxes["participants"].append(
                {
                    "role": "reactant",
                    "ref": ref,
                    "label": display_ref(ref, "molecule"),
                    "smiles": smiles,
                    "bbox_xyxy": [x, mol_y, x + tile_size[0], mol_y + tile_size[1]],
                    "label_bbox_xyxy": [
                        x + label_bbox[0],
                        mol_y + label_bbox[1],
                        x + label_bbox[2],
                        mol_y + label_bbox[3],
                    ],
                }
            )
            if idx < len(reactants) - 1:
                plus_x = x + tile_size[0] + gap // 2
                _draw_centered_text(draw, (plus_x, arrow_y), "+", plus_font)

        arrow_start_x = left_x + left_w + arrow_gap
        arrow_end_x = arrow_start_x + arrow_w
        _draw_arrow(draw, (arrow_start_x, arrow_y), (arrow_end_x, arrow_y))
        bboxes["arrow"] = {
            "bbox_xyxy": [arrow_start_x, arrow_y - 10, arrow_end_x, arrow_y + 10],
        }

        condition_text = "; ".join(reaction_refs.get("conditions") or ["conditions"])
        condition_lines = _wrap_text(condition_text, max_chars=48, max_lines=3)
        line_h = 18
        cond_h = len(condition_lines) * line_h
        cond_y = arrow_y - cond_h - 12
        cond_boxes: List[List[int]] = []
        for line_idx, line in enumerate(condition_lines):
            cond_boxes.append(
                _draw_centered_text(
                    draw,
                    (
                        (arrow_start_x + arrow_end_x) // 2,
                        cond_y + line_idx * line_h + line_h // 2,
                    ),
                    line,
                    condition_font,
                )
            )
        if cond_boxes:
            cond_bbox = [
                min(box[0] for box in cond_boxes),
                min(box[1] for box in cond_boxes),
                max(box[2] for box in cond_boxes),
                max(box[3] for box in cond_boxes),
            ]
        else:
            cond_bbox = [arrow_start_x, cond_y, arrow_end_x, cond_y]
        bboxes["conditions"] = {
            "text": condition_text,
            "bbox_xyxy": cond_bbox,
        }

        right_x = arrow_end_x + arrow_gap
        for idx, (ref, smiles) in enumerate(products):
            tile_result = _molecule_tile(smiles, ref, "product", tile_size)
            if tile_result is None:
                continue
            tile, label_bbox = tile_result
            x = right_x + idx * (tile_size[0] + gap)
            canvas.paste(tile, (x, mol_y))
            bboxes["participants"].append(
                {
                    "role": "product",
                    "ref": ref,
                    "label": display_ref(ref, "product"),
                    "smiles": smiles,
                    "bbox_xyxy": [x, mol_y, x + tile_size[0], mol_y + tile_size[1]],
                    "label_bbox_xyxy": [
                        x + label_bbox[0],
                        mol_y + label_bbox[1],
                        x + label_bbox[2],
                        mol_y + label_bbox[3],
                    ],
                }
            )
            if idx < len(products) - 1:
                plus_x = x + tile_size[0] + gap // 2
                _draw_centered_text(draw, (plus_x, arrow_y), "+", plus_font)

        item["image_annotation"] = bboxes
        canvas.save(image_dir / image_file)
        written += 1
    return written


def parse_styles(raw: str) -> List[str]:
    styles = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in styles if x not in STYLE_CHOICES]
    if unknown:
        raise ValueError(f"unknown styles {unknown}; choices: {', '.join(STYLE_CHOICES)}")
    return styles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/OpenExp_500.json"),
        help="Path to OpenExp_500.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/openexp_500_eval.json"),
        help="ChemReX-style eval JSON output path",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path("data/raw/openexp_500_coref_sidecar.json"),
        help="Identifier-to-SMILES sidecar output path",
    )
    parser.add_argument("--tasks", choices=("action", "ner", "both"), default="both")
    parser.add_argument(
        "--styles",
        default="dollar,compound,angle",
        help=f"Comma-separated reference styles. Choices: {', '.join(STYLE_CHOICES)}",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional directory for RDKit reference-sheet PNGs.",
    )
    args = parser.parse_args()

    records = load_json(args.input)
    if not isinstance(records, list):
        raise ValueError("OpenExp input must be a JSON list")
    styles = parse_styles(args.styles)
    rows, sidecar = build_eval_rows(records, args.tasks, styles)
    if args.image_dir:
        written = draw_reference_sheets(sidecar, args.image_dir)
        print(f"saved {written} reference-sheet images -> {args.image_dir}")
    dump_json(args.output, rows)
    dump_json(args.sidecar, sidecar)

    print(f"saved {len(rows)} eval rows -> {args.output}")
    print(f"saved {len(sidecar)} sidecar rows -> {args.sidecar}")


if __name__ == "__main__":
    main()
