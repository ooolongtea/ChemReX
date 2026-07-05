"""Format raw data into NER and Action training samples (JSONL).

Input: raw_data.json with fields: text, REACTION_PRODUCT, STARTING_MATERIAL,
       REAGENT_CATALYST, SOLVENT, TEMPERATURE, TIME, YIELD_PERCENT, action_sequence.

Output: data_ner.jsonl and data_action.jsonl for downstream training.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any


NER_INSTRUCTION = (
    "Analyze the text and extract key chemical reaction entities "
    "(reaction_product, starting_material, reagent_catalyst, solvent, "
    "temperature, time, yield_percent). Output as Markdown JSON code block."
)

ACTION_INSTRUCTION = (
    "Analyze the text and extract the action sequence describing chemical "
    "experiment operations. Output the ordered list as Markdown JSON code block."
)

FIELDS = [
    "REACTION_PRODUCT", "STARTING_MATERIAL", "REAGENT_CATALYST",
    "SOLVENT", "TEMPERATURE", "TIME", "YIELD_PERCENT",
]


def safe_multivalue(val) -> Optional[str]:
    """Convert semicolon-separated string to comma-separated string."""
    if not val or str(val).strip() in ("", "None"):
        return None
    if isinstance(val, list):
        return ", ".join(str(v).strip() for v in val)
    items = [v.strip() for v in str(val).split(";") if v.strip()]
    return ", ".join(items) if items else None


def safe_scalar(val) -> Optional[str]:
    if not val or str(val).strip() in ("", "None"):
        return None
    return str(val).strip()


def safe_action_list(val) -> List[str]:
    if not val or str(val).strip() in ("", "None"):
        return []
    if isinstance(val, list):
        return val
    return [v.strip() for v in str(val).split(";") if v.strip()]


def build_ner_output(item: Dict) -> str:
    output = {
        "reaction_product": safe_multivalue(item.get("REACTION_PRODUCT")),
        "starting_material": safe_multivalue(item.get("STARTING_MATERIAL")),
        "reagent_catalyst": safe_multivalue(item.get("REAGENT_CATALYST")),
        "solvent": safe_multivalue(item.get("SOLVENT")),
        "temperature": safe_scalar(item.get("TEMPERATURE")),
        "time": safe_scalar(item.get("TIME")),
        "yield_percent": safe_scalar(item.get("YIELD_PERCENT")),
    }
    return f"```json\n{json.dumps(output, ensure_ascii=False)}\n```"


def build_action_output(item: Dict) -> str:
    actions = safe_action_list(item.get("action_sequence") or item.get("ACTION_SEQUENCE"))
    output = {"action_sequence": actions}
    return f"```json\n{json.dumps(output, ensure_ascii=False)}\n```"


def format_data(input_path: Path, output_dir: Path):
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON list")

    output_dir.mkdir(parents=True, exist_ok=True)

    ner_samples = []
    action_samples = []
    mixed_samples = []

    for item in records:
        text = item.get("text", "")
        if not text or str(text).strip() == "":
            continue

        ner_sample = {
            "instruction": NER_INSTRUCTION,
            "input": text,
            "output": build_ner_output(item),
        }
        ner_samples.append(ner_sample)

        action_sample = {
            "instruction": ACTION_INSTRUCTION,
            "input": text,
            "output": build_action_output(item),
        }
        action_samples.append(action_sample)

        # Mixed: alternate task_type
        mixed_samples.append({
            "task_type": 0,  # NER
            "instruction": NER_INSTRUCTION,
            "input": text,
            "output": build_ner_output(item),
        })
        mixed_samples.append({
            "task_type": 1,  # Action
            "instruction": ACTION_INSTRUCTION,
            "input": text,
            "output": build_action_output(item),
        })

    def save_jsonl(samples: List[Dict], name: str):
        path = output_dir / name
        with path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} samples -> {path}")

    save_jsonl(ner_samples, "data_ner.jsonl")
    save_jsonl(action_samples, "data_action.jsonl")
    save_jsonl(mixed_samples, "data_mixed.jsonl")

    meta = {
        "total_records": len(records),
        "ner_samples": len(ner_samples),
        "action_samples": len(action_samples),
        "mixed_samples": len(mixed_samples),
    }
    meta_path = output_dir / "format_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Meta saved -> {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Format raw data into NER/Action JSONL")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw_data.json")
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"), help="Output directory")
    args = parser.parse_args()
    format_data(args.input, args.output_dir)


if __name__ == "__main__":
    main()
