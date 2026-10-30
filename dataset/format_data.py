"""Format raw data into NER and Action training samples (JSONL).

The generated NER/Action samples keep stable ``record_id`` and ``group_id``
fields.  Split code must use those fields so the two tasks for one procedure,
and ideally all procedures from one patent, stay in the same dataset split.
"""

import json
import argparse
import hashlib
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

FIELD_ALIASES = {
    "reaction_product": ("REACTION_PRODUCT", "reaction_product"),
    "starting_material": ("STARTING_MATERIAL", "starting_material"),
    "reagent_catalyst": ("REAGENT_CATALYST", "reagent_catalyst"),
    "solvent": ("SOLVENT", "solvent"),
    "temperature": ("TEMPERATURE", "temperature"),
    "time": ("TIME", "time"),
    "yield_percent": ("YIELD_PERCENT", "yield_percent"),
}


def stable_hash(*parts: Any) -> str:
    text = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def get_first(item: Dict, *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item.get(key)
    return None


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
        "reaction_product": safe_multivalue(get_first(item, *FIELD_ALIASES["reaction_product"])),
        "starting_material": safe_multivalue(get_first(item, *FIELD_ALIASES["starting_material"])),
        "reagent_catalyst": safe_multivalue(get_first(item, *FIELD_ALIASES["reagent_catalyst"])),
        "solvent": safe_multivalue(get_first(item, *FIELD_ALIASES["solvent"])),
        "temperature": safe_scalar(get_first(item, *FIELD_ALIASES["temperature"])),
        "time": safe_scalar(get_first(item, *FIELD_ALIASES["time"])),
        "yield_percent": safe_scalar(get_first(item, *FIELD_ALIASES["yield_percent"])),
    }
    return f"```json\n{json.dumps(output, ensure_ascii=False)}\n```"


def build_action_output(item: Dict) -> str:
    actions = safe_action_list(item.get("action_sequence") or item.get("ACTION_SEQUENCE"))
    output = {"action_sequence": actions}
    return f"```json\n{json.dumps(output, ensure_ascii=False)}\n```"


def build_identity(item: Dict, index: int) -> Dict[str, Any]:
    text = item.get("text", "")
    source = item.get("source")
    language = item.get("language")
    patent_id = item.get("patent_id") or item.get("patent_number")
    file_name = item.get("file_name")
    record_id = item.get("record_id") or stable_hash(source, index, text)

    if patent_id:
        group_id = f"{source or 'UNKNOWN'}:patent:{patent_id}"
    elif file_name:
        group_id = f"{source or 'UNKNOWN'}:file:{file_name}"
    else:
        group_id = f"{source or 'UNKNOWN'}:record:{stable_hash(text)}"

    identity = {
        "record_id": record_id,
        "patent_id": patent_id,
        "group_id": group_id,
    }
    if source:
        identity["source"] = source
    if language:
        identity["language"] = language
    return identity


def format_data(input_path: Path, output_dir: Path):
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON list")

    output_dir.mkdir(parents=True, exist_ok=True)

    ner_samples = []
    action_samples = []
    mixed_samples = []

    for index, item in enumerate(records):
        text = item.get("text", "")
        if not text or str(text).strip() == "":
            continue
        identity = build_identity(item, index)

        ner_sample = {
            **identity,
            "instruction": NER_INSTRUCTION,
            "input": text,
            "output": build_ner_output(item),
            "task_type": 0,
        }
        ner_samples.append(ner_sample)

        action_sample = {
            **identity,
            "instruction": ACTION_INSTRUCTION,
            "input": text,
            "output": build_action_output(item),
            "task_type": 1,
        }
        action_samples.append(action_sample)

        mixed_samples.append(ner_sample)
        mixed_samples.append(action_sample)

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
