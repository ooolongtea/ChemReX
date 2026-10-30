"""Build M1 extraction input JSONL from openexp_500_eval.json.

Takes the 500 unique NER rows (one per openexp_index) and writes a JSONL
file consumable by `pipeline.extraction.run_extraction` batch mode.

Each output line:
    {"text": "...", "language": "en", "task_type": "ner",
     "openexp_index": 58369, "reference_style": "dollar"}
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/openexp_500_eval.json"),
        help="Path to openexp_500_eval.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/m1_input.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            if row.get("task_type") != "NER":
                continue
            idx = row.get("openexp_index")
            if idx in seen:
                continue
            seen.add(idx)
            out = {
                "text": row["input"],
                "language": row.get("language", "en"),
                "task_type": "both",
                "openexp_index": idx,
                "reference_style": row.get("reference_style"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"saved {written} M1 input rows -> {args.output}")


if __name__ == "__main__":
    main()
