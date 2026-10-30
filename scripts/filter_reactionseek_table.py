"""Filter ReactionSeek CSV rows using current ChemReX chunk diagnostics.

Usage: python -m scripts.filter_reactionseek_table --batch-root DIR
       --input all_table.csv --output all_table.filtered.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

from pipeline.document.results import latest_by_id
from pipeline.document.segmenter import classify_process

# Products like "VII" / "IV" / "取决于溶剂）" are crystal-form screening
# fragments or truncated table cells, not molecules (v7 review P-12).
_PLACEHOLDER_PRODUCT = re.compile(
    r"^(?:[IVXLCDM]+\s*|取决于[^,;；]*|depends\s+on\s+solv\w*|N/?A)[\s）)\]]*$", re.I)


def _chunks(folder: Path) -> dict:
    path = folder / "chunks.jsonl"
    if not path.is_file():
        return {}
    chunks = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict) and chunk.get("id"):
                chunks[chunk["id"]] = chunk
    return chunks


def filter_table(batch_root: Path, input_csv: Path, output_csv: Path) -> tuple:
    if input_csv.resolve() == output_csv.resolve():
        raise ValueError("output must differ from input")
    cache = {}
    kept = removed = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open(newline="", encoding="utf-8-sig") as src, \
            output_csv.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        fields = reader.fieldnames or []
        writer = csv.DictWriter(dst, fieldnames=fields +
                                [name for name in ("rxn_status", "non_reaction")
                                 if name not in fields])
        writer.writeheader()
        for row in reader:
            sample = row.get("Sample") or input_csv.stem.removesuffix("_table")
            if sample not in cache:
                folder = batch_root / sample
                chunks = _chunks(folder)
                latest = latest_by_id(
                    folder / "results.jsonl",
                    ({cid: c.get("chunk_fingerprint") for cid, c in chunks.items()}
                     if all(c.get("chunk_fingerprint") for c in chunks.values()) else None)) if chunks else {}
                cache[sample] = (chunks, latest)
            chunks, latest = cache[sample]
            raw_id = row.get("Index", "")
            chunk_id = raw_id if raw_id in chunks else raw_id.rsplit("_", 1)[0]
            chunk = chunks.get(chunk_id)
            record = latest.get(chunk_id, {})
            reactions = record.get("reactions") or []
            reaction = None
            if raw_id.startswith(chunk_id + "_") and raw_id[len(chunk_id) + 1:].isdigit():
                reaction_index = int(raw_id[len(chunk_id) + 1:])
                reaction = next((item for item in reactions
                                 if item.get("index") == reaction_index), None)
            status = (reaction or record).get("status", "unknown")
            kind = (chunk.get("process_kind") or classify_process(
                chunk.get("title", ""), chunk.get("text", ""))) if chunk else "unknown"
            validation = ((reaction or record).get("validation") or {}).get("rxn_validation") or {}
            diagnosis = validation.get("diagnosis", "")
            expression = validation.get("base_expression", "")
            identity = ">>" in expression and expression.split(">>", 1)[0] == expression.split(">>", 1)[1]
            rxn_status = validation.get("status") or status
            # A skipped row with an empty side failed extraction (or names a
            # formulation), not a usable reaction (v7 review P-14).
            skipped_without_sides = (rxn_status == "skipped" and
                                     (row.get("Reactants") == "N/A" or
                                      row.get("Products") == "N/A"))
            empty_sides = row.get("Reactants") == "N/A" and row.get("Products") == "N/A"
            non_reaction = (empty_sides or status == "non_reaction" or
                            kind in ("assay", "analysis", "purification", "formulation") or
                            diagnosis in ("identity_reaction", "non_reaction_context") or
                            identity or skipped_without_sides or
                            bool(_PLACEHOLDER_PRODUCT.match(row.get("Products") or "")))
            if non_reaction:
                removed += 1
                continue
            row["rxn_status"] = rxn_status
            row["non_reaction"] = "false"
            writer.writerow(row)
            kept += 1
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kept, removed = filter_table(args.batch_root, args.input, args.output)
    print(f"kept={kept} removed={removed} output={args.output}")


if __name__ == "__main__":
    main()
