"""Rebuild the ReactionSeek-style CSV table from a ChemReX run.

Usage: python -m scripts.export_reactionseek_table --run-root DIR --output all_table.csv

One row per extracted reaction item from current chunk fingerprints, including
review-only placeholders with unavailable compounds.
The general filter removes non-reaction/identity/formulation rows; trusted-only
requires the current evidence contract and normal step/chunk/mapping states.
V9 appends reagent, participant, effective-expression and provenance fields.
"""

import argparse
import csv
import json
from pathlib import Path

from pipeline.document.results import latest_by_id
from pipeline.validation.quality import QUALITY_VERSION

COLUMNS = ["Sample", "Index", "Reactants", "Reactant amounts", "Products",
           "Product amounts", "Solvents", "Reaction temperature", "Reaction time",
           "Yield", "rxn_status", "record_status", "process_kind", "diagnosis"]
COLUMNS += ["step_id", "step_status", "Reagents", "Participants", "effective_expression",
            "step_source", "procedure_references", "field_evidence", "yield_info",
            "stereochemistry", "structure_status", "record_completeness",
            "blocking_issues", "admission_eligible", "quality_version"]


def _names(entries) -> str:
    parts = []
    for entry in entries or []:
        if isinstance(entry, dict):
            value = entry.get("name") or entry.get("smiles")
        else:
            value = entry
        if value:
            parts.append(str(value))
    return "; ".join(parts) if parts else "N/A"


def _scalar(value) -> str:
    if value in (None, "", []):
        return "N/A"
    if isinstance(value, list):
        joined = "; ".join(str(part) for part in value)
        return joined or "N/A"
    return str(value)


def _rows(sample: str, folder: Path) -> list:
    chunks_path = folder / "chunks.jsonl"
    if not chunks_path.is_file():
        return []
    chunks = [json.loads(line) for line in chunks_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    current = {c["id"]: c.get("chunk_fingerprint") for c in chunks}
    latest = latest_by_id(folder / "results.jsonl", current)
    rows = []
    for chunk in chunks:
        record = latest.get(chunk["id"])
        if not record:
            continue
        reactions = record.get("reactions") or []
        for item in reactions:
            ner = ((item.get("coref") or {}).get("ner") or
                   (item.get("extraction") or {}).get("ner") or {})
            if not isinstance(ner, dict):
                continue
            reactants = _names(ner.get("STARTING_MATERIAL"))
            products = _names(ner.get("REACTION_PRODUCT"))
            index = (chunk["id"] if len(reactions) == 1
                     else f"{chunk['id']}_{item.get('index')}")
            validation = (item.get("validation") or {}).get("rxn_validation") or {}
            quality = item.get("quality") or (item.get("validation") or {}).get("quality") or {}
            participants = (validation.get("normalized_compounds") or {}).get("participants") or []
            details = item.get("validation") or item.get("coref") or {}
            eligible = bool(quality.get("admission_eligible") and record.get("status") == "ok"
                            and quality.get("version") == QUALITY_VERSION and not quality.get("blocking_issues")
                            and item.get("status") == "ok" and validation.get("status") in ("passed", "corrected")
                            and not validation.get("needs_human_review", True))
            def encoded(value):
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            rows.append({
                "Sample": sample, "Index": index,
                "Reactants": "; ".join(x for x in (reactants, _names(participants)) if x != "N/A") or "N/A", "Reactant amounts": "N/A",
                "Products": products, "Product amounts": "N/A",
                "Solvents": _names(ner.get("SOLVENT")),
                "Reaction temperature": _scalar(ner.get("TEMPERATURE")),
                "Reaction time": _scalar(ner.get("TIME")),
                "Yield": _scalar(ner.get("YIELD_PERCENT")),
                # Reaction-level and record-level status are two different
                # counts (v7 review P-13: passed 102 vs ok 94); carrying both
                # makes the table self-consistent under either reading.
                "rxn_status": validation.get("status") or item.get("status")
                              or record.get("status") or "unknown",
                "record_status": record.get("status") or "unknown",
                "process_kind": chunk.get("process_kind") or "",
                "diagnosis": validation.get("diagnosis", ""),
                "step_id": item.get("step_id", ""), "step_status": item.get("status", "unknown"),
                "Reagents": _names(ner.get("REAGENT_CATALYST")), "Participants": _names(participants),
                "effective_expression": validation.get("effective_expression") or
                    (validation.get("best_expression") if validation.get("patch_accepted") else validation.get("base_expression", "")),
                "step_source": encoded(item.get("step_source") or {}),
                "procedure_references": encoded((item.get("step_source") or {}).get("procedure_references") or []),
                "field_evidence": encoded(details.get("field_evidence") or {}),
                "yield_info": encoded(details.get("yield_info") or {}),
                "stereochemistry": encoded(details.get("stereochemistry") or {}),
                "structure_status": quality.get("structure_status", "unknown"),
                "record_completeness": quality.get("record_completeness", "unknown"),
                "blocking_issues": encoded((quality.get("blocking_issues") or []) if quality else ["quality_evidence_missing"]),
                "admission_eligible": str(eligible).lower(), "quality_version": quality.get("version", ""),
            })
    return rows


def export_table(run_root: Path, output_csv: Path, trusted_only: bool = False) -> int:
    run_root = run_root.resolve()
    folders = [path for path in sorted(run_root.iterdir())
               if path.is_dir() and (path / "results.jsonl").is_file()]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_csv.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=COLUMNS)
        writer.writeheader()
        for folder in folders:
            for row in _rows(folder.name, folder):
                if trusted_only and row["admission_eligible"] != "true":
                    continue
                writer.writerow(row)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-only", action="store_true")
    args = parser.parse_args()
    count = export_table(args.run_root, args.output, args.trusted_only)
    print(f"rows={count} output={args.output}")


if __name__ == "__main__":
    main()
