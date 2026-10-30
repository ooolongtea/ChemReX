"""Build flat cross-document indexes over a ChemReX deliverable root.

Reads {fda,cn}/<uuid>/delivery/{reactions.jsonl,routes.jsonl} under the
deliverable root (the layout described by the root MANIFEST.json) and writes
one flattened record per reaction / per route, so the results can be browsed,
filtered and joined without walking 5863 document folders.

Outputs (under --out):
  reactions_index.jsonl / reactions_index.csv   one row per reaction
  routes_index.jsonl     / routes_index.csv     one row per route
  docs_index.jsonl                              one row per document (enriched documents.jsonl)
  summary.json                                  counts and generation metadata

Usage:
  python scripts/build_deliverable_index.py \
      --root /data/patents/chemrex_deliverable_20261018 \
      --out   /data/patents/chemrex_deliverable_20261018_indexes
"""

import argparse
import csv
import json
import time
from pathlib import Path

REACTION_CSV_COLUMNS = [
    "document_id", "source", "uuid", "reaction_id", "chunk_id", "pages",
    "reaction_smiles", "reactant_smiles", "product_smiles", "reagent_smiles",
    "product_name", "temperature", "time", "yield_value", "yield_kind",
    "admission_eligible", "blocking_issues", "process_kind", "record_kind",
    "n_compounds", "delivery_dir", "source_text",
]

ROUTE_CSV_COLUMNS = [
    "document_id", "source", "uuid", "route_id", "title",
    "target_compound_id", "target_reaction_id", "target_smiles",
    "n_steps", "steps", "admission_eligible", "blocking_issues",
    "delivery_dir",
]


def _iter_jsonl(path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _compound_smiles(compounds, role):
    return ".".join(c["canonical_smiles"] for c in compounds
                    if c.get("canonical_smiles") and role in (c.get("roles") or []))


def _first_product_name(compounds):
    for c in compounds:
        if "product" in (c.get("roles") or []) and c.get("name"):
            return c["name"]
    return None


def _flatten_reaction(rec, source, uuid, delivery_dir):
    compounds = rec.get("compounds") or []
    src = rec.get("source") or {}
    quality = rec.get("quality") or {}
    conditions = rec.get("conditions") or {}
    yld = conditions.get("yield") or {}
    text = (src.get("text") or "").replace("\n", " ")
    return {
        "document_id": rec.get("document_id"),
        "source": source,
        "uuid": uuid,
        "reaction_id": rec.get("reaction_id"),
        "chunk_id": src.get("chunk_id"),
        "pages": src.get("pages") or [],
        "reaction_smiles": rec.get("reaction_smiles"),
        "reactant_smiles": _compound_smiles(compounds, "reactant"),
        "product_smiles": _compound_smiles(compounds, "product"),
        "reagent_smiles": _compound_smiles(compounds, "reagent"),
        "product_name": _first_product_name(compounds),
        "temperature": conditions.get("temperature"),
        "time": conditions.get("time"),
        "yield_value": yld.get("value"),
        "yield_kind": yld.get("kind"),
        "admission_eligible": quality.get("admission_eligible"),
        "blocking_issues": quality.get("blocking_issues") or [],
        "process_kind": rec.get("process_kind"),
        "record_kind": rec.get("record_kind"),
        "n_compounds": len(compounds),
        "delivery_dir": str(delivery_dir),
        "source_text": text[:300],
    }


def _flatten_route(rec, source, uuid, delivery_dir):
    quality = rec.get("quality") or {}
    target = rec.get("target") or {}
    steps = rec.get("steps") or []
    return {
        "document_id": rec.get("document_id"),
        "source": source,
        "uuid": uuid,
        "route_id": rec.get("route_id"),
        "title": rec.get("title"),
        "target_compound_id": target.get("compound_id"),
        "target_reaction_id": target.get("reaction_id"),
        "target_smiles": target.get("canonical_smiles"),
        "n_steps": len(steps),
        "steps": steps,
        "admission_eligible": quality.get("admission_eligible"),
        "blocking_issues": quality.get("blocking_issues") or [],
        "delivery_dir": str(delivery_dir),
    }


def _csv_row(flat, columns):
    row = {}
    for col in columns:
        val = flat.get(col)
        if isinstance(val, list):
            val = ";".join(str(v) for v in val)
        row[col] = "" if val is None else val
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="deliverable root containing {fda,cn}/<uuid>/delivery/ and documents.jsonl")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for the index files")
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    documents_path = root / "documents.jsonl"
    if not documents_path.is_file():
        parser.error(f"missing {documents_path}; is --root a deliverable root?")

    started = time.time()
    n_docs = n_reactions = n_routes = 0
    n_docs_with_reactions = n_docs_with_routes = 0
    n_reaction_smiles = 0

    rx_jsonl_fh = (out / "reactions_index.jsonl").open("w", encoding="utf-8")
    rt_jsonl_fh = (out / "routes_index.jsonl").open("w", encoding="utf-8")
    rx_csv_fh = (out / "reactions_index.csv").open("w", encoding="utf-8", newline="")
    rt_csv_fh = (out / "routes_index.csv").open("w", encoding="utf-8", newline="")
    docs_jsonl_fh = (out / "docs_index.jsonl").open("w", encoding="utf-8")
    rx_jsonl, rt_jsonl, docs_jsonl = rx_jsonl_fh, rt_jsonl_fh, docs_jsonl_fh
    rx_csv = csv.DictWriter(rx_csv_fh, fieldnames=REACTION_CSV_COLUMNS)
    rt_csv = csv.DictWriter(rt_csv_fh, fieldnames=ROUTE_CSV_COLUMNS)
    rx_csv.writeheader()
    rt_csv.writeheader()

    for doc in _iter_jsonl(documents_path):
        n_docs += 1
        docs_jsonl.write(json.dumps(doc, ensure_ascii=False) + "\n")
        delivery_dir = Path(doc["delivery_dir"]) if doc.get("delivery_dir") else None
        if delivery_dir is None or not delivery_dir.is_dir():
            continue

        doc_rx = 0
        for rec in _iter_jsonl(delivery_dir / "reactions.jsonl"):
            flat = _flatten_reaction(rec, doc.get("source"), doc.get("uuid"), delivery_dir)
            rx_jsonl.write(json.dumps(flat, ensure_ascii=False) + "\n")
            rx_csv.writerow(_csv_row(flat, REACTION_CSV_COLUMNS))
            doc_rx += 1
            n_reactions += 1
            if flat["reaction_smiles"]:
                n_reaction_smiles += 1
        if doc_rx:
            n_docs_with_reactions += 1

        doc_rt = 0
        for rec in _iter_jsonl(delivery_dir / "routes.jsonl"):
            flat = _flatten_route(rec, doc.get("source"), doc.get("uuid"), delivery_dir)
            rt_jsonl.write(json.dumps(flat, ensure_ascii=False) + "\n")
            rt_csv.writerow(_csv_row(flat, ROUTE_CSV_COLUMNS))
            doc_rt += 1
            n_routes += 1
        if doc_rt:
            n_docs_with_routes += 1

        if n_docs % 500 == 0:
            print(f"  ... {n_docs} docs, {n_reactions} reactions, {n_routes} routes "
                  f"({time.time() - started:.0f}s)", flush=True)

    for fh in (rx_jsonl_fh, rt_jsonl_fh, docs_jsonl_fh, rx_csv_fh, rt_csv_fh):
        fh.close()

    summary = {
        "root": str(root),
        "documents": n_docs,
        "docs_with_reactions": n_docs_with_reactions,
        "docs_with_routes": n_docs_with_routes,
        "reactions": n_reactions,
        "reactions_with_smiles": n_reaction_smiles,
        "routes": n_routes,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
