"""Verify portable PDF business snapshots, hashes and chemical references offline."""

import argparse
import json
from pathlib import Path

from pipeline.document.business import digest, file_hash, validate_snapshot


def verify_document(folder, source=None):
    folder = Path(folder)
    if not (folder / "reactions.jsonl").is_file() and (folder / "delivery/artifacts/manifest.json").is_file():
        source, folder = folder, folder / "delivery"
    control = json.loads((folder / "artifacts/manifest.json").read_text(encoding="utf-8"))
    revision = control["delivery_revision"]
    payload = {k: v for k, v in control.items() if k != "delivery_revision"}
    if revision != "delivery_" + digest(payload):
        raise ValueError("Delivery revision does not match control payload")
    rule = control.get("document_id_rule")
    expected = control["original_filename"] if rule == "filename" else (
        f"{control['source_label']}/{control['original_filename']}" if rule == "result_folder/filename" else
        "doc_" + control["pdf_sha256"])
    if control["document_id"] != expected:
        raise ValueError("PDF filename identity mismatch")
    if file_hash(folder / "artifacts/source.pdf") != control["pdf_sha256"]:
        raise ValueError("Archived PDF identity mismatch")
    if source is not None:
        source = Path(source)
        if any(not (source / name).is_file() or file_hash(source / name) != value
               for name, value in control["input_sha256"].items()):
            raise ValueError("Source changed since document delivery")
    rows = {}
    for name, expected in {**control["business_files"], **control["archive_files"]}.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Archive path escapes snapshot")
        path = folder / relative
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"File integrity mismatch: {name}")
        if name in control["business_files"]:
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) != expected["lines"]:
                raise ValueError(f"Line count mismatch: {name}")
            rows[name] = [json.loads(line) for line in lines]
            if not all(isinstance(row, dict) for row in rows[name]):
                raise ValueError("JSONL lines must be business objects")
    if set(rows) != {"reactions.jsonl", "routes.jsonl"}:
        raise ValueError("Expected exactly two business files")
    validate_snapshot(control["document_id"], rows["reactions.jsonl"], rows["routes.jsonl"])
    if control["processing_status"] not in {"complete", "partial", "failed"}:
        raise ValueError("Unknown processing status")
    if control["processing_status"] == "complete" and (control["missing_chunks"] or control["processing_failures"]):
        raise ValueError("Complete snapshot contains known processing failures")
    return control


def verify_run(root):
    root = Path(root)
    if (root / "reactions.jsonl").is_file() or (root / "delivery/artifacts/manifest.json").is_file():
        manifests = [verify_document(root)]
    else:
        folders = [p for p in sorted(root.iterdir()) if p.is_dir() and
                   ((p / "chunks.jsonl").is_file() or (p / "reactions.jsonl").is_file())]
        if not folders:
            raise ValueError("No per-document results found")
        manifests = [verify_document(folder) for folder in folders]
        if len({m["document_id"] for m in manifests}) != len(manifests):
            raise ValueError("Duplicate PDF filename IDs; re-export the batch to apply folder namespaces")
    return {"verified_documents": len(manifests),
            "reactions": sum(m["business_files"]["reactions.jsonl"]["lines"] for m in manifests),
            "routes": sum(m["business_files"]["routes.jsonl"]["lines"] for m in manifests),
            "complete_documents": sum(m["processing_status"] == "complete" for m in manifests)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
