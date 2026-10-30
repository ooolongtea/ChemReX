"""Export reaction-v1 and route-v1 PDF snapshots without invoking models."""

import argparse
import json
from pathlib import Path

from pipeline.document.delivery import export_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, help="Optional mirrored result folders; default: each parsed result folder's delivery/")
    parser.add_argument("--pdf", type=Path, help="Original PDF for a single document")
    parser.add_argument("--pdf-map", type=Path, help="JSON object mapping source folder names to original PDF paths")
    args = parser.parse_args()
    pdf_map = json.loads(args.pdf_map.read_text(encoding="utf-8")) if args.pdf_map else {}
    if args.pdf:
        if not (args.run_root / "chunks.jsonl").is_file():
            parser.error("--pdf requires a single document --run-root")
        pdf_map[args.run_root.resolve().name] = str(args.pdf)
    manifest = export_run(args.run_root, args.output_root, pdf_map)
    for control in manifest["documents"].values():
        print(json.dumps({"document_id": control["document_id"], "source_folder": control["source_label"],
                          "statistics": control["statistics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
