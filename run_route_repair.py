"""Route-repair entry point: scan, offline reassembly, targeted supplement, export.

Examples (run from the repository root with the project Python):

    # full batch: scan delivery snapshots, reassemble, export
    python run_route_repair.py all \
        --deliverable /data/patents/chemrex_deliverable_20261018 \
        --out /data/patents/chemrex_batch/route-repair-20261020 \
        --workers 8

    # resumable stages
    python run_route_repair.py scan     --deliverable ... --out ... --workers 8
    python run_route_repair.py assemble --out ...
    python run_route_repair.py export   --out ...

    # targeted re-extraction for confirmed gaps (model calls happen only here)
    python run_route_repair.py supplement --out ... --charter charter.json

The offline stages never call a model.  ``scan`` reads each document's own
``delivery/artifacts`` snapshot; ``assemble`` only uses those archived records;
``export`` writes route-v2, fragments, per-document reaction files, drug
tables (when drug sources are given), the old->new mapping and statistics.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.document import route_repair  # noqa: E402


def cmd_scan(args):
    result = route_repair.run_scan(args.deliverable, args.out, workers=args.workers,
                                   limit=args.limit, force=args.force,
                                   documents_filter=args.documents)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_assemble(args):
    result = route_repair.run_assemble(args.out, limit=args.limit, force=args.force)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _drug_sources(args):
    if args.no_drugs:
        return False
    names = {"fda": args.fda, "cn_metadata": args.cn_metadata,
             "cn_registrations": args.cn_registrations, "chembl": args.chembl}
    return {key: value for key, value in names.items() if value} or True


def cmd_export(args):
    summary = route_repair.run_export(args.out, drug_sources=_drug_sources(args))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_supplement(args):
    from pipeline.document.route_supplement import run_charter
    result = run_charter(args.out, args.charter, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_prepare(args):
    from pipeline.document.route_supplement import run_prepare
    result = run_prepare(args.out, source_out=args.from_out,
                         documents_filter=args.documents, limit=args.limit, chunk_ids=args.chunks)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_all(args):
    out = Path(args.out).resolve()
    summary = {"scan": route_repair.run_scan(args.deliverable, out, workers=args.workers,
                                             limit=args.limit, force=args.force,
                                             documents_filter=args.documents)}
    summary["assemble"] = route_repair.run_assemble(out, limit=None, force=args.force)
    summary["export"] = route_repair.run_export(out, drug_sources=_drug_sources(args))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _add_drug_arguments(parser):
    parser.add_argument("--fda", help="FDA drug_patent_metadata.json")
    parser.add_argument("--cn-metadata", help="CN drug_patent_metadata.jsonl")
    parser.add_argument("--cn-registrations", help="CN registrations.jsonl")
    parser.add_argument("--chembl", help="ChemBL result7_drug_manifest.csv")
    parser.add_argument("--no-drugs", action="store_true",
                        help="skip drug catalog/attribution tables")


def build_parser():
    parser = argparse.ArgumentParser(prog="run_route_repair.py",
                                     description="ChemReX route repair (route-v2)")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="read delivery artifacts into the scan cache")
    scan.add_argument("--deliverable", required=True)
    scan.add_argument("--out", required=True)
    scan.add_argument("--workers", type=int, default=8)
    scan.add_argument("--limit", type=int)
    scan.add_argument("--documents", help="regex filter over documents.jsonl rows")
    scan.add_argument("--force", action="store_true")
    scan.set_defaults(func=cmd_scan)

    assemble = sub.add_parser("assemble", help="reassemble routes at patent scope")
    assemble.add_argument("--out", required=True)
    assemble.add_argument("--limit", type=int)
    assemble.add_argument("--force", action="store_true")
    assemble.set_defaults(func=cmd_assemble)

    export = sub.add_parser("export", help="write route-v2, fragments, mapping, drug tables, stats")
    export.add_argument("--out", required=True)
    _add_drug_arguments(export)
    export.set_defaults(func=cmd_export)

    supplement = sub.add_parser("supplement", help="targeted re-extraction for a charter")
    supplement.add_argument("--out", required=True)
    supplement.add_argument("--charter", required=True)
    supplement.add_argument("--workers", type=int, default=1,
                            help="independent document workers; index updates stay sequential")
    supplement.set_defaults(func=cmd_supplement)

    prepare = sub.add_parser("prepare", help="classify gaps and repair saved responses without LLM calls")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--from-out", help="copy an earlier scan/assembly into an independent output directory")
    prepare.add_argument("--documents", help="regex selecting documents for offline response/source repair")
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--chunks", nargs="+", help="limit offline repair to named chunks")
    prepare.set_defaults(func=cmd_prepare)

    run_all = sub.add_parser("all", help="scan + assemble + export")
    run_all.add_argument("--deliverable", required=True)
    run_all.add_argument("--out", required=True)
    run_all.add_argument("--workers", type=int, default=8)
    run_all.add_argument("--limit", type=int)
    run_all.add_argument("--documents", help="regex filter over documents.jsonl rows")
    run_all.add_argument("--force", action="store_true")
    _add_drug_arguments(run_all)
    run_all.set_defaults(func=cmd_all)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
