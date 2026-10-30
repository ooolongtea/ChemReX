"""Run ChemReX over every title/procedure unit in an ordered PDF extraction."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from .position_matcher import PositionMatcher
from .chunk_processing import COMPLETED_STATUSES, process_chunk
from .results import latest_by_id, legacy_completed_record
from .source_paths import resolve_source, source_reference
from pipeline.chemical_quality import image_label_evidence, normalize_image_label, valid_image_smiles
from pipeline.versions import IMAGE_PIPELINE_VERSION
from .segmenter import SEGMENTATION_VERSION, load_blocks, segment_blocks, select_procedure_chunks, coverage_gaps


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _read_completed(path: Path) -> set:
    completed = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            cid = record.get("id")
            if record.get("status") == "ok" and cid:
                completed.add(cid)
    return completed


def _image_for_chunk(chunk: Dict[str, Any], image_data: Dict[str, Any]) -> Dict[str, Any]:
    names = {Path(name).name for name in chunk["images"]}
    per_image = [item for item in image_data.get("per_image_detections", [])
                 if item.get("image_file_name") in names]
    reactions = [item for item in image_data.get("image_result", [])
                 if item.get("image_file_name") in names]
    return {**image_label_evidence(per_image), "image_result": reactions,
            "per_image_detections": per_image,
            "image_files": sorted(names), "matched_images": len(per_image)}


def merged_image_dict(global_dict, associated_image):
    # A current figure's unspecified/invalid structure is negative local
    # evidence. Do not replace it with another figure's same-number molecule.
    blocked = set(associated_image.get('unusable_identifiers') or [])
    return {**{k:v for k,v in global_dict.items() if k not in blocked},
            **associated_image['dict_result']}


def _global_dict(image_data: Dict[str, Any]) -> Dict[str, str]:
    """Document-level ID→SMILES dictionary: M2 top-level dict_result, or the
    union of all per-image dictionaries when only per-image output exists.

    The same key mapped to different SMILES across sources is kept
    last-write-wins but logged: it usually means two procedures numbered the
    same compound differently, and the loser silently misassigns SMILES
    (doc/batch-code-review-round2.md M6). Markush (wildcard) structures from
    scheme figures are collected separately and only fill labels no concrete
    structure claims; they never evict a concrete entry."""
    merged: Dict[str, str] = {}
    markush: Dict[str, str] = {}
    conflicts: Dict[str, str] = {}

    def _merge(source: Any) -> None:
        for key, value in (source or {}).items():
            if not value:
                continue
            key, value = normalize_image_label(key), str(value)
            if not key:
                continue
            if valid_image_smiles(value):
                if key in merged and merged[key] != value:
                    conflicts[key] = merged[key]
                merged[key] = value
            elif valid_image_smiles(value, allow_markush=True):
                if key in markush and markush[key] != value:
                    conflicts[key] = markush[key]
                else:
                    markush.setdefault(key, value)

    _merge(image_data.get("dict_result"))
    if not merged and not markush:
        for item in image_data.get("per_image_detections", []):
            _merge(item.get("dict_result"))
    for item in image_data.get('per_image_detections') or []:
        for key in image_label_evidence([item])['conflicting_identifiers']:
            conflicts.setdefault(key, merged.get(key) or markush.get(key))
    if conflicts:
        for key in conflicts:
            merged.pop(key, None)
            markush.pop(key, None)
        logger.warning(
            "global image dictionary conflicts (ambiguous keys dropped): {}",
            {key: {"candidate": previous} for key, previous in conflicts.items()})
    result = dict(merged)
    for key, value in markush.items():
        if key not in result:
            result[key] = value
    return result


def _load_image_result(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("image_pipeline_version") != IMAGE_PIPELINE_VERSION:
        raise ValueError(f"outdated image result: {path}; rerun the image stage")
    return payload


def _run_image_once(image_base: str, image_python: str, output: Path) -> Dict[str, Any]:
    if output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("image_pipeline_version") == IMAGE_PIPELINE_VERSION:
            return cached
    command = [image_python, "run_pipeline.py", "image", "--image-base", image_base,
               "--return-per-image", "--output", str(output)]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])
    return _load_image_result(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process all PDF reaction procedures, with resume")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--content-list", help="MinerU ordered *_content_list.json")
    source.add_argument("--pages-json", help="ChemReX PDF pages.json; limited for two-column scans")
    source.add_argument("--pdf", help="PDF to preprocess with MinerU (VLM backend) before segmentation")
    parser.add_argument("--source-pdf", help="Original PDF identity when using pre-parsed content/pages")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="en", choices=["en", "zh"])
    parser.add_argument("--pdf-backend", choices=("mineru", "poppler"),
                        default=os.getenv("PDF_BACKEND", "mineru"),
                        help="Preprocessor for --pdf (default: mineru; set PDF_BACKEND to override)")
    parser.add_argument("--mineru-backend", default=os.getenv("MINERU_BACKEND", "vlm-auto-engine"),
                        help="MinerU parse backend (default: vlm-auto-engine)")
    parser.add_argument("--mineru-url", default=os.getenv("MINERU_URL"),
                        help="Server URL for vlm/hybrid-http-client MinerU backends")
    parser.add_argument("--kind", default="all", choices=["all", "example", "intermediate"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-records", type=int, help="Process at most this many pending records")
    parser.add_argument("--preset", help="Extraction model preset; e.g. huawei-maas")
    parser.add_argument("--llm-provider", help="M3 provider, e.g. maas")
    parser.add_argument("--llm-model", help="M3 model, e.g. deepseek-v4.1-flash")
    parser.add_argument("--offline-standardize", action="store_true")
    parser.add_argument("--image-base", help="Directory containing MinerU images")
    parser.add_argument("--image-result", help="Existing M2 output JSON; reused without rerun")
    parser.add_argument("--image-python", default=os.getenv("IMGPARSER_PYTHON", sys.executable))
    parser.add_argument("--no-global-dict", dest="global_dict", action="store_false",
                        help="Disable the document-level image dictionary; use only "
                             "per-procedure image detections for coref (local entries "
                             "still take precedence over global ones when enabled)")
    parser.add_argument("--skip-final-product", action="store_true",
                        help="Skip the document-level final product extraction from the abstract")
    parser.add_argument("--retry-errors", action="store_true", help="Retry records previously marked error")
    parser.add_argument("--no-delivery", action="store_true", help="Disable compact route delivery after extraction")
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.pdf or args.source_pdf:
        _write_json(output / "source_identity.json", {"pdf_path": str(Path(args.pdf or args.source_pdf).expanduser().resolve())})
    if args.pdf:
        if args.pdf_backend == "mineru":
            from pipeline.pdf.pdf_pipeline import process_pdf_mineru
            pdf_prepared = process_pdf_mineru(
                args.pdf, str(output / "pdf_prepared"), args.language,
                backend=args.mineru_backend, server_url=args.mineru_url)
            source_path = Path(pdf_prepared["content_list_file"])
        else:
            from pipeline.pdf.pdf_pipeline import process_pdf
            pdf_prepared = process_pdf(args.pdf, str(output / "pdf_prepared"), args.language)
            source_path = Path(pdf_prepared["pages_file"])
    else:
        source_path = Path(args.content_list or args.pages_json).expanduser().resolve()
    source_path = resolve_source(str(source_path))
    blocks = load_blocks(str(source_path))
    segmented = segment_blocks(blocks, str(source_path))
    empty_headings = [c["id"] for c in segmented if not c["has_procedure"]]
    chunks = select_procedure_chunks(blocks, str(source_path), segmented)
    # Position matching against the ordered content_list blocks: every unit
    # gets page_idx + bbox (block-index primary, text-match fallback).
    matcher = PositionMatcher(blocks)
    for chunk in chunks:
        chunk["position"] = matcher.match_chunk(chunk)
    if args.kind != "all":
        chunks = [chunk for chunk in chunks if chunk["kind"] == args.kind]
    with (output / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    by_base = {}
    for chunk in chunks:
        by_base.setdefault(chunk["reference_id"], []).append(chunk["id"])
    links = [
        {"from": chunk["id"], "mentioned": ref, "candidate_targets": by_base.get(ref, []),
         "relation": "text_reference"}
        for chunk in chunks for ref in chunk["references"]
    ]
    _write_json(output / "links.json", links)
    counts = Counter(chunk["kind"] for chunk in chunks)
    manifest = {**source_reference(str(source_path)), "total_chunks": len(chunks),
                "segmentation_version": SEGMENTATION_VERSION,
                "coverage_gaps": coverage_gaps(blocks, chunks),
                "without_keyword_evidence": sum(not c["candidate_evidence"] for c in chunks),
                "empty_heading_ids": empty_headings,
                "empty_heading_count": len(empty_headings),
                "heading_count": len({c["reference_id"] for c in segmented if c["kind"] == "example"}),
                "covered_headings": len({c["reference_id"] for c in chunks if c["kind"] == "example"}),
                "by_kind": dict(counts), "with_procedure": sum(c["has_procedure"] for c in chunks),
                "cross_page": sum(c["page_start"] != c["page_end"] for c in chunks),
                "position_matched": sum(c["position"]["match_method"] == "block_index" for c in chunks),
                "text_reference_links": len(links), "links_file": str(output / "links.json"),
                "chunks_file": str(output / "chunks.jsonl")}
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return

    if args.preset:
        os.environ["EXTRACTION_PRESET"] = args.preset
    if args.llm_provider:
        os.environ["STANDARDIZE_PROVIDER"] = args.llm_provider
        os.environ["COREF_PROVIDER"] = args.llm_provider
    if args.llm_model:
        os.environ["STANDARDIZE_MODEL"] = args.llm_model
        os.environ["COREF_MODEL"] = args.llm_model
    if args.offline_standardize:
        os.environ["STANDARDIZE_OFFLINE"] = "1"

    image_data = {}
    if args.image_result:
        image_data = _load_image_result(Path(args.image_result))
    elif args.image_base:
        image_data = _run_image_once(args.image_base, args.image_python, output / "image_result.json")
    global_dict = _global_dict(image_data) if args.global_dict else {}
    print(f"Image dictionary: global={len(global_dict)} entries"
          f"{' (local entries take precedence)' if global_dict else ''}", flush=True)

    from pipeline.pipeline import ChemReXPipeline
    pipe = ChemReXPipeline(enable_image=False)
    results_path = output / "results.jsonl"
    current = {c["id"]: c["chunk_fingerprint"] for c in chunks}
    latest = latest_by_id(results_path, current)
    completed = {cid for cid, record in latest.items()
                 if record.get("status") in COMPLETED_STATUSES
                 and not legacy_completed_record(record)}
    selected = [c for c in chunks if c["has_procedure"] and c["id"] not in completed
                and (args.retry_errors or c["id"] not in latest
                     or legacy_completed_record(latest[c["id"]]))]
    if args.max_records is not None:
        selected = selected[:args.max_records]
    print(f"Processing {len(selected)} pending chunks; {len(completed)} completed", flush=True)
    with results_path.open("a", encoding="utf-8") as stream:
        for index, chunk in enumerate(selected, 1):
            start = time.monotonic()
            associated_image = _image_for_chunk(chunk, image_data)
            # Coref dictionary: document-level dict_result applied globally, with
            # the procedure's own image detections taking precedence on conflicts.
            local_dict = associated_image["dict_result"]
            merged_dict = merged_image_dict(global_dict, associated_image)
            associated_image["dict_scope"] = {
                "local_entries": len(local_dict),
                "global_entries": len(global_dict),
                "merged_entries": len(merged_dict),
                "global_dict_used": bool(global_dict),
            }
            try:
                # Run M1, then M3 and M4 on the same procedure. M2 is computed
                # once for the document; local image detections take precedence,
                # the document-level dictionary fills the remaining identifiers.
                result = process_chunk(pipe, chunk, args.language, merged_dict, associated_image)
                result["image"] = associated_image
            except Exception as exc:
                result = {"id": chunk["id"], "status": "error", "source": chunk,
                          "error": f"{type(exc).__name__}: {exc}"}
            result["elapsed_seconds"] = round(time.monotonic() - start, 2)
            result["chunk_fingerprint"] = chunk["chunk_fingerprint"]
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(f"{index}/{len(selected)} {chunk['id']} {result['status']} "
                  f"{result['elapsed_seconds']}s", flush=True)

    # Document-level final product (route direction): extracted once from the
    # document head / abstract region, position-matched, cached for resume.
    final_product = None
    if not args.skip_final_product:
        from .target_extractor import FINAL_PRODUCT_VERSION, TargetExtractor
        final_path = output / "final_product.json"
        cached = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
        if isinstance(cached, dict) and cached.get("_version") == FINAL_PRODUCT_VERSION:
            final_product = cached
            print(f"Final product: reused from {final_path.name}", flush=True)
        else:
            try:
                head = matcher.head_text(args.language)
                final_product = TargetExtractor().extract(head, args.language)
                if final_product:
                    final_product["position"] = matcher.match_reference(
                        final_product.get("reference") or "")
                    if final_product["position"] is None:
                        final_product = None
                    else:
                        final_product["_version"] = FINAL_PRODUCT_VERSION
                _write_json(final_path, final_product)
                name = final_product.get("product") if final_product else None
                print(f"Final product: {name or 'None'}", flush=True)
            except Exception as exc:
                # Not cached — a transient API failure is retried on the next run.
                final_product = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"Final product: failed — {final_product['error']}", flush=True)

    # Only chunks of the CURRENT segmentation participate in the summary:
    # records left over from a previous parse (whose chunk ids no longer
    # exist) must not fail the run (doc/batch-code-review-round2.md H1).
    current_fingerprints = {c["id"]: c["chunk_fingerprint"] for c in chunks}
    current_latest = latest_by_id(results_path, current_fingerprints)
    from .analogy_instantiate import apply_analogy_repair
    ordered_records = [current_latest[c['id']] for c in chunks if c['id'] in current_latest]
    analogy_repair = apply_analogy_repair(ordered_records, chunks)
    if analogy_repair:
        changed = ({r['chunk_id'] for r in analogy_repair.get('instantiated') or []}
                   | set(analogy_repair.get('reclassified_chunk_ids') or [])
                   | set(analogy_repair.get('refreshed_chunk_ids') or []))
        with results_path.open('a', encoding='utf-8') as stream:
            for row in ordered_records:
                if row['id'] in changed:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    status = Counter(row.get("status", "unknown") for row in current_latest.values())
    heading_count = len({c["reference_id"] for c in segmented if c["kind"] == "example"})
    covered_headings = len({c["reference_id"] for c in chunks if c["kind"] == "example"})
    product_count = structured_count = 0
    for row in current_latest.values():
        for reaction in row.get("reactions") or []:
            ner = (reaction.get("coref") or {}).get("ner") or {}
            for product in (ner.get("REACTION_PRODUCT") or []) if isinstance(ner, dict) else []:
                product_count += 1
                structured_count += bool(isinstance(product, dict) and product.get("smiles"))
    coverage = {
        "identified_examples": heading_count, "covered_examples": covered_headings,
        "processed_examples": sum(c["kind"] == "example" and
                                   current_latest.get(c["id"], {}).get("status") in COMPLETED_STATUSES
                                   for c in chunks),
        "products_with_structure": structured_count, "products_total": product_count,
    }
    if status.get("ok"):
        outcome = "procedure_processed"
    elif status.get("error") and not (status.get("needs_review") or status.get("non_reaction")):
        outcome = "extraction_failed"
    elif chunks and any(c.get("process_kind") in ("synthesis", "formulation", "unverified")
                        for c in chunks):
        outcome = "procedure_unverified"
    else:
        outcome = "no_reaction_found"
    document_result = {
        "status": outcome, "procedure_chunks": len(chunks),
        "successful_chunks": status.get("ok", 0),
        "needs_review_chunks": status.get("needs_review", 0),
        "non_reaction_chunks": status.get("non_reaction", 0),
        "image_failures": len(image_data.get("failed_images") or []),
        "coverage": coverage, "final_product": final_product,
        "result_file": str(results_path),
    }
    _write_json(output / "document_result.json", document_result)
    summary = dict(manifest, result_status=dict(status), result_file=str(results_path),
                   image_result=args.image_result or (str(output / "image_result.json") if args.image_base else None),
                   image_failures=len(image_data.get("failed_images") or []),
                   failed_images=image_data.get("failed_images") or [],
                   needs_review_chunks=status.get("needs_review", 0),
                   non_reaction_chunks=status.get("non_reaction", 0),
                   global_dict_entries=len(global_dict), coverage=coverage,
                   extraction_outcome=outcome, analogy_repair=analogy_repair,
                   document_result_file=str(output / "document_result.json"),
                   final_product=final_product,
                   final_product_file=str(output / "final_product.json")
                   if not args.skip_final_product else None)
    if not args.no_delivery:
        from .delivery import try_export_document
        summary["delivery"] = try_export_document(output)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
