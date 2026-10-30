"""Per-document processing for the batch scheduler.

Refactors run_document.py's core flow (segment → position-match → per-chunk
M1→M3→M4 → final product → summary) with two batch-oriented changes:

- chunks run concurrently (ThreadPoolExecutor over LLM API calls);
- no PDF preprocessing and no M2 subprocess here — the scheduler feeds the
  MinerU content_list path and the resident image worker's result dict.

Reuse, not duplication: segmentation / position matching / pipeline modules
and the small run_document helpers are imported from pipeline.document.
"""

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from pipeline.document.position_matcher import PositionMatcher
from pipeline.document.chunk_processing import COMPLETED_STATUSES, process_chunk, extraction_variant
from pipeline.document.results import latest_by_id, legacy_completed_record
from pipeline.document.source_paths import resolve_source, source_reference
from pipeline.document.segmenter import (
    SEGMENTATION_VERSION, load_blocks, segment_blocks, select_procedure_chunks, coverage_gaps,
)
from pipeline.document.run_document import (
    _global_dict,
    _image_for_chunk,
    merged_image_dict,
    _write_json,
)


class PreparedDocument:
    """Segmentation output for one document (cheap to rebuild on resume)."""

    def __init__(self, chunks: List[Dict[str, Any]], matcher: PositionMatcher,
                 manifest: Dict[str, Any]):
        self.chunks = chunks
        self.matcher = matcher
        self.manifest = manifest
        self.extraction_reuse = None


SHORTCIRCUIT_KINDS = ("assay", "analysis", "purification", "formulation")


def select_reclassified(chunks: List[Dict[str, Any]], latest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chunks whose stored record was short-circuited but no longer classify as such.

    Only a segmentation rule change can move a chunk out of the short-circuit
    kinds. Those chunks were never extracted, so their non_reaction record is a
    classification artifact rather than evidence — they must run through M1→M4.
    """
    return [
        chunk for chunk in chunks
        if chunk.get("process_kind") not in SHORTCIRCUIT_KINDS
        and (record := latest.get(chunk["id"])) is not None
        and record.get("status") == "non_reaction"
        and record.get("reason") in SHORTCIRCUIT_KINDS
    ]


def _record_extraction_configurations(record: Dict[str, Any]):
    """Collect M1 provenance from single- and multi-reaction chunk records.

    Returns None when a reaction called M1 but carries no provenance: such a
    record cannot be verified and must not be silently reused.
    """
    configurations = []
    top = (record.get("extraction") or {}).get("extraction_configuration")
    if top:
        configurations.append(top)
    for reaction in record.get("reactions") or []:
        extraction = reaction.get("extraction") or {}
        configuration = extraction.get("extraction_configuration")
        if configuration:
            configurations.append(configuration)
        elif extraction or reaction.get("status") != "non_reaction":
            return None
    return configurations


def _configuration_matches(record: Dict[str, Any], expected: Dict[str, Any],
                           expected_by_step=None) -> bool:
    """Only skip a completed chunk when its M1 model/prompt config matches."""
    configurations = _record_extraction_configurations(record)
    if configurations is None:
        return False
    if not configurations:
        # Short-circuit classifications (assay/analysis/formulation) never
        # called M1 and are independent of the model.
        return (record.get("status") == "non_reaction"
                and record.get("extraction") is None
                and not record.get("reactions"))
    if expected_by_step is None:
        return all(configuration == expected for configuration in configurations)
    top = (record.get('extraction') or {}).get('extraction_configuration')
    reactions = record.get('reactions') or []
    top_expected = (expected_by_step.get(reactions[0].get('step_id'), expected)
                    if len(reactions) == 1 else expected)
    if top and top != top_expected:
        return False
    for reaction in reactions:
        actual = (reaction.get('extraction') or {}).get('extraction_configuration')
        if actual and actual != expected_by_step.get(reaction.get('step_id'), expected):
            return False
    return True


class DocumentProcessor:
    """Runs M1→M3→M4 over every procedure chunk of one document."""

    def __init__(self, language: str = "en", chunk_workers: int = 16,
                 skip_final_product: bool = False, global_dict_enabled: bool = True,
                 no_delivery: bool = False, preset: Optional[str] = None):
        from pipeline.pipeline import ChemReXPipeline

        self.language = language
        self.chunk_workers = max(1, chunk_workers)
        self.skip_final_product = skip_final_product
        self.global_dict_enabled = global_dict_enabled
        self.no_delivery = no_delivery
        self.preset = preset
        self.pipe = ChemReXPipeline(enable_image=False)
        if preset:
            # M1 is routed per language (zh -> local LoRA, en -> MaaS). Set the
            # extractor before pre-warming so every lazy caller sees it.
            from pipeline.extraction.extractor import ChemReXExtractor
            self.pipe._extraction = ChemReXExtractor(preset=preset)
        # Pre-warm module singletons once (thread-race-free lazy loaders).
        extraction = self.pipe._get_extraction()
        client = getattr(extraction, "client", None)
        logger.info("DocumentProcessor language={} m1 preset={} model={}",
                    language, getattr(client, "preset", preset),
                    getattr(client, "model", None))
        self.pipe._get_coref()
        getattr(self.pipe._get_coref(), "standardizer", None)
        self.pipe._get_validation()

    @staticmethod
    def _model_value(obj, name):
        """Return a JSON-safe model/provider string (ignore Mock/custom objects)."""
        value = getattr(obj, name, None)
        return value if isinstance(value, str) and value else None

    def _source_models(self) -> Dict[str, Any]:
        """Provenance for every LLM stage that produced this document."""
        import config as root_config
        # Never trigger lazy construction just to write provenance; __init__
        # pre-warms the real pipeline and offline callers set _extraction.
        extraction = getattr(self.pipe, "_extraction", None)
        extraction_client = getattr(extraction, "client", None)
        # Offline/custom pipelines may set only _coref/_validation and may not
        # have a live standardizer. Prefer already-built objects; never force a
        # new Standardizer just to write provenance.
        standardizer = getattr(self.pipe, "_standardize", None)
        if standardizer is None:
            coref_obj = getattr(self.pipe, "_coref", None)
            standardizer = getattr(coref_obj, "_standardizer", None)
        if standardizer is None and hasattr(self.pipe, "_get_standardize"):
            standardizer = self.pipe._get_standardize()
        coref = self.pipe._get_coref() if hasattr(self.pipe, "_get_coref") else None
        std_api = getattr(getattr(standardizer, "compound_std", None), "api", None)
        coref_api = getattr(coref, "api", None)
        if self.skip_final_product:
            final_product = {"skipped": True}
        else:
            final_model = getattr(root_config, "FINAL_PRODUCT_MODEL", None)
            final_provider = getattr(root_config, "FINAL_PRODUCT_PROVIDER", None)
            final_product = {
                "model": final_model if isinstance(final_model, str) else None,
                "provider": (final_provider
                             if isinstance(final_provider, str) and final_provider
                             else None),
            }
        return {
            "m1": {"model": self._model_value(extraction_client, "model"),
                   "preset": (self._model_value(extraction_client, "preset")
                              or getattr(self, "preset", None))},
            "m3": {"model": self._model_value(std_api, "model"),
                   "provider": self._model_value(std_api, "provider")},
            "m4": {"model": self._model_value(coref_api, "model"),
                   "provider": self._model_value(coref_api, "provider")},
            "final_product": final_product,
        }

    # ── stage 1: segmentation (no LLM calls) ──
    def prepare(self, content_list_file: str, output_dir: Path) -> PreparedDocument:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        from pipeline.document.reuse_sources import load_document_reuse
        reuse = load_document_reuse(output, self.language)
        reuse_root = os.getenv('CHEMREX_REUSE_RESULTS_ROOT')
        if reuse_root:
            from pipeline.document.extraction_reuse import ExtractionReuse
            db_name = Path(os.environ['BATCH_DB_PATH']).stem
            historical = ExtractionReuse.load(Path(reuse_root) / db_name / output.name)
            if reuse is None:
                reuse = historical if historical.inputs else None
            else:
                reuse.include(historical)
        if reuse is not None:
            from pipeline.document.extraction_reuse import ExtractionReuse
            # Preserve work completed since the backup, including partial
            # documents stopped for an operational change. Read before prepare
            # writes the current chunks and changes their fingerprints.
            reuse.overlay_recent(ExtractionReuse.load(output))
        content_list_file = str(resolve_source(content_list_file))
        blocks = load_blocks(content_list_file)
        segmented = segment_blocks(blocks, content_list_file)
        empty_headings = [c["id"] for c in segmented if not c["has_procedure"]]
        chunks = select_procedure_chunks(blocks, content_list_file, segmented)
        matcher = PositionMatcher(blocks)
        for chunk in chunks:
            chunk["position"] = matcher.match_chunk(chunk)
        with (output / "chunks.jsonl").open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        by_base: Dict[str, List[str]] = {}
        for chunk in chunks:
            by_base.setdefault(chunk["reference_id"], []).append(chunk["id"])
        links = [
            {"from": chunk["id"], "mentioned": ref,
             "candidate_targets": by_base.get(ref, []), "relation": "text_reference"}
            for chunk in chunks for ref in chunk["references"]
        ]
        _write_json(output / "links.json", links)
        counts = Counter(chunk["kind"] for chunk in chunks)
        manifest = {
            **source_reference(content_list_file), "total_chunks": len(chunks),
            "segmentation_version": SEGMENTATION_VERSION,
            "coverage_gaps": coverage_gaps(blocks, chunks),
            "by_kind": dict(counts),
            "without_keyword_evidence": sum(not c["candidate_evidence"] for c in chunks),
            "empty_heading_ids": empty_headings,
            "empty_heading_count": len(empty_headings),
            "heading_count": len({c["reference_id"] for c in segmented if c["kind"] == "example"}),
            "covered_headings": len({c["reference_id"] for c in chunks if c["kind"] == "example"}),
            "with_procedure": sum(c["has_procedure"] for c in chunks),
            "cross_page": sum(c["page_start"] != c["page_end"] for c in chunks),
            "position_matched": sum(c["position"]["match_method"] == "block_index" for c in chunks),
            "text_reference_links": len(links),
            "links_file": str(output / "links.json"),
            "chunks_file": str(output / "chunks.jsonl"),
        }
        _write_json(output / "manifest.json", manifest)
        prepared = PreparedDocument(chunks, matcher, manifest)
        prepared.extraction_reuse = reuse
        return prepared

    # ── stage 2: concurrent M1→M3→M4 per chunk ──
    def _process_chunk(self, chunk: Dict[str, Any],
                       merged_dict: Dict[str, str],
                       image_evidence: Dict[str, Any] = None) -> Dict[str, Any]:
        return process_chunk(self.pipe, chunk, self.language, merged_dict, image_evidence)

    def extract(
        self,
        prepared: PreparedDocument,
        output_dir: Path,
        image_data: Dict[str, Any],
        retry_errors: bool = False,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        reextract_reclassified: bool = False,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run all pending chunks; returns the summary dict (also written to disk).

        Resume semantics mirror run_document.py: results.jsonl is append-only,
        each record fsync'd; ok records are skipped on rerun, error records are
        skipped unless *retry_errors* (the scheduler sets it on task retries).
        Legacy completed records without step reactions are re-extracted.
        *reextract_reclassified* additionally re-runs chunks whose stored
        non_reaction record reflects an obsolete short-circuit classification.
        """
        output = Path(output_dir)
        results_path = output / "results.jsonl"
        chunks = [c for c in prepared.chunks if c["has_procedure"]]
        chunks_by_id = {c['id']: c for c in chunks}
        current = {c["id"]: c["chunk_fingerprint"] for c in chunks}
        # An id can survive a changed boundary or text. Only a matching
        # content fingerprint belongs to the current segmentation.
        latest = latest_by_id(results_path, current)
        extractor = getattr(self.pipe, "_extraction", None)
        if extractor is None:
            extractor = self.pipe._get_extraction()
        expected_configuration = (extractor.configuration(self.language)
                                  if hasattr(extractor, "configuration") else None)
        configurations = {None: expected_configuration}

        def configuration_matches(chunk, record):
            if expected_configuration is None:
                return True
            steps = chunk.get('steps') or [chunk]
            variants = {extraction_variant(step) for step in steps}
            for variant in variants:
                if variant not in configurations:
                    configurations[variant] = extractor.configuration(self.language, variant)
            expected = (configurations[next(iter(variants))] if len(variants) == 1
                        else expected_configuration)
            by_step = {step.get('step_id'): configurations[extraction_variant(step)]
                       for step in steps}
            return _configuration_matches(record, expected, by_step)

        completed = {cid for cid, record in latest.items()
                     if record.get("status") in COMPLETED_STATUSES
                     and not legacy_completed_record(record)
                     and configuration_matches(chunks_by_id[cid], record)}
        selected = [c for c in chunks if c["id"] not in completed
                    and (retry_errors or c["id"] not in latest
                         or legacy_completed_record(latest[c["id"]])
                         # Completed status but a different M1 model/prompt:
                         # re-extract instead of silently reusing it.
                         or latest[c["id"]].get("status") in COMPLETED_STATUSES)]
        reclassified_ids: List[str] = []
        if reextract_reclassified:
            selected_ids = {c["id"] for c in selected}
            reclassified = [c for c in select_reclassified(chunks, latest)
                            if c["id"] not in selected_ids]
            reclassified_ids = [c["id"] for c in reclassified]
            selected.extend(reclassified)

        global_dict = _global_dict(image_data) if self.global_dict_enabled else {}
        total = len(selected)
        done = len(completed)
        logger.info("document {}: {} pending chunks, {} completed",
                    output.name, total, len(completed))

        started = time.monotonic()
        grand_total = len(completed) + total
        should_stop = stop_requested or (lambda: False)
        executed = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='final-product') as target_pool, \
                results_path.open("a", encoding="utf-8") as stream:
            target_future = (None if should_stop() else
                             target_pool.submit(self._final_product, prepared, output, chunks))
            if total > 0 and self.chunk_workers <= 1:
                for chunk in selected:
                    if should_stop():
                        break
                    executed.append(chunk['id'])
                    result = self._run_one(chunk, image_data, global_dict, getattr(prepared, 'extraction_reuse', None))
                    self._append(stream, result)
                    done += 1
                    if progress_cb:
                        progress_cb(done, grand_total)
            elif total > 0:
                with ThreadPoolExecutor(max_workers=self.chunk_workers) as pool:
                    pending = iter(selected)
                    futures = set()
                    exhausted = False
                    while futures or not exhausted:
                        while not exhausted and len(futures) < self.chunk_workers:
                            if should_stop():
                                exhausted = True
                                break
                            chunk = next(pending, None)
                            if chunk is None:
                                exhausted = True
                                break
                            executed.append(chunk['id'])
                            futures.add(pool.submit(self._run_one, chunk, image_data, global_dict,
                                                    getattr(prepared, 'extraction_reuse', None)))
                        if not futures:
                            break
                        ready, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in ready:
                            self._append(stream, future.result())
                            done += 1
                            if progress_cb:
                                progress_cb(done, grand_total)

        final_product = target_future.result() if target_future is not None else None

        current_latest = latest_by_id(results_path, current)
        # Analogical examples ("依据实施例X的方法，将A替换为B") whose own
        # extraction failed get an instantiated reaction; ones misclassified
        # as procedure templates are flipped back to concrete instances.
        analogy_repair = None
        if chunks and not should_stop():
            from pipeline.document.analogy_instantiate import apply_analogy_repair
            ordered_records = [current_latest[c["id"]] for c in chunks
                               if c["id"] in current_latest]
            analogy_repair = apply_analogy_repair(ordered_records, chunks)
            if analogy_repair:
                repaired = ({row["chunk_id"] for row in analogy_repair.get("instantiated") or []}
                            | set(analogy_repair.get("reclassified_chunk_ids") or [])
                            | set(analogy_repair.get("refreshed_chunk_ids") or []))
                with results_path.open("a", encoding="utf-8") as stream:
                    for record in ordered_records:
                        if record.get("id") in repaired:
                            self._append(stream, record)
                current_latest = latest_by_id(results_path, current)
        latest_status = {cid: record.get("status", "unknown")
                         for cid, record in current_latest.items()}
        # Only current chunks participate in the summary: an error record of
        # an id that no longer exists can never be fixed by re-running and
        # must not keep the task FAILED forever (review H1).
        status = Counter(latest_status.values())
        summary = dict(
            prepared.manifest,
            result_status=dict(status),
            result_file=str(results_path),
            global_dict_entries=len(global_dict),
            image_failures=len(image_data.get("failed_images") or []),
            failed_images=image_data.get("failed_images") or [],
            needs_review_chunks=status.get("needs_review", 0),
            non_reaction_chunks=status.get("non_reaction", 0),
            # Provenance for reuse audits (v7 review P-17: rerun_chunk_ids in
            # reuse_metadata.json was empty while 44 chunks were re-extracted).
            extracted_chunk_ids=sorted(executed),
            reclassified_chunk_ids=sorted(reclassified_ids),
            analogy_repair=analogy_repair,
            final_product=final_product,
            final_product_file=str(output / "final_product.json")
            if not self.skip_final_product else None,
            source_models=self._source_models(),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        headings = prepared.manifest.get("heading_count", 0)
        covered = prepared.manifest.get("covered_headings", 0)
        product_count = structured_count = 0
        for record in current_latest.values():
            for reaction in record.get("reactions") or []:
                ner = (reaction.get("coref") or {}).get("ner") or {}
                for product in (ner.get("REACTION_PRODUCT") or []) if isinstance(ner, dict) else []:
                    product_count += 1
                    structured_count += bool(isinstance(product, dict) and product.get("smiles"))
        coverage = {
            "identified_examples": headings,
            "covered_examples": covered,
            "processed_examples": sum(
                c["kind"] == "example" and
                current_latest.get(c["id"], {}).get("status") in COMPLETED_STATUSES
                for c in chunks),
            "products_with_structure": structured_count,
            "products_total": product_count,
        }
        if status.get("ok"):
            outcome = "procedure_processed"
        elif status.get("error") and not (status.get("needs_review") or status.get("non_reaction")):
            outcome = "extraction_failed"
        elif any(c.get("process_kind") in ("synthesis", "formulation", "unverified")
                 for c in chunks) and chunks:
            outcome = "procedure_unverified"
        else:
            outcome = "no_reaction_found"
        document_result = {
            "status": outcome,
            "procedure_chunks": len(chunks),
            "successful_chunks": status.get("ok", 0),
            "needs_review_chunks": status.get("needs_review", 0),
            "non_reaction_chunks": status.get("non_reaction", 0),
            "image_failures": len(image_data.get("failed_images") or []),
            "coverage": coverage,
            "final_product": final_product,
            "source_models": summary["source_models"],
            "result_file": str(results_path),
            "reason": None if chunks else
                "No extractable synthesis procedure was found in the parsed PDF text",
        }
        if prepared.manifest.get('document_chunks', len(chunks)) > len(chunks):
            document_result['status'] = 'partial'
            document_result['reason'] = 'Selected source chunks only; not full-document acceptance'
        missing = [c['id'] for c in chunks if c['id'] not in current_latest]
        if should_stop() or missing:
            document_result['status'] = 'partial'
            document_result['reason'] = 'Safely stopped before full-document completion' if should_stop() else 'Unprocessed chunks remain'
            document_result['missing_chunks'] = missing
        summary["coverage"] = coverage
        _write_json(output / "document_result.json", document_result)
        summary["document_result_file"] = str(output / "document_result.json")
        summary["extraction_outcome"] = document_result["status"]
        if not self.no_delivery:
            from pipeline.document.delivery import try_export_document
            summary["delivery"] = try_export_document(output)
        _write_json(output / "summary.json", summary)
        return summary

    # ── helpers ──
    def _final_product(self, prepared, output, chunks):
        """Independent of chunk extraction; keep the same resume/error semantics."""
        if self.skip_final_product:
            return None
        from pipeline.document.target_extractor import FINAL_PRODUCT_VERSION, TargetExtractor
        final_path = output / 'final_product.json'
        cached = json.loads(final_path.read_text(encoding='utf-8')) if final_path.exists() else None
        if isinstance(cached, dict) and cached.get('_version') == FINAL_PRODUCT_VERSION:
            return cached
        if not any(c.get('process_kind') not in SHORTCIRCUIT_KINDS for c in chunks):
            return None
        try:
            final_product = TargetExtractor().extract(prepared.matcher.head_text(self.language), self.language)
            if final_product:
                final_product['position'] = prepared.matcher.match_reference(final_product.get('reference') or '')
                if final_product['position'] is None:
                    final_product = None
                else:
                    final_product['_version'] = FINAL_PRODUCT_VERSION
            _write_json(final_path, final_product)
            return final_product
        except Exception as exc:
            # Failed calls remain uncached and trigger the scheduler's retry.
            error = f'{type(exc).__name__}: {exc}'
            logger.warning('final product failed: {}', error)
            return {'error': error}

    def _run_one(self, chunk: Dict[str, Any], image_data: Dict[str, Any],
                 global_dict: Dict[str, str], reuse=None) -> Dict[str, Any]:
        start = time.monotonic()
        associated_image = _image_for_chunk(chunk, image_data)
        local_dict = associated_image["dict_result"]
        conflicts = {key: {"global": global_dict[key], "local": local_dict[key]}
                     for key in global_dict.keys() & local_dict.keys()
                     if global_dict[key] != local_dict[key]}
        if conflicts:
            logger.warning("chunk {}: image ID conflicts (local wins): {}",
                           chunk["id"], conflicts)
        merged_dict = merged_image_dict(global_dict, associated_image)
        associated_image["dict_scope"] = {
            "local_entries": len(local_dict),
            "global_entries": len(global_dict),
            "merged_entries": len(merged_dict),
            "global_dict_used": bool(global_dict),
            "id_conflicts": len(conflicts),
        }
        try:
            result = (process_chunk(self.pipe, chunk, self.language, merged_dict, associated_image, reuse)
                      if reuse else self._process_chunk(chunk, merged_dict, associated_image))
            result["image"] = associated_image
        except Exception as exc:
            # Keep the full traceback in the scheduler log; the result record
            # only carries the short error string.
            logger.exception("chunk {} extraction failed", chunk["id"])
            result = {"id": chunk["id"], "status": "error", "source": chunk,
                      "error": f"{type(exc).__name__}: {exc}"}
        result["chunk_fingerprint"] = chunk["chunk_fingerprint"]
        result["elapsed_seconds"] = round(time.monotonic() - start, 2)
        return result

    @staticmethod
    def _append(stream, result: Dict[str, Any]) -> None:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
