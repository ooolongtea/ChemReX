"""Targeted re-extraction for confirmed route gaps.

This is the only route-repair stage that may call a model.  A charter names the
exact document and chunk ids, plus the original-text evidence that justifies
the call. Unchanged inputs may be repaired offline using saved responses;
only changed source inputs may trigger new model calls.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path

from loguru import logger

from .route_repair import (encoded, group_scopes, read_gzip_jsonl, read_jsonl,
                           read_json, scope_file_name, update_state, write_gzip,
                           write_gzip_jsonl, write_json, write_jsonl, write_supplement_charter,
                           _disclosed_records)


def _document_dir(row):
    return Path(row["delivery_dir"]).parent


def _locate_content_list(row):
    document_dir = _document_dir(row)
    prepared = document_dir / "pdf_prepared"
    candidates = sorted(prepared.rglob("*_content_list.json")) if prepared.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no pdf_prepared content_list under {prepared}")
    return candidates[0]


def _image_result(row):
    for path in (Path(row["delivery_dir"]) / "artifacts" / "image_result.json",
                 _document_dir(row) / "image_result.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _set_model_env(item):
    """Pin the model environment before pipeline modules read config."""
    language = item.get("language") or "en"
    preset = item.get("preset") or "huawei-maas"
    os.environ["EXTRACTION_PRESET"] = preset
    if item.get("extraction_model"):
        os.environ["EXTRACTION_MODEL"] = item["extraction_model"]
    provider = item.get("standardize_provider") or "maas"
    model = item.get("llm_model") or os.environ.get("MAAS_MODEL") or "deepseek-v4.1-flash"
    os.environ["STANDARDIZE_PROVIDER"] = provider
    os.environ["COREF_PROVIDER"] = provider
    os.environ["STANDARDIZE_MODEL"] = model
    os.environ["COREF_MODEL"] = model
    os.environ["FINAL_PRODUCT_PROVIDER"] = provider
    os.environ["FINAL_PRODUCT_MODEL"] = model
    os.environ.setdefault("LLM_TIMEOUT", "300")
    # ``config`` may already have been imported by the CLI import chain (the
    # delivery module imports the target extractor). Reload it so the pinned
    # environment, not the repository .env default, selects the model.
    try:
        import importlib
        import config as config_module
        importlib.reload(config_module)
    except Exception:  # config is optional for offline tests
        pass
    return {"preset": preset, "standardize_provider": provider, "llm_model": model}


def _select_chunks(row, chunk_ids):
    from .position_matcher import PositionMatcher
    from .segmenter import load_blocks, segment_blocks, select_procedure_chunks

    content_list = _locate_content_list(row)
    blocks = load_blocks(str(content_list))
    segmented = segment_blocks(blocks, str(content_list))
    chunks = select_procedure_chunks(blocks, str(content_list), segmented)
    matcher = PositionMatcher(blocks)
    for chunk in chunks:
        chunk["position"] = matcher.match_chunk(chunk)
    archived = row.get("archived_sources") or {}
    wanted, assigned = [], defaultdict(list)
    for cid in chunk_ids:
        before = archived.get(cid)
        exact = next((c for c in chunks if c["id"] == cid), None)
        if before:
            candidates = [c for c in chunks if _same_source_chunk(before, c)]
            candidate = exact if exact is not None and _same_source_chunk(before, exact) else (
                candidates[0] if len(candidates) == 1 else None)
        else:
            candidate = exact
        if candidate is None:
            continue
        assigned[candidate["id"]].append(cid)
        selected = deepcopy(candidate)
        if selected["id"] != cid:
            selected["current_chunk_id"] = selected["id"]
            prefix = selected["id"] + ":"
            for step in selected.get("steps") or []:
                if step.get("step_id", "").startswith(prefix):
                    step["step_id"] = cid + ":" + step["step_id"][len(prefix):]
            selected["id"] = cid
        if before:
            # A newly uncertain classifier must not erase the archived
            # disclosure. Keep the evidence, while retaining its new kind.
            selected["candidate_evidence"] = bool(selected.get("candidate_evidence") or before.get("candidate_evidence"))
            selected["has_procedure"] = bool(selected.get("has_procedure") or before.get("has_procedure"))
        wanted.append(selected)
    wanted = [c for c in wanted if len(assigned[c.get("current_chunk_id") or c["id"]]) == 1]
    missing = sorted(set(chunk_ids) - {chunk["id"] for chunk in wanted})
    return wanted, missing, {"content_list": str(content_list), "chunks": len(chunks),
                             "segmentation_version": chunks[0].get("segmentation_version")
                             if chunks else None}


def _same_source_chunk(before, current):
    """Chunk counters are mutable; source positions identify the method."""
    start, other = before.get("block_start"), current.get("block_start")
    if start is not None and start == other:
        return True
    anchor = (before.get("title_source") or {}).get("block_idx")
    other_anchor = (current.get("title_source") or {}).get("block_idx")
    blocks = set(before.get("block_indices") or [])
    overlap = blocks & set(current.get("block_indices") or [])
    return anchor is not None and anchor == other_anchor and bool(overlap)


def _model_value(value):
    """Return a JSON-safe model/provider string (ignore Mock/custom objects)."""
    return value if isinstance(value, str) and value else None


def source_models_from_records(records):
    """Deduplicated M1 model/preset provenance from re-extraction records.

    ``supplement_document`` appends raw ``process_chunk`` records; the extraction
    configuration lives on the record-level ``extraction`` of single-reaction
    records and on every reaction's ``extraction`` for multi-reaction records.
    Only ``model``/``preset`` are kept so the manifest stays small and JSON-safe.
    """
    models, seen = [], set()
    for record in records or []:
        extractions = [record.get("extraction")]
        extractions.extend((reaction or {}).get("extraction")
                           for reaction in record.get("reactions") or [])
        for extraction in extractions:
            configuration = (extraction or {}).get("extraction_configuration") or {}
            if not isinstance(configuration, dict):
                continue
            model = _model_value(configuration.get("model"))
            preset = _model_value(configuration.get("preset"))
            if not model and not preset:
                continue
            key = (model, preset)
            if key in seen:
                continue
            seen.add(key)
            models.append({"model": model, "preset": preset})
    return models


@lru_cache(maxsize=4)
def _online_pipeline(configuration):
    """Keep the model clients and local mapper within one worker process."""
    from pipeline.pipeline import ChemReXPipeline
    return ChemReXPipeline(enable_image=False)


def _reuse_records(records):
    """Expose saved raw responses at the path used by ExtractionReuse."""
    from .step_context import extraction_text
    result = deepcopy(records)
    for record in result:
        for reaction in record.get("reactions") or []:
            source = reaction.get("step_source") or {}
            if not reaction.get("text") and source.get("local_text"):
                source = dict(source, text=source.get("text") or source["local_text"],
                              procedure_references=source.get("procedure_references") or [])
                reaction["text"] = extraction_text(source)
            layer = next((reaction.get(k) for k in ("extraction", "coref", "validation")
                          if isinstance(reaction.get(k), dict) and reaction[k].get("raw")), None)
            if layer:
                reaction["extraction"] = deepcopy(layer)
    return result


def _offline_pipeline(records, deferred, scorer=None):
    # Reuse the existing zero-LLM extraction, naming, coreference and local
    # mapping implementations. None of these clients can issue an LLM request.
    from scripts.reprocess_without_llm_20261016 import (
        CachedScorer, DeferredReuse, OfflineCoref, OfflineExtractor, OfflineNames)
    from pipeline.pipeline import ChemReXPipeline
    from pipeline.standardize.standardizer import Standardizer, COMPOUND_FIELDS
    from pipeline.validation.validator import ReactionValidator
    previous_offline = os.environ.get("STANDARDIZE_OFFLINE")
    os.environ["STANDARDIZE_OFFLINE"] = "1"
    try:
        names = OfflineNames()
    finally:
        if previous_offline is None:
            os.environ.pop("STANDARDIZE_OFFLINE", None)
        else:
            os.environ["STANDARDIZE_OFFLINE"] = previous_offline
    names.saved_names, names.deferred = defaultdict(dict), deferred
    for record in records:
        for reaction in record.get("reactions") or []:
            for key in ("coref", "validation"):
                for field in COMPOUND_FIELDS:
                    for compound in (reaction.get(key) or {}).get("ner", {}).get(field) or []:
                        if not isinstance(compound, dict):
                            continue
                        name = compound.get("name")
                        parsed = names._parse_text_response(compound.get("naming_raw") or "").get(name)
                        if parsed and parsed["standard_name"] == compound.get("standard_name"):
                            names.saved_names[name][parsed["standard_name"]] = parsed
    standardizer = object.__new__(Standardizer)
    standardizer.compound_std = names
    coref = object.__new__(OfflineCoref)
    coref.deferred, coref.auto_standardize, coref._standardizer = deferred, True, standardizer
    pipe = ChemReXPipeline(enable_image=False)
    pipe._extraction = OfflineExtractor(records, deferred)
    pipe._coref = coref
    fresh_scorer = CachedScorer(records, deferred)
    if scorer is None:
        scorer = fresh_scorer
    else:
        scorer.cache.update(fresh_scorer.cache)
        scorer.deferred = deferred
    pipe._validation = ReactionValidator(mapping_scorer=scorer)
    return pipe, DeferredReuse(records, "saved_document_responses", deferred), scorer


def _effective_changes(chunk, previous):
    """Compare actual source inputs, ignoring versions and fingerprints."""
    from .step_context import extraction_text
    reactions = previous.get("reactions") or []
    old = {(r.get("step_source") or {}).get("step_id") or r.get("step_id"):
           r.get("step_source") or {} for r in reactions}
    changes = {}
    for step in chunk.get("steps") or []:
        before = old.get(step["step_id"])
        kinds = []
        if before is None:
            if reactions and step.get("source_parts"):
                kinds.append("step_boundaries_changed")
        else:
            if (before.get("title") or "").strip() != (step.get("title") or "").strip():
                kinds.append("source_product_title_corrected")
            if (before.get("local_text") or "").strip() != (step.get("local_text") or "").strip():
                kinds.append("source_step_text_changed")
            old_refs = [(r.get("id"), r.get("status"), r.get("text"))
                        for r in before.get("procedure_references") or []]
            new_refs = [(r.get("id"), r.get("status"), r.get("text"))
                        for r in step.get("procedure_references") or []]
            old_resolved = {r.get("id") for r in before.get("procedure_references") or [] if r.get("status") == "resolved"}
            if old_refs != new_refs and any(r.get("status") == "resolved" and r.get("id") not in old_resolved
                                           for r in step.get("procedure_references") or []):
                kinds.append("referenced_procedure_context_resolved")
        if kinds and step.get("local_text") and step.get("source_parts"):
            changes[step["step_id"]] = [{"kind": kind,
                "previous_title": (before or {}).get("title"), "current_title": step.get("title"),
                "previous_input": extraction_text(dict(before, text=before.get("text") or before.get("local_text") or "",
                                                       procedure_references=before.get("procedure_references") or [])) if before else None,
                "current_input": extraction_text(step)} for kind in kinds]
        elif step["step_id"] not in (previous.get("model_attempted_step_ids") or []):
            pending = (previous.get("repair_plan") or {}).get(step["step_id"], {}).get("effective_changes") or []
            if pending and all(c.get("current_input") == extraction_text(step) for c in pending):
                changes[step["step_id"]] = deepcopy(pending)
    return changes


def run_prepare(out_root, source_out=None, documents_filter=None, limit=None, chunk_ids=None):
    """Classify all disclosures; repair selected sources using saved responses."""
    from .chunk_processing import process_chunk
    from .extraction_reuse import ExtractionReuse
    from .step_context import CONTEXT_VERSION
    out_root = Path(out_root).resolve()
    if source_out and not (out_root / "scan/documents.jsonl").exists():
        source_out = Path(source_out).resolve()
        for name in ("scan", "assemble", "supplement"):
            if (source_out / name).is_dir():
                shutil.copytree(source_out / name, out_root / name, dirs_exist_ok=True)
    index_rows = read_jsonl(out_root / "scan/documents.jsonl")
    if not index_rows:
        raise SystemExit("scan stage has not run")
    _set_model_env({"preset": "huawei-maas"})
    progress = out_root / "prepare/progress.jsonl"
    progress.parent.mkdir(parents=True, exist_ok=True)
    previous_progress = read_jsonl(progress)
    done = {r["uuid"] for r in previous_progress if r.get("offline_completed")}
    counts = Counter(previous_progress[-1].get("counts") or {}) if previous_progress else Counter()
    changed, scorer = set(done), None
    selected_count = 0
    for number, row in enumerate(index_rows, 1):
        if row["uuid"] in done:
            continue
        cache = out_root / "scan/records" / (row["uuid"] + ".json.gz")
        records = read_gzip_jsonl(cache)
        artifacts = Path(row["delivery_dir"]) / "artifacts"
        archived_sources = {c["id"]: c for c in read_jsonl(artifacts / "chunks.jsonl")}
        shifted = {r["id"] for r in records if r.get("offline_repair") and r["id"] in archived_sources
                   and not _same_source_chunk(archived_sources[r["id"]], r.get("source") or {})}
        if shifted:
            from .delivery import current_records
            canonical = {r["id"]: r for r in current_records(artifacts)[0]}
            for i, record in enumerate(records):
                if record["id"] not in shifted:
                    continue
                original = deepcopy(canonical.get(record["id"]) or {
                    "id": record["id"], "status": "needs_review", "reason": "source_alignment_requires_review"})
                original["source"] = deepcopy(archived_sources[record["id"]])
                records[i] = original
            counts["restored_source_identity_records"] += len(shifted)
        sources = [r["source"] for r in records if (r.get("source") or {}).get("id")]
        disclosures = _disclosed_records(records, sources)
        candidates = {d["chunk_id"] for d in disclosures if d["repair_action"] not in {
            "extracted", "retain_formulation", "retain_nonreaction", "review_purification"}}
        if chunk_ids:
            candidates &= set(chunk_ids)
        selected = (not documents_filter or re.search(documents_filter, row["document_id"]))
        selected = selected and (limit is None or selected_count < limit)
        if candidates and selected:
            selected_count += 1
            row.pop("prepare_error", None)
            try:
                selection_row = dict(row, archived_sources={r["id"]: archived_sources.get(r["id"]) or r.get("source")
                                                           for r in records if r["id"] in candidates})
                chunks, missing, meta = _select_chunks(selection_row, candidates)
                counts["chunks_not_found_in_current_source"] += len(missing)
                counts["aligned_renamed_chunks"] += sum(bool(c.get("current_chunk_id")) for c in chunks)
                for record in records:
                    if record["id"] in missing:
                        record["source_alignment_review"] = {"reason": "no_unique_current_chunk_for_archived_source"}
                        record["repair_plan"] = {}
                    elif record["id"] in {c["id"] for c in chunks}:
                        record.pop("source_alignment_review", None)
                pool = _reuse_records(records + read_jsonl(_document_dir(row) / "results.jsonl"))
                by_id = {r["id"]: r for r in records}
                image = _image_result(row)
                from .run_document import _global_dict, _image_for_chunk, merged_image_dict
                global_dict = _global_dict(image)
                fresh = []
                reuse = ExtractionReuse(pool, allow_unversioned=True)
                pipe, offline_reuse, deferred = None, None, []
                for chunk in chunks:
                    previous = by_id.get(chunk["id"]) or {}
                    changes = _effective_changes(chunk, previous)
                    language = "en" if row.get("source") == "fda" else "zh"
                    decisions = {s["step_id"]: reuse.assess(s, language) for s in chunk.get("steps") or []}
                    # An unchanged, empty failed response has nothing to reuse.
                    if not changes and not any(d["parsed"] for d in decisions.values()):
                        counts["unchanged_without_reusable_response"] += 1
                        continue
                    deferred_start = len(deferred)
                    if pipe is None:
                        pipe, offline_reuse, scorer = _offline_pipeline(pool, deferred, scorer)
                    associated = _image_for_chunk(chunk, image)
                    dictionary = merged_image_dict(global_dict, associated)
                    result = process_chunk(pipe, chunk, language, dictionary, associated, reuse=offline_reuse)
                    result["chunk_fingerprint"] = chunk.get("chunk_fingerprint")
                    result["image"] = associated
                    result["repair_plan"] = {}
                    for step in chunk.get("steps") or []:
                        decision = offline_reuse.assess(step, language, pipe._extraction.configuration(language))
                        tasks = [task for task in ("ner", "action") if decision["needs_" + task]]
                        if changes.get(step["step_id"]) and tasks:
                            result["repair_plan"][step["step_id"]] = {
                                "effective_changes": changes[step["step_id"]], "tasks": tasks}
                    result["offline_repair"] = {"new_llm_calls": 0, "deferred": deepcopy(deferred[deferred_start:]),
                                                "context_version": CONTEXT_VERSION}
                    result = _preserve_unchanged_products(previous, result, chunk)
                    counts["retained_unchanged_product_steps"] += len(
                        result.get("offline_repair", {}).get("retained_step_ids") or [])
                    fresh.append(result)
                    counts["offline_reprocessed_chunks"] += 1
                    counts["steps_with_effective_changes"] += len(changes)
                if fresh:
                    records = merge_records(out_root, row, fresh)
                    sources_by_id = {c["id"]: c for c in sources}
                    sources_by_id.update({c["id"]: c for c in chunks})
                    sources = list(sources_by_id.values())
                    row["supplemented_chunks"] = sorted(set(row.get("supplemented_chunks") or []) | {r["id"] for r in fresh})
                    changed.add(row["uuid"])
            except Exception as exc:
                counts["prepare_errors"] += 1
                row["prepare_error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("offline prepare failed {}", row["document_id"])
        row.update(disclosures=_disclosed_records(records, sources), record_count=len(records))
        row["disclosed_steps"] = len(row["disclosures"])
        # This also persists removal of the old non-reaction placeholders.
        write_gzip_jsonl(cache, records)
        write_json(out_root / "scan/index" / (row["uuid"] + ".json"), row)
        changed.add(row["uuid"])
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(encoded({"uuid": row["uuid"], "counts": dict(counts),
                                  "offline_completed": not chunk_ids and (not candidates or selected)
                                  and not row.get("prepare_error")}) + "\n")
        if number % 50 == 0 or number == len(index_rows):
            logger.info("prepare {}/{} {}", number, len(index_rows), dict(counts))
    # Reload patched rows as well as rows completed before a resume.
    index_rows = [read_json(out_root / "scan/index" / (r["uuid"] + ".json")) for r in index_rows]
    write_jsonl(out_root / "scan/documents.jsonl", index_rows)
    write_supplement_charter(out_root, index_rows)
    invalidated = invalidate_scopes(out_root, changed)
    summary = {"documents": len(index_rows), "counts": dict(counts), "new_llm_calls": 0,
               "classification": read_json(out_root / "scan/repair_classification.json"),
               "model_documents": len(read_json(out_root / "scan/supplement_charter.json").get("items") or []),
               "invalidated_scopes": len(invalidated)}
    write_json(out_root / "prepare/summary.json", summary)
    update_state(out_root, prepare=summary)
    return summary


def supplement_document(out_root, row, item):
    """Re-extract the named chunks of one document with the current code."""
    language = item.get("language") or ("en" if row.get("source") == "fda" else "zh")
    config = _set_model_env(dict(item, language=language))
    from .run_document import _global_dict, _image_for_chunk, merged_image_dict
    from .chunk_processing import process_chunk
    from .extraction_reuse import ExtractionReuse
    from pipeline.pipeline import ChemReXPipeline

    prior = read_gzip_jsonl(Path(out_root) / "scan/records" / (row["uuid"] + ".json.gz"))
    selection_row = dict(row, archived_sources={r["id"]: r.get("source") for r in prior if r["id"] in item["chunks"]})
    chunks, missing, meta = _select_chunks(selection_row, item["chunks"])
    image_data = _image_result(row)
    global_dict = _global_dict(image_data)
    pipe = None
    supplement_dir = Path(out_root) / "supplement" / row["uuid"]
    supplement_dir.mkdir(parents=True, exist_ok=True)
    results_path = supplement_dir / "results.jsonl"
    cached = {record.get("id"): record for record in read_jsonl(results_path)}
    previous = {record["id"]: record for record in prior}
    pool = _reuse_records(prior + read_jsonl(_document_dir(row) / "results.jsonl"))
    reuse = ExtractionReuse(pool, allow_unversioned=True)
    if cached:
        reuse.overlay_recent(ExtractionReuse(_reuse_records(list(cached.values())), allow_unversioned=True))
    records = [cached[chunk["id"]] for chunk in chunks if chunk["id"] in cached
               and _cached_supplement_covers(cached[chunk["id"]], chunk, item.get("step_ids"),
                   merged_image_dict(global_dict, _image_for_chunk(chunk, image_data)))]
    done = {record["id"] for record in records}
    with results_path.open("a", encoding="utf-8") as stream:
        for chunk in chunks:
            if chunk["id"] in done:
                continue
            started = time.monotonic()
            if pipe is None:
                pipe = _online_pipeline(tuple(sorted(config.items())))
            associated = _image_for_chunk(chunk, image_data)
            merged_dict = merged_image_dict(global_dict, associated)
            associated["dict_scope"] = {
                "local_entries": len(associated.get("dict_result") or {}),
                "global_entries": len(global_dict),
                "merged_entries": len(merged_dict),
                "global_dict_used": bool(global_dict),
            }
            selected = set(item.get("step_ids") or [])
            try:
                source = deepcopy(chunk)
                if selected:
                    source["steps"] = [s for s in chunk.get("steps") or [] if s["step_id"] in selected]
                    if not source["steps"]:
                        continue
                record = process_chunk(pipe, source, language, merged_dict, associated, reuse=reuse)
                if selected:
                    record = _combine_selected_record(previous.get(chunk["id"]) or {}, record, selected, chunk)
                record["image"] = associated
            except Exception as exc:  # keep the gap visible, do not hide it
                record = {"id": chunk["id"], "status": "error", "source": chunk,
                          "error": f"{type(exc).__name__}: {exc}"}
                if selected:
                    base = previous.get(chunk["id"]) or {}
                    record["reactions"] = [deepcopy(r) for r in base.get("reactions") or []
                                           if r.get("step_id") in selected]
                    record = _combine_selected_record(base, record, selected, chunk)
            attempted = set((previous.get(chunk["id"]) or {}).get("model_attempted_step_ids") or [])
            record["model_attempted_step_ids"] = sorted(attempted | (selected or {s["step_id"] for s in chunk.get("steps") or []}))
            from .step_context import extraction_text
            snapshots = dict(record.get("model_inputs") or {})
            snapshots.update({s["step_id"]: {"text": extraction_text(s), "material_dictionary": merged_dict}
                              for s in chunk.get("steps") or [] if not selected or s["step_id"] in selected})
            record["model_inputs"] = snapshots
            record["chunk_fingerprint"] = chunk.get("chunk_fingerprint")
            record["elapsed_seconds"] = round(time.monotonic() - started, 2)
            records.append(record)
            stream.write(encoded(record) + "\n")
            stream.flush()
            logger.info("supplement {} {} {}", row["document_id"], chunk["id"], record["status"])
    source_models = source_models_from_records(records)
    manifest = {"document_id": row["document_id"], "uuid": row["uuid"],
                "chunks": list(item["chunks"]), "missing_chunks": missing,
                "reason": item.get("reason"), "evidence": item.get("evidence"),
                "language": language, "model_config": config,
                "source_models": {"m1": source_models},
                "image_source": "delivery/artifacts/image_result.json",
                "content_list": meta["content_list"],
                "segmentation_version": meta["segmentation_version"],
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "records": [{"id": record.get("id"), "status": record.get("status"),
                             "fingerprint": record.get("chunk_fingerprint")}
                            for record in records]}
    write_json(supplement_dir / "manifest.json", manifest)
    return records, manifest


def _cached_supplement_covers(record, chunk, step_ids=None, material_dictionary=None):
    from .step_context import extraction_text
    steps = {s["step_id"]: s for s in chunk.get("steps") or []}
    requested = set(step_ids or steps) & set(steps)
    if not requested:
        return not step_ids and record.get("chunk_fingerprint") == chunk.get("chunk_fingerprint")
    snapshots = record.get("model_inputs")
    if snapshots:
        for sid in requested:
            saved = snapshots.get(sid) or {}
            if saved.get("text") != extraction_text(steps[sid]) or saved.get("material_dictionary") != material_dictionary:
                return False
    elif record.get("chunk_fingerprint") != chunk.get("chunk_fingerprint"):
        return False
    attempted = record.get("model_attempted_step_ids")
    if attempted is None:
        attempted = [r.get("step_id") for r in record.get("reactions") or []
                     if any((r.get(k) or {}).get("raw") for k in ("extraction", "coref", "validation"))]
    return requested <= set(attempted)


def _preserve_unchanged_products(previous, fresh, chunk):
    """A failed offline replay must not erase an unchanged saved extraction."""
    from .route_repair import _disclosure_reactions
    from .routes import compound_lists, _smiles, _material_form, _name
    from .step_context import extraction_text
    if not _same_source_chunk(previous.get("source") or {}, chunk):
        return fresh
    result = deepcopy(fresh)
    retained = []
    for step in chunk.get("steps") or []:
        if step["step_id"] in (fresh.get("model_attempted_step_ids") or []):
            continue
        products = [c for r in _disclosure_reactions(result, step)
                    for c in compound_lists(r).get("products") or []]
        old = _disclosure_reactions(previous, step)
        if not any(compound_lists(r).get("products") for r in old):
            continue
        if products:
            # A parsed name or "Form C" label does not replace a saved valid
            # structure. A fresh valid structure remains authoritative.
            old_structures = any(_smiles(c) for r in old for c in compound_lists(r).get("products") or [])
            def same_parent(p, c):
                base = re.sub(r"[,，]\s*(?:(?:crystalline\s+)?(?:form|polymorph)\s+|晶型\s*)"
                              r"[A-Z0-9αβγδ]+\s*$", "", str(c.get("name") or ""), flags=re.I)
                return ((_smiles(c) and _smiles(p) == _smiles(c)) or
                        (base != c.get("name") and _name(p.get("name")) == _name(base)))
            split_form = any(_material_form(c)
                             and any(same_parent(p, c) and _material_form(p) != _material_form(c) for p in products)
                             and any(not _smiles(p) and _material_form(p) == _material_form(c) for p in products)
                             for r in old for c in compound_lists(r).get("products") or [])
            if not split_form and (not old_structures or any(_smiles(c) for c in products)):
                continue
        contexts = [r.get("step_source") or {} for r in old]
        if not all(context and (context.get("title") or "").strip() == (step.get("title") or "").strip()
                   and (context.get("local_text") or "").strip() == (step.get("local_text") or "").strip()
                   and extraction_text(dict(context,
                text=context.get("text") or context.get("local_text") or "",
                procedure_references=context.get("procedure_references") or [])) == extraction_text(step)
                   for context in contexts):
            continue
        sid = step["step_id"]
        replaced = {id(r) for r in _disclosure_reactions(result, step)}
        result["reactions"] = [r for r in result.get("reactions") or [] if id(r) not in replaced]
        for reaction in deepcopy(old):
            reaction.update(step_id=sid, step_source=deepcopy(step))
            reaction["retained_previous_extraction"] = {"reason": "unchanged_input_offline_replay_lost_product_or_structure"}
            result["reactions"].append(reaction)
        result.get("repair_plan", {}).pop(sid, None)
        retained.append(sid)
    if retained:
        positions = {s["step_id"]: i for i, s in enumerate(chunk.get("steps") or [])}
        result["reactions"].sort(key=lambda r: (positions.get(r.get("step_id"), len(positions)), r.get("index", 0)))
        # Preserve the saved quality flags; retaining data does not promote it.
        if any(r.get("status") == "needs_review" for r in result["reactions"]):
            result["status"] = "needs_review"
        if len(result["reactions"]) > 1:
            result.update(extraction=None, coref=None, validation=None)
        else:
            result.update({key: deepcopy(result["reactions"][0].get(key))
                           for key in ("extraction", "coref", "validation")})
        if chunk["steps"][-1]["step_id"] in retained and "terminal_product_coverage" in previous:
            result["terminal_product_coverage"] = previous["terminal_product_coverage"]
        metadata = result.setdefault("offline_repair", {})
        metadata["retained_step_ids"] = sorted(set(metadata.get("retained_step_ids") or []) | set(retained),
                                                key=lambda sid: positions.get(sid, len(positions)))
    return result


def _combine_selected_record(previous, fresh, selected, chunk):
    """Replace only selected steps; preserve unaffected chemistry and quality."""
    record = dict(deepcopy(previous), **fresh)
    retained = [deepcopy(r) for r in previous.get("reactions") or [] if r.get("step_id") not in selected]
    reactions = retained + list(fresh.get("reactions") or [])
    positions = {step["step_id"]: i for i, step in enumerate(chunk.get("steps") or [])}
    reactions.sort(key=lambda r: (positions.get(r.get("step_id"), len(positions)), r.get("index", 0)))
    record.update(source=deepcopy(chunk), reactions=reactions)
    if len(reactions) > 1:
        record.update(extraction=None, coref=None, validation=None)
    if retained and any(r.get("status") == "needs_review" for r in retained):
        record["status"] = "needs_review"
    if chunk.get("steps") and chunk["steps"][-1]["step_id"] not in selected:
        if "terminal_product_coverage" in previous:
            record["terminal_product_coverage"] = previous["terminal_product_coverage"]
    return record


def _reaction_signature(reaction):
    from .routes import compound_lists, _smiles, _name
    compounds = compound_lists(reaction)
    context = reaction.get("step_source") or {}
    text = context.get("local_text") or reaction.get("text") or ""
    return (" ".join(text.split()), tuple(
        (role, tuple(sorted(_smiles(c) or "name:" + _name(c.get("name")) for c in compounds.get(role) or [])))
        for role in ("reactants", "products", "participants", "reagents", "solvents")))


def _source_blocks(reaction):
    return {(p.get("page_idx"), p["block_idx"])
            for p in (reaction.get("step_source") or {}).get("source_parts") or []
            if p.get("block_idx") is not None}


def _retain_reaction_ids(document_id, previous, original, fresh):
    """Keep identical steps' ids; allocate unused indices for changed steps."""
    from .business import reaction_id
    from .routes import compound_lists, _scope_match
    old_reactions = (previous or {}).get("reactions") or []
    originals = (original or previous or {}).get("reactions") or []
    original_by_index = {r["index"]: r for r in originals}
    reserved = {r["index"] for r in [*old_reactions, *originals]}
    next_index = max(max(reserved, default=0) + 1, (previous or {}).get("next_reaction_index") or 0)
    by_signature = defaultdict(list)
    for reaction in [*old_reactions, *originals]:
        if (reaction["index"] not in original_by_index or
                _reaction_signature(reaction) == _reaction_signature(original_by_index[reaction["index"]])):
            by_signature[_reaction_signature(reaction)].append(reaction)
    used, lineage = set(), defaultdict(list)
    for reaction in fresh.get("reactions") or []:
        match = next((r for r in by_signature[_reaction_signature(reaction)] if r["index"] not in used), None)
        if match is not None:
            reaction["index"] = match["index"]
        else:
            reaction["index"] = next_index
            next_index += 1
        used.add(reaction["index"])
        new_ref = reaction_id(document_id, {"id": fresh["id"], "index": reaction["index"]})
        blocks = _source_blocks(reaction)
        product_list = compound_lists(reaction).get("products") or []
        sources = []
        for old in originals:
            old_blocks = _source_blocks(old)
            same_product = any(_scope_match(p, q) for p in product_list
                               for q in compound_lists(old).get("products") or [])
            if blocks & old_blocks or same_product or not (blocks and old_blocks):
                old_ref = reaction_id(document_id, {"id": fresh["id"], "index": old["index"]})
                sources.append(old_ref)
                sources.extend((old.get("provenance") or {}).get("source_reaction_ids") or [])
                lineage[old_ref].append(new_ref)
        reaction["provenance"] = {
            **(reaction.get("provenance") or {}), "source_reaction_ids": sorted(set(sources)),
            "change": "unchanged" if match is not None else "resegmented_or_reextracted",
            "source_parts": [{k: p[k] for k in ("block_idx", "page_idx", "span") if k in p}
                             for p in (reaction.get("step_source") or {}).get("source_parts") or []]}
    fresh["reaction_lineage"] = [{"old_reaction_id": old_ref, "new_reaction_ids": sorted(set(refs))}
                                 for old_ref, refs in sorted(lineage.items())]
    fresh["next_reaction_index"] = next_index


def merge_records(out_root, row, new_records):
    """Replace the re-extracted chunks in the scan cache and keep source order."""
    cache = Path(out_root) / "scan" / "records" / (row["uuid"] + ".json.gz")
    records = read_gzip_jsonl(cache)
    previous = {record["id"]: record for record in records}
    original = {}
    if row.get("delivery_dir"):
        from .delivery import current_records
        artifacts = Path(row["delivery_dir"]) / "artifacts"
        if (artifacts / "chunks.jsonl").is_file():
            original = {record["id"]: record for record in current_records(artifacts)[0]}
    new_records = deepcopy(new_records)
    for record in new_records:
        _retain_reaction_ids(row["document_id"], previous.get(record["id"]), original.get(record["id"]), record)
    replaced = {record.get("id") for record in new_records}
    records = [record for record in records if record.get("id") not in replaced]
    records.extend(new_records)

    def order(record):
        source = record.get("source") or {}
        block = source.get("block_start")
        return (block if isinstance(block, int) else 10 ** 9,
                source.get("page_start") or 0, record.get("id") or "")

    records.sort(key=order)
    write_gzip_jsonl(cache, records)
    return records


def invalidate_scopes(out_root, uuids):
    """Drop assembled shards for the patent scopes containing these documents."""
    out_root = Path(out_root)
    index_rows = read_jsonl(out_root / "scan" / "documents.jsonl")
    scopes = group_scopes(index_rows)
    scope_ids = set()
    for scope in scopes:
        members = {document["uuid"] for document in scope["documents"]}
        members |= {alias for alias, canonical in scope["aliases"].items() if canonical in members}
        if members & set(uuids):
            scope_ids.add(scope["scope_id"])
    assemble_dir = out_root / "assemble"
    for scope_id in scope_ids:
        shard = assemble_dir / "scopes" / (scope_file_name(scope_id) + ".json.gz")
        if shard.is_file():
            shard.unlink()
    progress = assemble_dir / "progress.jsonl"
    if progress.is_file():
        rows = [row for row in read_jsonl(progress) if row.get("scope_id") not in scope_ids]
        write_jsonl(progress, rows, sort_key=lambda row: row.get("scope_id") or "")
    return sorted(scope_ids)


def update_index(out_root, updates):
    """Patch index rows (supplemented chunk ids) and rewrite the index file."""
    out_root = Path(out_root)
    index_path = out_root / "scan" / "documents.jsonl"
    rows = read_jsonl(index_path)
    for row in rows:
        patch = updates.get(row["uuid"])
        if patch:
            row.update(patch)
        write_json(out_root / "scan" / "index" / (row["uuid"] + ".json"), row)
    write_jsonl(index_path, rows, sort_key=lambda row: (row.get("source") or "", row.get("document_id") or ""))
    write_supplement_charter(out_root, rows)
    return len(updates)


def _supplement_jobs(out_root, jobs, workers):
    if workers <= 1 or len({row["uuid"] for row, _ in jobs}) != len(jobs):
        for row, item in jobs:
            yield supplement_document(out_root, row, item)
        return
    # Workers write only their own document's response files. All scan-cache,
    # index and assembly mutations remain sequential in the parent process.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        yield from pool.map(supplement_document, repeat(out_root),
                            (row for row, _ in jobs), (item for _, item in jobs))


def run_charter(out_root, charter_path, workers=1):
    out_root = Path(out_root).resolve()
    charter = json.loads(Path(charter_path).read_text(encoding="utf-8"))
    index_rows = read_jsonl(out_root / "scan" / "documents.jsonl")
    by_uuid = {row["uuid"]: row for row in index_rows}
    updates, summaries, invalidated = {}, [], set()
    jobs = []
    for item in charter.get("items") or []:
        uuid = item.get("uuid")
        row = by_uuid.get(uuid)
        if row is None:
            target = next((candidate for candidate in index_rows
                           if candidate.get("document_id") == item.get("document_id")), None)
            if target is None:
                raise SystemExit(f"document not found in scan index: {item.get('document_id') or uuid}")
            row, uuid = target, target["uuid"]
        jobs.append((row, item))
    for (row, item), (records, manifest) in zip(jobs, _supplement_jobs(out_root, jobs, max(1, int(workers)))):
        uuid = row["uuid"]
        sources = [r["source"] for r in records if (r.get("source") or {}).get("id")]
        # An error/no-response still leaves a disclosed preparation to export;
        # create its unknown step before allocating ids and saving the cache.
        fresh_disclosures = _disclosed_records(records, sources)
        merged = merge_records(out_root, row, records)
        updated_chunks = {r["id"] for r in records}
        disclosures = [d for d in row.get("disclosures") or [] if d["chunk_id"] not in updated_chunks]
        disclosures.extend(fresh_disclosures)
        updates[uuid] = {"supplemented_chunks": sorted(set(row.get("supplemented_chunks") or []) | set(item["chunks"])),
                         "supplemented_at": manifest["completed_at"],
                         "record_count": len(merged), "disclosed_steps": len(disclosures), "disclosures": disclosures,
                         "reaction_rows": sum(len(record.get("reactions") or [])
                                              for record in merged),
                         "scan_error": None}
        invalidated.update(invalidate_scopes(out_root, [uuid]))
        summaries.append({"document_id": row["document_id"], "uuid": uuid,
                          "chunks": [record.get("id") for record in records],
                          "statuses": [record.get("status") for record in records],
                          "source_models": manifest.get("source_models") or {"m1": []},
                          "manifest": manifest})
    update_index(out_root, updates)
    summary = {"charter": charter.get("charter_id") or Path(charter_path).name,
               "documents": len(summaries), "invalidated_scopes": sorted(invalidated),
               "items": summaries,
               "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(out_root / "supplement" / "summary.json", summary)
    update_state(out_root, supplement={"completed_at": summary["completed_at"],
                                       "documents": len(summaries),
                                       "invalidated_scopes": sorted(invalidated)})
    return {"documents": len(summaries), "invalidated_scopes": sorted(invalidated)}
