"""Two business JSONL files per PDF; evidence and control live in artifacts/."""

from copy import deepcopy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .results import latest_by_id, legacy_completed_record
from .routes import COMPOUND_FIELDS, assemble_routes, pick
from .business import EXPORTER_VERSION, QUALITY_VERSION, business_records, encoded, digest, file_hash, filename_ids
from .target_extractor import FINAL_PRODUCT_VERSION


SOURCE_KEYS = ("id", "reference_id", "kind", "number", "title", "source", "process_kind",
               "page_start", "page_end", "block_start", "block_end", "block_indices", "references")
COMPOUND_KEYS = ("name", "standard_name", "canonical_smiles", "source_field", "source_index",
                 "status", "ignored", "reason", "structure_source", "cxsmiles", "stereochemistry",
                 "identifier_only", "compound_class", "auxiliary", "amount", "unit",
                 "resolution_status", "resolution_trace", "discrete_form_evidence", "rejected_structure_smiles", "structure_rejection")
RV_KEYS = ("status", "diagnosis", "reason", "needs_human_review", "mapping_confidence_best", "mass_check", "atom_source_check",
           "base_expression", "best_expression", "effective_expression", "patch_accepted",
           "normalized_compounds")
DETAIL_KEYS = ("ner", "action", "yield_info", "stereochemistry", "process_kind", "record_kind",
              "action_parse_details", "ner_parse_diagnostics", "source_conflicts", "extraction_configuration", "extraction_retry", "reactant_side_participants", "operation_conditions")


def _project_compounds(values, normalized=False):
    if not isinstance(values, list):
        return deepcopy(values)
    keys = COMPOUND_KEYS if normalized else COMPOUND_KEYS + ("smiles",)
    return [deepcopy(pick(value, keys)) if isinstance(value, dict) else deepcopy(value) for value in values]


def compact_record(record):
    result = deepcopy(pick(record, ("id", "chunk_fingerprint", "context_version", "status", "reason", "position", "error", "terminal_product_coverage", "non_reaction_units")))
    if "source" in record:
        result["source"] = deepcopy(pick(record["source"], SOURCE_KEYS))
    if "analysis_conditions" in record:
        result["analysis_conditions"] = deepcopy(record["analysis_conditions"])
    if "reactions" not in record:
        return result
    if record["reactions"] is None:
        result["reactions"] = None
        return result
    result["reactions"] = []
    for reaction in record["reactions"]:
        item = deepcopy(pick(reaction, ("index", "step_id", "status", "reason", "quality")))
        if "step_source" in reaction:
            context = reaction["step_source"]
            if context is None:
                item["step_source"] = None
            else:
                projected = deepcopy(pick(context, ("step_id", "step_index", "title", "process_kind", "record_kind", "local_text",
                    "title_source", "context_issues", "unit_kind", "boundary_status", "source_span")))
                if "title_source" in projected and isinstance(projected["title_source"], dict):
                    projected["title_source"].pop("text", None)
                for key, keys in (("source_parts", ("block_idx", "page_idx", "source", "span")),
                                  ("procedure_references", ("id", "kind", "parent_id", "relation", "status", "source", "blocks"))):
                    if key in context:
                        value = context[key]
                        projected[key] = None if value is None else [deepcopy(pick(v, keys)) for v in value]
                        if key == "procedure_references" and isinstance(projected[key], list):
                            for ref in projected[key]:
                                if isinstance(ref.get("blocks"), list):
                                    ref["blocks"] = [pick(b, ("block_idx", "page_idx", "source", "span")) for b in ref["blocks"]]
                item["step_source"] = projected
        # Exactly one existing extraction layer survives, at its original path.
        layer = next((key for key in ("validation", "coref", "extraction") if reaction.get(key)), None)
        if layer:
            details = reaction[layer]
            projection = deepcopy(pick(details, DETAIL_KEYS))
            rv = details.get("rxn_validation")
            normalized = rv.get("normalized_compounds") if isinstance(rv, dict) else None
            if isinstance(projection.get("ner"), dict):
                if isinstance(normalized, dict):
                    projection["ner"] = {k: v for k, v in projection["ner"].items() if k not in COMPOUND_FIELDS.values()}
                else:
                    for key in COMPOUND_FIELDS.values():
                        if key in projection["ner"]:
                            projection["ner"][key] = _project_compounds(projection["ner"][key])
            if "rxn_validation" in details:
                projection["rxn_validation"] = None if rv is None else deepcopy(pick(rv, RV_KEYS))
                if isinstance(normalized, dict):
                    projection["rxn_validation"]["normalized_compounds"] = {
                        role: _project_compounds(values, normalized=True) for role, values in normalized.items()}
            if "reactant_side_participants" in projection:
                projection["reactant_side_participants"] = _project_compounds(projection["reactant_side_participants"])
            item[layer] = projection
            if "quality" not in item and "quality" in details:
                # Preserve the original path when a legacy record has no top-level quality.
                projection["quality"] = deepcopy(details["quality"])
            # Some legacy validators return only rxn_validation. Keep business
            # fields at their actual fallback path rather than silently losing them.
            if layer == "validation" and "ner" not in details:
                fallback = next((k for k in ("coref", "extraction") if (reaction.get(k) or {}).get("ner")), None)
                if fallback:
                    data = deepcopy(pick(reaction[fallback], DETAIL_KEYS))
                    for key in COMPOUND_FIELDS.values():
                        if key in (data.get("ner") or {}):
                            if isinstance(normalized, dict):
                                data["ner"].pop(key)
                            else:
                                data["ner"][key] = _project_compounds(data["ner"][key])
                    item[fallback] = data
        result["reactions"].append(item)
    return result


def current_records(source):
    chunks_file = source / "chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    fingerprints = {c["id"]: c.get("chunk_fingerprint") for c in chunks}
    latest = latest_by_id(source / "results.jsonl", fingerprints)
    def order(c):
        return (c.get("block_start") if c.get("block_start") is not None else float("inf"),
                c.get("page_start") or 0, c["id"])
    return [latest[c["id"]] for c in sorted(chunks, key=order) if c["id"] in latest], chunks


def _atomic_directory(output, files):
    output = Path(output).resolve()
    owned = any((_read_object(output / name).get("exporter_version") in {EXPORTER_VERSION, "2026-10-07-business-v1", "2026-10-07-business-v2-filename"}
                 or _read_object(output / name).get("delivery_version"))
                for name in ("delivery_manifest.json", "artifacts/manifest.json", "manifest.json"))
    if output.exists() and not owned:
        raise ValueError(f"Refusing to replace a non-delivery directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    size = sum(value.stat().st_size if isinstance(value, Path) else len(value.encode("utf-8")) for value in files.values())
    if shutil.disk_usage(output.parent).free < size + 1024 * 1024:
        raise OSError("Insufficient writable space for delivery staging")
    with tempfile.TemporaryDirectory(prefix=".chemrex-delivery-", dir=output.parent) as scratch:
        staged = Path(scratch) / "new"
        staged.mkdir()
        for relative, value in files.items():
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, Path):
                shutil.copyfile(value, path)
            else:
                path.write_text(value, encoding="utf-8")
        # Validate the bytes actually staged, including portable source evidence.
        for manifest_path in staged.glob("**/artifacts/manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "business_files" not in manifest:
                continue
            folder = manifest_path.parent.parent
            for name, expected in {**manifest["business_files"], **manifest["archive_files"]}.items():
                actual = folder / name
                if file_hash(actual) != expected["sha256"] or actual.stat().st_size != expected["bytes"]:
                    raise ValueError(f"Staged file integrity mismatch: {name}")
        previous = Path(scratch) / "previous"
        if output.exists():
            os.replace(output, previous)
        try:
            os.replace(staged, output)
        except Exception:
            if previous.exists():
                os.replace(previous, output)
            raise


def _read_object(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _pdf_for(source, explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Original PDF unavailable: {path}")
        return path, path.name
    identity = _read_object(source / "source_identity.json")
    if identity.get("pdf_path"):
        return _pdf_for(source, identity["pdf_path"])
    # A single local original is unambiguous. No fuzzy filename matching.
    local = sorted(p for p in source.glob("*.pdf") if p.is_file())
    if len(local) == 1:
        return local[0], local[0].name
    source_path = _read_object(source / "manifest.json").get("source")
    if source_path:
        parsed = Path(source_path)
        for parent in parsed.parents:
            samples = parent / "samples_manifest.json"
            if samples.is_file():
                matches = [s for s in _read_object(samples).get("samples", []) if s.get("sample") == source.name]
                if len(matches) == 1:
                    original = Path(matches[0]["original_pdf"])
                    copy = parent / "original_pdf" / (source.name + ".pdf")
                    available = original if original.is_file() else copy
                    if available.is_file():
                        return available, original.name
            copy = parent / "original_pdf" / (source.name + ".pdf")
            if copy.is_file():
                return copy, copy.name
    raise FileNotFoundError("Original PDF required for document identity; supply --pdf / --pdf-map or source_identity.json")


def _prepare_document(source, pdf=None, document_id=None):
    source = Path(source).resolve()
    pdf, original_name = _pdf_for(source, pdf)
    pdf_sha = file_hash(pdf)
    previous = _read_object(source / "delivery/artifacts/manifest.json")
    identity = _read_object(source / "source_identity.json")
    if document_id is None:
        if previous.get("document_id_rule") and previous.get("original_filename") == original_name:
            document_id = previous["document_id"]
        else:
            document_id = identity.get("document_id") or original_name
    if document_id not in {original_name, f"{source.name}/{original_name}"}:
        raise ValueError("Document ID must match PDF filename or result-folder/filename")
    inputs = {p.name: file_hash(p) for p in (source / "chunks.jsonl", source / "results.jsonl", source / "document_result.json") if p.is_file()}
    if "results.jsonl" not in inputs:
        raise FileNotFoundError("results.jsonl missing; cannot publish an empty result as success")
    records, chunks = current_records(source)
    original_document = _read_object(source / "document_result.json")
    final_product = original_document.get("final_product")
    cached_final = _read_object(source / "final_product.json")
    if isinstance(cached_final, dict) and cached_final.get("_version") == FINAL_PRODUCT_VERSION:
        # Same resume rule as run_document: a reprocessed document keeps its
        # recomputed final product in the versioned cache, while
        # document_result.json may hold only a deferred stub.
        final_product = cached_final
    groups, unassigned = assemble_routes(records, final_product)
    reactions, routes, id_map = business_records(document_id, records, groups)
    present = {r["id"] for r in records}
    missing = [c["id"] for c in chunks if c["id"] not in present]
    failures = [{"chunk_id": r["id"], "status": r.get("status"), "reason": r.get("error") or r.get("reason")}
                for r in records if r.get("status") not in {"ok", "needs_review", "non_reaction"}]
    failures.extend({"chunk_id": r["id"], "status": r.get("status"),
                     "reason": "legacy_result_missing_reactions; rerun extraction"}
                    for r in records if legacy_completed_record(r))
    for r in records:
        for reaction in r.get("reactions") or []:
            if reaction.get("status") in {"error", "failed"}:
                failures.append({"chunk_id": r["id"], "reaction_index": reaction["index"],
                                 "status": reaction["status"], "reason": reaction.get("error") or reaction.get("reason")})
    final = original_document.get("final_product") or {}
    if isinstance(final, dict) and final.get("error"):
        failures.append({"kind": "final_product", "reason": final["error"]})
    failed_images = original_document.get("image_failures", 0)
    if failed_images:
        failures.append({"kind": "image_extraction", "count": failed_images})
    known_complete = original_document.get("status") in {"procedure_processed", "procedure_unverified", "no_reaction_found"}
    status = "complete" if known_complete and not missing and not failures else "partial" if records else "failed"
    business_files = {"reactions.jsonl": "".join(encoded(r) + "\n" for r in reactions),
                      "routes.jsonl": "".join(encoded(r) + "\n" for r in routes)}
    file_info = {name: {"lines": len(reactions) if name == "reactions.jsonl" else len(routes),
                       "bytes": len(value.encode("utf-8")), "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
                 for name, value in business_files.items()}
    document = dict(original_document, routes=groups, unassigned_reactions=unassigned, missing_chunks=missing,
                    filtered_reactions=id_map["filtered_reactions"], document_id=document_id)
    files = dict(business_files)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".log"}:
            name = {"manifest.json": "source_manifest.json", "document_result.json": "raw_document_result.json"}.get(path.name, path.name)
            files["artifacts/" + name] = path
    files.update({"artifacts/document_result.json": encoded(document) + "\n",
                  "artifacts/id_map.json": encoded(id_map) + "\n", "artifacts/source.pdf": pdf})
    archive = {name: {"sha256": file_hash(value) if isinstance(value, Path) else hashlib.sha256(value.encode("utf-8")).hexdigest(),
                      "bytes": value.stat().st_size if isinstance(value, Path) else len(value.encode("utf-8"))}
               for name, value in files.items() if name.startswith("artifacts/")}
    stats = {"chunks": len(records), "missing_chunks": len(missing), "source_reactions": sum(len(r.get("reactions") or []) for r in records),
             "reactions": len(reactions), "routes": len(routes), "target_groups": len({r["target_group_id"] for r in routes}),
             "trusted_steps": sum(r["quality"]["admission_eligible"] for r in reactions),
             "trusted_routes": sum(r["quality"]["admission_eligible"] for r in routes),
             "filtered_reactions": len(id_map["filtered_reactions"]),
             "source_result_bytes": (source / "results.jsonl").stat().st_size,
             "business_bytes": sum(f["bytes"] for f in file_info.values())}
    # Confirmed publication numbers only; Chinese application numbers are not
    # silently relabelled as publication numbers.
    import re
    patents = list(dict.fromkeys(re.findall(r"(?:US\d{7,}(?:[AB]\d?)?|CN\d{9}[AB])", original_name, re.I)))
    manifest = {"document_id": document_id, "pdf_sha256": pdf_sha, "original_filename": original_name,
                "document_id_rule": "filename" if document_id == original_name else "result_folder/filename",
                "patent_numbers": patents, "exporter_version": EXPORTER_VERSION,
                "schema_versions": {"reactions": "reaction-v1", "routes": "route-v1"},
                "policy_version": QUALITY_VERSION, "processing_status": status,
                "missing_chunks": missing, "processing_failures": failures,
                "processing_note": "Status describes identified processing scope, not chemical correctness or full-text recall.",
                "business_files": file_info, "input_sha256": inputs, "archive_files": archive,
                "statistics": stats, "source_label": source.name,
                "archive_lookup": {"raw_records": "artifacts/results.jsonl", "chunks": "artifacts/chunks.jsonl",
                                   "id_map": "artifacts/id_map.json", "pdf": "artifacts/source.pdf"}}
    source_models = original_document.get("source_models")
    if source_models:
        manifest["source_models"] = source_models
    manifest["delivery_revision"] = "delivery_" + digest(manifest)
    files["artifacts/manifest.json"] = encoded(manifest) + "\n"
    if any(file_hash(source / name) != value for name, value in inputs.items()):
        raise ValueError("Source inputs changed while assembling delivery; retry offline")
    return files, manifest


def export_document(source, output=None, pdf=None, document_id=None):
    source = Path(source).resolve()
    output = Path(output).resolve() if output is not None else source / "delivery"
    if source == output or source.is_relative_to(output):
        raise ValueError("Delivery output must not replace its source or a source ancestor")
    files, manifest = _prepare_document(source, pdf, document_id)
    _atomic_directory(output, files)
    return manifest["statistics"]


def try_export_document(source, pdf=None):
    """A delivery error must never schedule another chemical model run."""
    from loguru import logger
    try:
        stats = export_document(source, pdf=pdf)
        return {"status": "ready", "directory": str(Path(source).resolve() / "delivery"), "statistics": stats}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error("delivery export failed for {}: {} (rebuild offline)", source, error)
        return {"status": "error", "error": error}


def export_run(run_root, output_root=None, pdf_map=None):
    """Publish each document locally; return counts without writing a batch table."""
    run_root = Path(run_root).resolve()
    output_root = Path(output_root).resolve() if output_root is not None else None
    if output_root is not None and (run_root == output_root or run_root.is_relative_to(output_root)):
        raise ValueError("Delivery output must not replace the run or a source ancestor")
    folders = [run_root] if (run_root / "chunks.jsonl").is_file() else [
        p for p in sorted(run_root.iterdir()) if p.is_dir() and (p / "chunks.jsonl").is_file()]
    if not folders:
        raise ValueError("No document inputs found")
    if output_root is not None and output_root.exists():
        for child in output_root.iterdir():
            marker = _read_object(child / "artifacts/manifest.json") if child.is_dir() else {}
            if not marker.get("exporter_version"):
                raise ValueError(f"Refusing to add deliveries to a non-delivery directory: {output_root}")
    originals = {source.name: _pdf_for(source, (pdf_map or {}).get(source.name)) for source in folders}
    ids = filename_ids([(label, name) for label, (pdf, name) in originals.items()])
    preferred = dict(ids)
    for source in folders:
        original_name = originals[source.name][1]
        previous = _read_object(source / "delivery/artifacts/manifest.json")
        identity = _read_object(source / "source_identity.json")
        prior_id = previous.get("document_id") if (previous.get("document_id_rule") and
                   previous.get("original_filename") == original_name) else identity.get("document_id")
        if prior_id in {original_name, f"{source.name}/{original_name}"}:
            preferred[source.name] = prior_id
    counts = Counter(preferred.values())
    ids = {label: value if counts[value] == 1 else ids[label] for label, value in preferred.items()}
    per_document = {}
    for source in folders:
        output = output_root / source.name if output_root is not None else source / "delivery"
        if source == output or source.is_relative_to(output):
            raise ValueError("Delivery output overlaps an input document")
        # Pass the selected ID, but retain the original filename from provenance.
        files, manifest = _prepare_document(source, (pdf_map or {}).get(source.name), ids[source.name])
        _atomic_directory(output, files)
        per_document[manifest["document_id"]] = manifest
    totals = {key: sum(m["statistics"][key] for m in per_document.values())
              for key in next(iter(per_document.values()))["statistics"]}
    return {"exporter_version": EXPORTER_VERSION, "documents": per_document, "statistics": totals}
