"""Patent-scope route repair over an existing delivery snapshot.

Stages (offline by default, resumable):

    scan       read each document's delivery artifacts into a compact cache
    assemble   reassemble routes at confirmed-patent scope with route-v2
    supplement targeted re-extraction for confirmed gaps (explicit charter)
    export     write route-v2, fragments, updated reactions, drug tables,
               old->new mapping and statistics into the output directory

The material extraction chain is reused: the scan stage only reads the
delivery's own archived ``results.jsonl``/``chunks.jsonl`` and never calls a
model.  Only the supplement stage may call the configured extraction model,
and only for the chunk ids named by the caller.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from .business import (QUALITY_VERSION, _pages, business_reaction,
                       compounds as business_compounds,
                       reaction_id as business_reaction_id)
from .delivery import current_records
from .routes import (ROUTE_SCOPE_VERSION, ROUTE_V2_VERSION, TEMPLATE_ROUTE_VERSION,
                     assemble_routes_v2, assemble_template_routes, compound_lists,
                     eligible)

STAGE_VERSION = "route-repair-2026-10-21"
SCHEMA_ROUTE = "route-v2"
SCHEMA_FRAGMENT = "route-fragment-v1"
SCHEMA_REACTION = "reaction-v1"

_DOCUMENT_PATENT = re.compile(
    r"(?<![A-Z0-9])(USRE\d{4,}|US\d{7,}[A-Z]?\d?|CN\d{7,}[A-Z]?|WO\d{9,}[A-Z]?\d?|EP\d{6,}[A-Z]?\d?|ZL\d{9,}\.?[0-9X]?)(?!\d)",
    re.I)
_VOLUME = re.compile(r"(?:part|volume|分卷|部分)?\s*[\(（]\s*(\d{1,2})\s*[\)）]", re.I)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return value if value is not None else {}


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_jsonl(path, rows, sort_key=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if sort_key is not None:
        rows.sort(key=sort_key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(encoded(row) + "\n")
    temp.replace(path)


def write_gzip(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=6) as stream:
        stream.write(encoded(value))
    temp.replace(path)


def read_gzip(path):
    path = Path(path)
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.loads(stream.read())


def write_gzip_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=6) as stream:
        for row in rows:
            stream.write(encoded(row) + "\n")
    temp.replace(path)


def read_gzip_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def update_state(out_root, **stages):
    path = Path(out_root) / "state.json"
    state = read_json(path)
    state.update({"stage_version": STAGE_VERSION, "updated_at": _now()})
    for name, payload in stages.items():
        state[name] = payload
    write_json(path, state)
    return state


# ---------------------------------------------------------------------------
# Patent identity
# ---------------------------------------------------------------------------

def normalize_patent_key(value):
    if not value:
        return None
    value = re.sub(r"[\s\-.()（）]", "", str(value)).upper()
    if re.fullmatch(r"CN\d{7,}[A-Z]?", value):
        return value
    match = re.fullmatch(r"(?:US)?(RE?\d{4,})([AB]\d?)?", value)
    if match and len(value) >= 6:
        return "US" + match[1] + (match[2] or "")
    match = re.fullmatch(r"(?:CN|ZL)?(\d{9,}[0-9X]?)([AB])?", value)
    if match:
        return "CN" + match[1] + (match[2] or "")
    if re.fullmatch(r"(?:WO|EP)\d{6,}[A-Z]?\d?", value):
        return value
    return value


def patent_key_from_text(text):
    match = _DOCUMENT_PATENT.search(str(text or ""))
    return normalize_patent_key(match.group(1)) if match else None


def volume_from_name(name):
    match = _VOLUME.search(str(name or ""))
    return int(match.group(1)) if match else None


def document_identity(entry, manifest, identity):
    filename = manifest.get("original_filename") or Path(entry["document_id"]).name
    key, source = None, None
    for patent in manifest.get("patent_numbers") or []:
        normalized = normalize_patent_key(patent)
        if normalized:
            key, source = normalized, "manifest"
            break
    if not key:
        key = patent_key_from_text(filename) or patent_key_from_text((identity or {}).get("pdf_path"))
        source = "filename" if key else None
    return {"filename": filename, "patent_key": key, "patent_key_source": source,
            "volume_index": volume_from_name(filename)}


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

_COMPOUND_FIELDS = ("STARTING_MATERIAL", "REACTION_PRODUCT", "REAGENT_CATALYST", "SOLVENT")


def _detail_with_ner(reaction):
    return next((reaction[key] for key in ("validation", "coref", "extraction")
                 if isinstance(reaction.get(key), dict) and "ner" in reaction[key]), None)


def _missing_extraction(reaction):
    """True when the archived reaction has no usable compound extraction."""
    layer = _detail_with_ner(reaction)
    if layer is None:
        return True
    ner = layer.get("ner") or {}
    if not isinstance(ner, dict):
        return True
    if any(ner.get(field) for field in _COMPOUND_FIELDS):
        return False
    extraction = reaction.get("extraction") if isinstance(reaction.get("extraction"), dict) else {}
    parse_status = extraction.get("parse_status") or {}
    if extraction.get("deferred_llm") or parse_status.get("ner") in {
            "deferred_no_compatible_ner", "empty_ner", "invalid_or_empty_ner"}:
        return True
    quality = reaction.get("quality") or {}
    return any("invalid_or_empty_ner" in str(issue)
               for issue in quality.get("blocking_issues") or [])


def _step_text(reaction):
    context = reaction.get("step_source") or {}
    return (context.get("local_text") or reaction.get("text") or "").strip()


def reuse_previous_snapshot(records, previous_results):
    """Graft model output from an older same-document snapshot.

    The extraction round of 2026-10-16 invalidated some cached responses when
    only the context version changed, leaving steps with an identical source
    text without NER. Reusing the earlier model response is offline, uses no
    new model call, and is recorded per reaction. Only a unique identical step
    text is reused; ambiguity falls through to targeted re-extraction.
    """
    if not previous_results:
        return 0
    by_text = defaultdict(list)
    for old in previous_results:
        for reaction in old.get("reactions") or []:
            text = _step_text(reaction)
            if text:
                by_text[text].append(reaction)
    reused = 0
    for record in records:
        for reaction in record.get("reactions") or []:
            if not _missing_extraction(reaction):
                continue
            candidates = by_text.get(_step_text(reaction)) or []
            if len(candidates) != 1:
                continue
            old = candidates[0]
            layer = _detail_with_ner(old)
            if layer is None:
                continue
            for key in ("validation", "coref", "extraction"):
                reaction.pop(key, None)
            reaction["validation"] = json.loads(json.dumps(layer))
            reaction["reused_previous_snapshot"] = {
                "match": "identical_step_text",
                "previous_status": old.get("status"),
                "previous_context_version": old.get("context_version"),
            }
            quality = reaction.get("quality")
            if isinstance(quality, dict):
                issues = list(quality.get("blocking_issues") or [])
                if "extraction_reused_from_previous_snapshot" not in issues:
                    issues.append("extraction_reused_from_previous_snapshot")
                quality["blocking_issues"] = issues
                quality["admission_eligible"] = False
            reused += 1
    return reused


def _disclosure_reactions(record, step):
    """Use step identity, then a unique identical source body, never position."""
    reactions = record.get("reactions") or []
    matched = [r for r in reactions if r.get("step_id") == step.get("step_id")]
    if matched:
        return matched
    text = (step.get("local_text") or "").strip()
    matched = [r for r in reactions if text and _step_text(r) == text]
    identities = {r.get("step_id") for r in matched}
    return matched if len(identities) == 1 else []


def _disclosure_action(record, step, matched):
    kind = step.get("process_kind") or (record.get("source") or {}).get("process_kind")
    reason = record.get("reason")
    if kind == "formulation" or (record.get("status") == "non_reaction" and reason == "formulation"):
        return "retain_formulation"
    if kind in {"assay", "analysis"}:
        return "retain_nonreaction"
    if record.get("status") == "non_reaction":
        return "review_purification" if reason == "purification" else "retain_nonreaction"
    if any(compound_lists(r).get("products") for r in matched):
        return "extracted"
    if record.get("source_alignment_review"):
        return "review_source_alignment"
    if not matched and any(compound_lists(r).get("products") for r in record.get("reactions") or []):
        return "review_step_alignment"
    layers = [r.get(key) or {} for r in matched for key in ("extraction", "coref", "validation")]
    raw_ner = [layer.get("raw", {}).get("ner") for layer in layers if isinstance(layer.get("raw"), dict)]
    if any(raw_ner):
        return "reuse_response_or_repair_mapping"
    if reason == "mapping_or_structure_requires_review":
        return "repair_structure_mapping"
    if reason == "extraction_missing":
        return "missing_extraction_record"
    if any(layer.get("deferred_llm") for layer in layers):
        return "review_deferred_input"
    return "review_failed_extraction"


def _disclosed_records(records, chunks):
    """Keep source-grounded preparations even when extraction never ran."""
    by_id = {record["id"]: record for record in records}
    disclosures = []
    for chunk in chunks:
        has_evidence = chunk.get("candidate_evidence") or (
            chunk.get("process_kind") in {"synthesis", "purification", "formulation"}
            and chunk.get("has_procedure"))
        if not has_evidence or chunk.get("process_kind") in {"assay", "analysis"}:
            continue
        record = by_id.get(chunk["id"])
        if record is None:
            record = {"id": chunk["id"], "source": deepcopy(chunk),
                      "chunk_fingerprint": chunk.get("chunk_fingerprint"),
                      "status": "needs_review", "reason": "extraction_missing"}
            records.append(record)
            by_id[chunk["id"]] = record
        steps = chunk.get("steps") or [{
            "step_id": chunk["id"] + ":step:1", "title": chunk.get("title"),
            "local_text": chunk.get("procedure") or chunk.get("text") or "",
            "source_parts": chunk.get("source_parts") or [],
            "process_kind": chunk.get("process_kind"), "record_kind": "reaction_candidate"}]
        # An intentional non-reaction has no missing chemical extraction.
        # Remove only the synthetic placeholders made by the previous scanner.
        nonreaction = record.get("status") == "non_reaction" or chunk.get("process_kind") in {
            "formulation", "assay", "analysis"}
        if nonreaction:
            record["reactions"] = [r for r in record.get("reactions") or []
                                   if r.get("reason") != "extraction_missing"]
        if not record.get("reactions") and not nonreaction:
            record["terminal_product_coverage"] = "unknown"
            record["reactions"] = [{"index": i, "step_id": step.get("step_id"),
                                    "step_source": deepcopy(step), "status": "needs_review",
                                    "reason": "extraction_missing"}
                                   for i, step in enumerate(steps, 1)]
        for step in steps:
            matched = _disclosure_reactions(record, step)
            action = _disclosure_action(record, step, matched)
            plan = (record.get("repair_plan") or {}).get(step.get("step_id")) or {}
            changes = plan.get("effective_changes") or []
            attempted = step.get("step_id") in (record.get("model_attempted_step_ids") or [])
            if attempted and action not in {"extracted", "retain_formulation", "retain_nonreaction", "review_purification"}:
                action = "review_after_model_attempt"
            elif changes and action not in {"extracted", "retain_formulation", "retain_nonreaction", "review_purification"}:
                action = "targeted_extraction"
            disclosures.append({"chunk_id": chunk["id"], "step_id": step.get("step_id"),
                                "title": step.get("title") or chunk.get("title"),
                                "source_parts": [{k: p[k] for k in ("block_idx", "page_idx", "span", "source") if k in p}
                                                 for p in step.get("source_parts") or []],
                                "extraction_available": any(bool(compound_lists(r).get("products")) for r in matched),
                                "repair_action": action, "effective_changes": changes,
                                "tasks": plan.get("tasks") or []})
    return disclosures


def scan_document(entry):
    """Read one delivered document's archived records without model calls."""
    delivery = Path(entry["delivery_dir"])
    artifacts = delivery / "artifacts"
    document_dir = Path(entry["output_dir"])
    manifest = read_json(artifacts / "manifest.json")
    identity = (read_json(document_dir / "source_identity.json")
                or read_json(artifacts / "source_identity.json"))
    raw_document = read_json(artifacts / "raw_document_result.json")
    cached_final = read_json(artifacts / "final_product.json")
    final_product = (cached_final.get("product") and cached_final) or raw_document.get("final_product")
    records, chunks, error = [], [], None
    reused = 0
    disclosures = []
    analogy_repair = None
    try:
        records, chunks = current_records(artifacts)
        if any(_missing_extraction(reaction)
               for record in records for reaction in record.get("reactions") or []):
            previous = read_jsonl(document_dir / "results.jsonl")
            reused = reuse_previous_snapshot(records, previous)
        # Analogical-example repair is deterministic and model-free, so the
        # offline scan can apply it to archived results directly.
        from .analogy_instantiate import apply_analogy_repair
        analogy_repair = apply_analogy_repair(records, chunks)
        disclosures = _disclosed_records(records, chunks)
    except Exception as exc:  # a broken snapshot must stay visible, not vanish
        error = f"{type(exc).__name__}: {exc}"
    base = document_identity(entry, manifest, identity)
    return {
        "uuid": entry["uuid"],
        "source": entry.get("source"),
        "document_id": manifest.get("document_id") or entry["document_id"],
        "delivery_dir": str(delivery),
        "pdf_path": (identity or {}).get("pdf_path") or "",
        "pdf_sha256": manifest.get("pdf_sha256"),
        "source_label": manifest.get("source_label") or document_dir.name,
        "drug_folder": Path((identity or {}).get("pdf_path") or "").parent.name,
        "reused_previous_reactions": reused,
        "analogy_repair": analogy_repair,
        "processing_status": entry.get("processing_status") or manifest.get("processing_status"),
        "delivery_revision": entry.get("delivery_revision") or manifest.get("delivery_revision"),
        "reaction_rows": entry.get("reaction_rows"),
        "route_rows": entry.get("route_rows"),
        "chunks": entry.get("chunks"),
        "disclosed_steps": len(disclosures),
        "disclosures": disclosures,
        "record_count": len(records),
        **base,
        "records": records,
        "final_product": final_product if isinstance(final_product, dict) and final_product.get("product") else None,
        "scan_error": error,
    }


def run_scan(deliverable_root, out_root, workers=8, limit=None, force=False, documents_filter=None):
    deliverable_root = Path(deliverable_root).resolve()
    out_root = Path(out_root).resolve()
    scan_dir = out_root / "scan"
    cache_dir = scan_dir / "records"
    cache_dir.mkdir(parents=True, exist_ok=True)
    documents = read_jsonl(deliverable_root / "documents.jsonl")
    if documents_filter:
        documents = [row for row in documents
                     if re.search(documents_filter, json.dumps(row, ensure_ascii=False))]
    if limit:
        documents = documents[:int(limit)]
    index_rows = []
    pending = []
    for document in documents:
        cache = cache_dir / (document["uuid"] + ".json.gz")
        index_path = scan_dir / "index" / (document["uuid"] + ".json")
        if cache.is_file() and index_path.is_file() and not force:
            index_rows.append(read_json(index_path))
        else:
            pending.append(document)

    def work(document):
        row = scan_document(document)
        records = row.pop("records")
        write_json(scan_dir / "index" / (document["uuid"] + ".json"), row)
        write_gzip_jsonl(cache_dir / (document["uuid"] + ".json.gz"), records)
        return row

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            for number, row in enumerate(pool.map(work, pending), 1):
                index_rows.append(row)
                if number % 100 == 0 or number == len(pending):
                    logger.info("scan {}/{} {}", number, len(pending), row.get("document_id"))
    index_rows.sort(key=lambda row: (row.get("source") or "", row.get("document_id") or ""))
    write_jsonl(scan_dir / "documents.jsonl", index_rows)
    write_supplement_charter(out_root, index_rows)
    errors = [row for row in index_rows if row.get("scan_error")]
    write_jsonl(scan_dir / "scan_errors.jsonl", errors,
                sort_key=lambda row: (row.get("source") or "", row.get("document_id") or ""))
    update_state(out_root, scan={"completed_at": _now(), "documents": len(index_rows),
                                 "with_records": sum(bool(row.get("record_count", row.get("reaction_rows"))) for row in index_rows),
                                 "errors": len(errors)})
    return {"documents": len(index_rows), "errors": len(errors)}


def write_supplement_charter(out_root, index_rows):
    """Export evidence-backed extraction gaps; scanning itself stays offline."""
    supplement_items = []
    diagnostics, counts = [], Counter()
    for row in index_rows:
        disclosures = row.get("disclosures") or []
        counts.update(d.get("repair_action", "unclassified") for d in disclosures)
        diagnostics.extend(dict(d, uuid=row["uuid"], document_id=row["document_id"])
                           for d in disclosures if not d["extraction_available"])
        missing = [d for d in disclosures if d.get("repair_action") == "targeted_extraction"
                   and d.get("effective_changes") and not d["extraction_available"]]
        if missing:
            supplement_items.append({"uuid": row["uuid"], "document_id": row["document_id"],
                                     "chunks": list(dict.fromkeys(d["chunk_id"] for d in missing)),
                                     "step_ids": [d["step_id"] for d in missing],
                                     "preset": "huawei-maas",
                                     "reason": "source_input_changed_after_offline_repair",
                                     "evidence": missing})
    write_json(Path(out_root) / "scan" / "supplement_charter.json", {"items": supplement_items})
    write_jsonl(Path(out_root) / "scan" / "repair_actions.jsonl", diagnostics)
    write_json(Path(out_root) / "scan" / "repair_classification.json", dict(counts))


# ---------------------------------------------------------------------------
# Patent scope grouping
# ---------------------------------------------------------------------------

def group_scopes(index_rows):
    """Group documents into confirmed patent scopes; identical PDFs collapse."""
    by_patent = defaultdict(list)
    for row in index_rows:
        key = row.get("patent_key")
        scope = f"{row.get('source')}::{key}" if key else f"{row.get('source')}::DOC::{row['document_id']}"
        by_patent[scope].append(row)
    scopes = []
    for scope_id, rows in sorted(by_patent.items()):
        canonical, aliases = {}, {}
        for row in sorted(rows, key=lambda item: (item.get("pdf_sha256") or "",
                                                  item.get("source_label") or "",
                                                  item.get("document_id") or "")):
            sha = row.get("pdf_sha256") or ("uuid:" + row["uuid"])
            if sha in canonical:
                aliases[row["uuid"]] = canonical[sha]["uuid"]
                canonical[sha]["duplicate_uuids"].append(row["uuid"])
                continue
            row["duplicate_uuids"] = []
            canonical[sha] = row
        documents = sorted(canonical.values(), key=lambda item: (
            item.get("volume_index") if item.get("volume_index") is not None else 10 ** 6,
            item.get("filename") or "", item.get("document_id") or ""))
        for order, document in enumerate(documents):
            document["volume_order"] = order
        scopes.append({"scope_id": scope_id, "patent_key": rows[0].get("patent_key"),
                       "source": rows[0].get("source"), "documents": documents,
                       "aliases": aliases,
                       "volume_count": len(documents)})
    return scopes


def load_scope_input(scope, cache_dir, alias_rows=None):
    documents, finals = [], {}
    rows = list(scope["documents"])
    for document in rows:
        records = read_gzip_jsonl(Path(cache_dir) / (document["uuid"] + ".json.gz"))
        if records:
            documents.append({"document_id": document["document_id"],
                              "volume_order": document.get("volume_order", 0),
                              "records": records, "uuid": document["uuid"]})
            if document.get("final_product"):
                finals[document["document_id"]] = document["final_product"]
    return documents, finals


def scope_file_name(scope_id):
    return hashlib.sha1(scope_id.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Assembly conversion (route-v2 rows)
# ---------------------------------------------------------------------------

def _ref_reaction(document_id, ref):
    return business_reaction_id(document_id, {"id": ref["id"], "index": ref["index"]})


def _convert_refs(values):
    out = []
    for value in values or []:
        if isinstance(value, dict) and "id" in value and "index" in value:
            out.append(_ref_reaction(value.get("document_id") or "", value))
        elif isinstance(value, str):
            out.append(value)
    return out


def _unit_by_ref(selected, ref):
    return next((unit for unit in selected if unit.ref == ref), None)


def _pages_of(unit):
    parts = (unit.context or {}).get("source_parts") or []
    return _pages(parts)


def _convert_target(terminal, terminal_slots, target, document_id):
    target_reaction = _ref_reaction(document_id, terminal.ref)
    product_index = target.get("product_index")
    product_number = (product_index + 1) if product_index is not None else "title"
    target_compound = terminal_slots.get(("products", product_index)) if product_index is not None else None
    payload = {"reaction_id": target_reaction, "compound_id": target_compound}
    display = target.get("display") or {}
    if target_compound is None:
        payload.update({"name": display.get("name"),
                        "canonical_smiles": display.get("canonical_smiles"),
                        "basis": "source_title"})
    else:
        if display.get("name"):
            payload["name"] = display["name"]
        if display.get("canonical_smiles"):
            payload["canonical_smiles"] = display["canonical_smiles"]
    return payload, product_number


def convert_candidate(candidate, kind):
    terminal = candidate["terminal"]
    selected = candidate["selected"]
    document_id = terminal.document_id
    slots = {}
    for unit in selected:
        key = (unit.document_id, unit.record["id"], unit.reaction["index"])
        if key not in slots:
            _values, slot_map, _normalized = business_compounds(unit.reaction)
            slots[key] = slot_map
    terminal_slots = slots[(terminal.document_id, terminal.record["id"], terminal.reaction["index"])]
    target_payload, product_number = _convert_target(
        terminal, terminal_slots, candidate["target"], document_id)
    target_payload["source_reaction_ids"] = (terminal.reaction.get("provenance") or {}).get("source_reaction_ids") or []
    target_reaction = target_payload["reaction_id"]
    steps = []
    for order, unit in enumerate(selected, 1):
        steps.append({"document_id": unit.document_id,
                      "reaction_id": _ref_reaction(unit.document_id, unit.ref),
                      "chunk_id": unit.ref["id"],
                      "step_id": unit.ref.get("step_id"),
                      "source_order": order,
                      "source_reaction_ids": (unit.reaction.get("provenance") or {}).get("source_reaction_ids") or [],
                      "pages": _pages_of(unit)})
    gaps = []
    for gap in candidate["gaps"]:
        row = {"code": gap["code"]}
        converted = _convert_refs(gap.get("reaction_ids"))
        if converted:
            row["reaction_ids"] = converted
        candidates = _convert_refs(gap.get("candidate_reaction_ids"))
        if candidates:
            row["candidate_reaction_ids"] = candidates
        for key in ("material", "reference_id", "evidence_text"):
            if gap.get(key) is not None:
                row[key] = gap[key]
        gaps.append(row)
    connections = []
    for internal in candidate["connections"]:
        from_unit = _unit_by_ref(selected, internal["from_ref"])
        to_unit = _unit_by_ref(selected, internal["to_ref"])
        if from_unit is None or to_unit is None:
            continue
        from_key = _ref_reaction(from_unit.document_id, from_unit.ref)
        to_key = _ref_reaction(to_unit.document_id, to_unit.ref)
        product_id = slots[(from_unit.document_id, from_unit.record["id"], from_unit.reaction["index"])].get(
            ("products", internal["product_index"]))
        input_id = slots[(to_unit.document_id, to_unit.record["id"], to_unit.reaction["index"])].get(
            (internal["input_role"], internal["input_index"]))
        if product_id is None or input_id is None:
            gaps.append({"code": "connection_material_unavailable",
                         "reaction_ids": [from_key, to_key]})
            continue
        connection = {"from": from_key, "to": to_key,
                      "product_id": product_id, "input_id": input_id,
                      "basis": internal["basis"]}
        if internal.get("evidence_text"):
            connection["evidence_text"] = internal["evidence_text"]
        connections.append(connection)
    boundaries = []
    for boundary in candidate["boundaries"]:
        row = {"reaction_id": _ref_reaction(boundary["reaction_id"].get("document_id") or "",
                                            boundary["reaction_id"]),
               "name": boundary.get("name"), "reason": boundary.get("reason")}
        if boundary.get("evidence_text"):
            row["evidence_text"] = boundary["evidence_text"]
        boundaries.append(row)
    alternatives = []
    for alternative in candidate["alternatives"]:
        row = {"reaction_id": _ref_reaction(alternative["reaction_id"].get("document_id") or "",
                                            alternative["reaction_id"]),
               "material": alternative.get("material")}
        chosen = alternative.get("chosen")
        if chosen:
            row["chosen_reaction_id"] = _ref_reaction(chosen.get("document_id") or "", chosen)
        row["alternative_reaction_ids"] = [
            _ref_reaction(item.get("document_id") or "", item)
            for item in alternative.get("alternatives") or []]
        alternatives.append(row)
    blocking = []
    ineligible = [_ref_reaction(unit.document_id, unit.ref) for unit in selected
                  if not eligible(unit.record, unit.reaction)]
    if ineligible:
        blocking.append({"code": "step_quality_blocked", "reaction_ids": ineligible})
    if target_payload["compound_id"] is None:
        blocking.append({"code": "target_from_title_without_extracted_product",
                         "reaction_ids": [target_reaction]})
    quality = {"admission_eligible": not blocking,
               "blocking_issues": blocking,
               "policy_version": QUALITY_VERSION}
    record_ids = list(dict.fromkeys(unit.ref["id"] for unit in selected))
    row = {
        "schema_version": SCHEMA_ROUTE if kind == "route" else SCHEMA_FRAGMENT,
        "patent_id": candidate.get("patent_key"),
        "scope_id": candidate.get("scope_id"),
        "document_id": document_id,
        "volumes": list(dict.fromkeys(step["document_id"] for step in steps)),
        "title": (terminal.source.get("title") or terminal.context.get("title") or terminal.ref["id"]),
        "process_kind": terminal.kind if terminal.kind in {"synthesis", "formulation", "purification", "unknown"} else "unknown",
        "target": target_payload,
        "target_identity": candidate["target"].get("identity"),
        "target_group_id": target_reaction + "::target::" + str(product_number),
        "target_group_basis": candidate["target"].get("basis"),
        "steps": steps,
        "reaction_ids": [step["reaction_id"] for step in steps],
        "connections": connections,
        "start_boundaries": boundaries,
        "alternatives": alternatives,
        "coverage": {"documents": list(dict.fromkeys(step["document_id"] for step in steps)),
                     "records": record_ids, "steps": len(steps),
                     "cross_volume": len({step["document_id"] for step in steps}) > 1},
        "completeness": candidate["completeness"],
        "gaps": gaps,
        "quality": quality,
        "provenance": {"scope_version": ROUTE_SCOPE_VERSION,
                       "assembler_version": ROUTE_V2_VERSION,
                       "old_route_ids": [], "repair_actions": []},
    }
    row["schema_version"] = SCHEMA_ROUTE if kind == "route" else SCHEMA_FRAGMENT
    if kind == "route":
        row["route_id"] = target_reaction + "::route::" + str(product_number)
    else:
        row["fragment_id"] = target_reaction + "::fragment::" + str(product_number)
    return row


def _load_old_routes(uuid, delivery_by_uuid):
    return read_jsonl(Path(delivery_by_uuid[uuid]) / "routes.jsonl")


def convert_template_route(scope, candidate):
    """Convert a reference-route candidate (scheme-template steps chained by
    letter-code continuity) into a route-v2 row with kind="reference_route"."""
    terminal = candidate["terminal"]
    document_id = terminal.document_id
    steps = [{"document_id": unit.document_id,
              "reaction_id": _ref_reaction(unit.document_id, unit.ref),
              "chunk_id": unit.ref["id"], "step_id": unit.ref.get("step_id"),
              "source_order": order, "pages": _pages_of(unit)}
             for order, unit in enumerate(candidate["selected"], 1)]
    connections = [{"from": _ref_reaction(document_id, edge["from"]),
                    "to": _ref_reaction(document_id, edge["to"]),
                    "basis": edge["basis"], "code": edge["code"]}
                   for edge in candidate["connections"]]
    target = candidate["target"]
    return {"schema_version": SCHEMA_ROUTE, "kind": "reference_route",
            "route_id": (f"{document_id}::{terminal.ref['id']}::template::"
                         f"{terminal.ref['index']}:{target['product_index']}"),
            "patent_key": scope.get("patent_key"), "scope_id": scope["scope_id"],
            "document_id": document_id,
            "title": (terminal.record.get("source") or {}).get("title") or terminal.ref["id"],
            "process_kind": "synthesis",
            "target": {"identity": target["identity"], "display": target["display"],
                       "basis": target["basis"]},
            "steps": steps, "connections": connections,
            "alternatives": [{**choice,
                "consumer": _ref_reaction(document_id, choice['consumer']),
                "producers": [_ref_reaction(document_id, ref) for ref in choice['producers']]}
                for choice in candidate.get('alternatives') or []],
            "gaps": candidate["gaps"], "completeness": candidate["completeness"],
            "volumes": candidate["volumes"],
            "provenance": {"assembler_version": TEMPLATE_ROUTE_VERSION}}


def convert_scope(scope, routes, fragments, delivery_by_uuid):
    route_rows, fragment_rows = [], []
    for candidate in routes:
        candidate.setdefault("patent_key", scope.get("patent_key"))
        candidate.setdefault("scope_id", scope["scope_id"])
        route_rows.append(convert_candidate(candidate, "route"))
    for candidate in fragments:
        candidate.setdefault("patent_key", scope.get("patent_key"))
        candidate.setdefault("scope_id", scope["scope_id"])
        fragment_rows.append(convert_candidate(candidate, "fragment"))
    related = defaultdict(list)
    related_upstream = defaultdict(set)
    for row in route_rows:
        related[row["target_identity"]].append(row["route_id"])
    for candidate, route in zip(routes, route_rows):
        for unit in candidate["selected"]:
            for choice in unit.preparation_choices:
                for producer, _, _ in choice["pool"]:
                    if producer is not choice["edge"]["producer"]:
                        related_upstream[_ref_reaction(producer.document_id, producer.ref)].add(route["route_id"])
    for row in fragment_rows:
        row["related_route_ids"] = list(related.get(row["target_identity"]) or [])
        # An unfinished upstream method remains a fragment associated with
        # routes that use another preparation of the same material.
        terminal = row["target"]["reaction_id"]
        row["related_route_ids"].extend(related_upstream.get(terminal, ()))
        row["related_route_ids"] = sorted(set(row["related_route_ids"]))
    mapping_rows = _old_mapping(scope, route_rows, fragment_rows, delivery_by_uuid)
    return route_rows, fragment_rows, mapping_rows


def _old_mapping(scope, route_rows, fragment_rows, delivery_by_uuid):
    new_by_target = defaultdict(list)
    new_by_origin = defaultdict(list)
    new_by_step = defaultdict(list)
    for row in route_rows + fragment_rows:
        row_id = row.get("route_id") or row.get("fragment_id")
        new_by_target[row["target"]["reaction_id"]].append(row_id)
        for old_ref in row["target"].get("source_reaction_ids") or []:
            new_by_origin[old_ref].append(row_id)
        for step in row["steps"]:
            for ref in [step["reaction_id"], *(step.get("source_reaction_ids") or [])]:
                new_by_step[ref].append(row_id)
    route_ids = {row["route_id"]: row for row in route_rows}
    fragment_ids = {row["fragment_id"]: row for row in fragment_rows}
    mappings = []
    document_ids = {document["uuid"]: document["document_id"] for document in scope["documents"]}
    alias_targets = {alias: canonical for alias, canonical in scope["aliases"].items()}
    for uuid in list(document_ids) + list(alias_targets):
        delivery = delivery_by_uuid.get(uuid)
        if not delivery:
            continue
        old_rows = read_jsonl(Path(delivery) / "routes.jsonl")
        old_reactions = {r["reaction_id"]: r for r in read_jsonl(Path(delivery) / "reactions.jsonl")}
        canonical_id = document_ids.get(uuid) or document_ids.get(alias_targets.get(uuid), "")
        for old in old_rows:
            old_id = old.get("route_id")
            if not old_id:
                continue
            target_reaction = (old.get("target") or {}).get("reaction_id")
            matches = new_by_target.get(target_reaction) or []
            action = "removed"
            reason = "not_present_in_repaired_candidates"
            if not matches:
                matches = list(dict.fromkeys(new_by_origin.get(target_reaction) or []))
                old_compound = next((c for c in (old_reactions.get(target_reaction) or {}).get("compounds") or []
                                     if c.get("compound_id") == (old.get("target") or {}).get("compound_id")), None)
                if old_compound:
                    from .routes import _scope_match
                    matches = [new_id for new_id in matches
                               if _scope_match(old_compound, (route_ids.get(new_id) or fragment_ids[new_id])["target"])]
                if matches:
                    action, reason = "remapped", "target_resegmented_or_reextracted"
                else:
                    matches = list(dict.fromkeys(new_by_step.get(target_reaction) or []))
                    if matches:
                        action, reason = "merged", "included_in_upstream_closure"
            if matches:
                if action == "removed":
                    action = "kept" if len(matches) == 1 and matches[0] in route_ids else (
                        "fragmented" if len(matches) == 1 else "remapped")
                    reason = "target_preserved" if action == "kept" else "target_reclassified"
                for new_id in matches:
                    row = route_ids.get(new_id) or fragment_ids.get(new_id)
                    if row is not None:
                        if old_id not in row["provenance"]["old_route_ids"]:
                            row["provenance"]["old_route_ids"].append(old_id)
                        if new_id in route_ids:
                            row["provenance"]["repair_actions"].append(
                                action if action in {"merged", "remapped"} else
                                "extended_upstream" if len(row["steps"]) > len(old.get("steps") or []) else "kept")
                        else:
                            row["provenance"]["repair_actions"].append("fragmented")
            mappings.append({"document_id": canonical_id, "uuid": uuid,
                             "old_route_id": old_id,
                             "old_target_reaction_id": target_reaction,
                             "new_ids": matches, "action": action, "reason": reason})
    return mappings


# ---------------------------------------------------------------------------
# Assemble stage
# ---------------------------------------------------------------------------

def run_assemble(out_root, limit=None, force=False):
    out_root = Path(out_root).resolve()
    scan_dir = out_root / "scan"
    cache_dir = scan_dir / "records"
    index_rows = read_jsonl(scan_dir / "documents.jsonl")
    if not index_rows:
        raise SystemExit("scan stage has not run")
    scopes = group_scopes(index_rows)
    if limit:
        scopes = scopes[:int(limit)]
    delivery_by_uuid = {row["uuid"]: row.get("delivery_dir") for row in index_rows}
    assemble_dir = out_root / "assemble"
    scopes_dir = assemble_dir / "scopes"
    scopes_dir.mkdir(parents=True, exist_ok=True)
    progress_path = assemble_dir / "progress.jsonl"
    done = set()
    if progress_path.is_file() and not force:
        done = {row["scope_id"] for row in read_jsonl(progress_path)}
    pending = [scope for scope in scopes if scope["scope_id"] not in done]
    for number, scope in enumerate(pending, 1):
        documents, finals = load_scope_input(scope, cache_dir)
        if not documents:
            payload = {"scope_id": scope["scope_id"], "documents": 0,
                       "routes": [], "fragments": [], "mapping": [],
                       "status": "no_records"}
        else:
            routes, fragments = assemble_routes_v2(documents, finals)
            route_rows, fragment_rows, mapping = convert_scope(
                scope, routes, fragments, delivery_by_uuid)
            template_rows = [convert_template_route(scope, candidate)
                             for candidate in assemble_template_routes(documents)]
            payload = {"scope_id": scope["scope_id"], "documents": len(documents),
                       "volumes": len({doc["document_id"] for doc in documents}),
                       "routes": route_rows, "fragments": fragment_rows,
                       "template_routes": template_rows,
                       "mapping": mapping, "status": "assembled"}
        write_gzip(scopes_dir / (scope_file_name(scope["scope_id"]) + ".json.gz"), payload)
        summary = {"scope_id": scope["scope_id"], "documents": payload["documents"],
                   "routes": len(payload["routes"]), "fragments": len(payload["fragments"]),
                   "template_routes": len(payload.get("template_routes") or []),
                   "completed_at": _now()}
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded(summary) + "\n")
        if number % 50 == 0 or number == len(pending):
            logger.info("assemble {}/{} {}", number, len(pending), scope["scope_id"])
    update_state(out_root, assemble={"completed_at": _now(), "scopes": len(scopes),
                                     "assembled_now": len(pending)})
    return {"scopes": len(scopes), "assembled_now": len(pending)}


def collect_assembled(out_root):
    assemble_dir = Path(out_root) / "assemble"
    routes, fragments, mapping, template_routes = [], [], [], []
    for path in sorted((assemble_dir / "scopes").glob("*.json.gz")):
        payload = read_gzip(path) or {}
        routes.extend(payload.get("routes") or [])
        fragments.extend(payload.get("fragments") or [])
        mapping.extend(payload.get("mapping") or [])
        template_routes.extend(payload.get("template_routes") or [])
    write_jsonl(assemble_dir / "routes.jsonl", routes, sort_key=lambda row: row["route_id"])
    write_jsonl(assemble_dir / "route_fragments.jsonl", fragments, sort_key=lambda row: row["fragment_id"])
    write_jsonl(assemble_dir / "template_routes.jsonl", template_routes,
                sort_key=lambda row: row["route_id"])
    write_jsonl(assemble_dir / "old_to_new.jsonl", mapping,
                sort_key=lambda row: (row["uuid"], row["old_route_id"]))
    return routes, fragments, mapping, template_routes


# ---------------------------------------------------------------------------
# Export stage
# ---------------------------------------------------------------------------

def _referenced_reaction_ids(rows):
    by_doc = defaultdict(set)

    def add(reaction_id):
        if isinstance(reaction_id, str) and "::" in reaction_id:
            by_doc[reaction_id.split("::", 1)[0]].add(reaction_id)

    for row in rows:
        for step in row.get("steps") or []:
            add(step.get("reaction_id"))
        target = row.get("target") or {}
        add(target.get("reaction_id"))
        for connection in row.get("connections") or []:
            add(connection.get("from"))
            add(connection.get("to"))
        for boundary in row.get("start_boundaries") or []:
            add(boundary.get("reaction_id"))
        for gap in row.get("gaps") or []:
            for reaction_id in gap.get("reaction_ids") or []:
                add(reaction_id)
            for reaction_id in gap.get("candidate_reaction_ids") or []:
                add(reaction_id)
        for alternative in row.get("alternatives") or []:
            add(alternative.get("reaction_id"))
            add(alternative.get("chosen_reaction_id"))
            for reaction_id in alternative.get("alternative_reaction_ids") or []:
                add(reaction_id)
    return by_doc


def _generate_reaction_file(document_id, records, required_ids):
    """Regenerate business reactions from the current records for one document."""
    rows, filtered = [], []
    for record in records:
        for reaction in record.get("reactions") or []:
            row, _slots = business_reaction(document_id, record, reaction)
            reaction_identity = business_reaction_id(
                document_id, {"id": record["id"], "index": reaction["index"]})
            raw_kind = (record.get("source") or {}).get("process_kind")
            meaningful = bool(row["compounds"] or row["actions"]
                              or row["conditions"]["temperature"] or row["conditions"]["time"])
            if reaction_identity not in required_ids and (
                    raw_kind in {"assay", "analysis"} or not meaningful):
                filtered.append({"chunk_id": record["id"],
                                 "reaction_index": reaction["index"],
                                 "reason": "analysis_only" if raw_kind in {"assay", "analysis"}
                                 else "failed_placeholder_without_business_content"})
                continue
            if reaction.get("reused_previous_snapshot"):
                row["provenance"] = {"reused_previous_snapshot":
                                     reaction["reused_previous_snapshot"]}
            if reaction.get("provenance"):
                row.setdefault("provenance", {}).update(deepcopy(reaction["provenance"]))
            rows.append(row)
    return rows, filtered


def _write_indexes(out_root, routes, fragments, index_rows):
    """Document index, patent/volume index and per-document route counts."""
    route_counts = Counter((row.get("scope_id"), row["document_id"]) for row in routes)
    fragment_counts = Counter((row.get("scope_id"), row["document_id"]) for row in fragments)
    route_docs = defaultdict(int)
    fragment_docs = defaultdict(int)
    for row in routes:
        route_docs[row["document_id"]] += 1
    for row in fragments:
        fragment_docs[row["document_id"]] += 1
    documents = []
    for row in index_rows:
        documents.append({
            "document_id": row.get("document_id"), "uuid": row["uuid"],
            "source": row.get("source"), "patent_id": row.get("patent_key"),
            "patent_id_source": row.get("patent_key_source"),
            "volume_index": row.get("volume_index"), "filename": row.get("filename"),
            "pdf_path": row.get("pdf_path"), "pdf_sha256": row.get("pdf_sha256"),
            "source_label": row.get("source_label"), "drug_folder": row.get("drug_folder"),
            "processing_status": row.get("processing_status"),
            "delivery_revision": row.get("delivery_revision"),
            "chunks": row.get("chunks"), "source_reactions": row.get("reaction_rows"),
            "old_routes": row.get("route_rows"),
            "disclosed_steps": row.get("disclosed_steps"),
            "archived_records": row.get("record_count"),
            "reused_previous_reactions": row.get("reused_previous_reactions"),
            "supplemented_chunks": row.get("supplemented_chunks") or [],
            "scan_error": row.get("scan_error"),
            "complete_routes": route_docs.get(row.get("document_id"), 0),
            "fragments": fragment_docs.get(row.get("document_id"), 0),
        })
    write_jsonl(out_root / "index" / "documents.jsonl", documents,
                sort_key=lambda row: (row["source"] or "", row["document_id"] or ""))
    scopes = group_scopes(index_rows)
    patents = []
    for scope in scopes:
        scope_id = scope["scope_id"]
        patents.append({
            "scope_id": scope_id,
            "patent_id": scope.get("patent_key"),
            "source": scope.get("source"),
            "documents": [document["document_id"] for document in scope["documents"]],
            "uuids": [document["uuid"] for document in scope["documents"]],
            "volumes": [document.get("volume_index") for document in scope["documents"]],
            "volume_count": scope["volume_count"],
            "duplicate_aliases": scope.get("aliases") or {},
            "complete_routes": sum(count for (sid, _doc), count in route_counts.items() if sid == scope_id),
            "fragments": sum(count for (sid, _doc), count in fragment_counts.items() if sid == scope_id),
        })
    write_jsonl(out_root / "index" / "patents.jsonl", patents,
                sort_key=lambda row: row["scope_id"])


def run_export(out_root, reactions_mode="copy", drug_sources=True):
    out_root = Path(out_root).resolve()
    routes, fragments, mapping, template_routes = collect_assembled(out_root)
    index_rows = read_jsonl(out_root / "scan" / "documents.jsonl")
    write_jsonl(out_root / "routes.jsonl", routes, sort_key=lambda row: row["route_id"])
    write_jsonl(out_root / "route_fragments.jsonl", fragments, sort_key=lambda row: row["fragment_id"])
    write_jsonl(out_root / "template_routes.jsonl", template_routes,
                sort_key=lambda row: row["route_id"])
    write_jsonl(out_root / "mapping" / "old_to_new.jsonl", mapping,
                sort_key=lambda row: (row["uuid"], row["old_route_id"]))
    _write_indexes(out_root, routes, fragments, index_rows)
    referenced = _referenced_reaction_ids(routes + fragments)
    cache_dir = out_root / "scan" / "records"
    reaction_dir = out_root / "reactions"
    copied = regenerated = 0
    reference_gaps = []
    for row in index_rows:
        source = row.get("source") or "unknown"
        target = reaction_dir / source / (row["uuid"] + ".jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        old = Path(row["delivery_dir"]) / "reactions.jsonl"
        required = referenced.get(row.get("document_id")) or set()
        changed = bool(row.get("reused_previous_reactions")) or bool(row.get("supplemented_chunks"))
        missing = set()
        if required:
            old_ids = {item.get("reaction_id") for item in read_jsonl(old)}
            missing = required - old_ids
        if changed or missing:
            records = read_gzip_jsonl(cache_dir / (row["uuid"] + ".json.gz"))
            rows, _filtered = _generate_reaction_file(row.get("document_id"), records, required)
            write_jsonl(target, rows, sort_key=lambda item: item["reaction_id"])
            written = {item.get("reaction_id") for item in rows}
            for reaction_id in required - written:
                reference_gaps.append({"document_id": row.get("document_id"),
                                       "reaction_id": reaction_id})
            regenerated += 1
        elif old.is_file():
            shutil.copyfile(old, target)
            copied += 1
        else:
            write_jsonl(target, [])
    summary = build_summary(out_root, routes, fragments, index_rows, mapping)
    summary["reaction_files"] = copied + regenerated
    summary["reaction_files_copied"] = copied
    summary["reaction_files_regenerated"] = regenerated
    summary["reaction_reference_gaps"] = len(reference_gaps)
    write_jsonl(out_root / "stats" / "reaction_reference_gaps.jsonl", reference_gaps)
    if drug_sources:
        paths = drug_sources if isinstance(drug_sources, dict) else None
        summary["drug"] = run_drug_tables(out_root, paths)
    write_json(out_root / "stats" / "summary.json", summary)
    update_state(out_root, export={"completed_at": _now(), "routes": len(routes),
                                   "fragments": len(fragments), "mapping": len(mapping)})
    return summary


def build_summary(out_root, routes, fragments, index_rows, mapping):
    gap_counts = Counter()
    incomplete = sum(1 for row in fragments if row.get("completeness") == "incomplete")
    unconfirmed = sum(1 for row in fragments if row.get("completeness") == "unconfirmed")
    for row in fragments:
        for gap in row.get("gaps") or []:
            gap_counts[gap["code"]] += 1
    single_step = sum(1 for row in routes if len(row["steps"]) == 1)
    cross_volume = sum(1 for row in routes if (row.get("coverage") or {}).get("cross_volume"))
    extended = sum(1 for row in routes if "extended_upstream" in (row.get("provenance") or {}).get("repair_actions", []))
    actions = Counter(item.get("action") for item in mapping)
    by_source = Counter((row.get("scope_id") or "").split("::", 1)[0] for row in routes + fragments)
    return {
        "stage_version": STAGE_VERSION,
        "schema": {"routes": SCHEMA_ROUTE, "fragments": SCHEMA_FRAGMENT, "reactions": SCHEMA_REACTION},
        "documents": len(index_rows),
        "documents_with_records": sum(bool(row.get("record_count", row.get("reaction_rows"))) for row in index_rows),
        "documents_with_disclosed_steps": sum(bool(row.get("disclosed_steps")) for row in index_rows),
        "disclosed_steps": sum(row.get("disclosed_steps") or 0 for row in index_rows),
        "patents": len({row.get("scope_id") for row in routes + fragments}),
        "complete_routes": len(routes),
        "complete_single_step_routes": single_step,
        "cross_volume_routes": cross_volume,
        "extended_upstream_routes": extended,
        "fragments": len(fragments),
        "fragments_incomplete": incomplete,
        "fragments_unconfirmed": unconfirmed,
        "fragment_gaps": dict(gap_counts),
        "old_route_mapping": {
            "rows": len(mapping),
            "actions": dict(actions),
        },
        "scopes_by_source": dict(by_source),
        "generated_at": _now(),
    }


# ---------------------------------------------------------------------------
# Drug attribution: source relations and route-target chemical relations
# ---------------------------------------------------------------------------

DEFAULT_DRUG_SOURCES = {
    "fda": "/data/patents/FDA_RLD/fda_server_bundle/output/drug_patent_metadata.json",
    "cn_metadata": "/data/patents/CN_RLD/cn_server_bundle/cn_metadata_package/cn_normalized/drug_patent_metadata.jsonl",
    "cn_registrations": "/data/patents/CN_RLD/cn_server_bundle/cn_metadata_package/cn_normalized/registrations.jsonl",
    "chembl": "/data/patents/ChemBL_DrugList_Patent/result7_drug_manifest.csv",
}

_SALT_NAME = re.compile(
    r"盐酸|氢溴酸|硫酸|磷酸|硝酸|甲磺酸|马来酸|富马酸|酒石酸|柠檬酸|琥珀酸|乙酸|苯磺酸|"
    r"钠$|钾$|钙$|镁$|铵$|hydrochloride|hydrobromide|sulfate|phosphate|mesylate|maleate|"
    r"fumarate|tartrate|citrate|succinate|acetate|besylate|sodium|potassium|calcium", re.I)
_HYDRATE_NAME = re.compile(r"水合物|hydrate|一水|二水|三水", re.I)
_CRYSTAL_NAME = re.compile(r"晶型|结晶|晶体|无定形|crystal|polymorph|amorphous|\bform\s+[A-Z0-9]+\b", re.I)
_DOSAGE_NAME = re.compile(
    r"片|胶囊|注射液|注射剂|颗粒|口服液|软膏|乳膏|滴眼液|贴剂|喷雾剂|组合物|制剂|"
    r"tablet|capsule|injection|ointment|cream|suspension|emulsion|formulation|dosage", re.I)
# Products of degradation/hydrolysis/metabolism are not salt or crystal forms of
# the parent drug, so they must never be reported as 相关形式.  Bare 氧化/oxid is
# deliberately absent: 氧化镁 and similar drug names are not degradation markers,
# so only explicit degradation/impurity phrases are matched.
_DEGRADATION_NAME = re.compile(
    r"降解|水解|代谢|杂质|副产物|氧化产物|氧化降解|氧化杂质|有关物质|"
    r"degrad|hydroly|metaboli|impurit|by-?product|related\s+substances?|"
    r"(?:oxidation|oxidi[sz]ed)\s+(?:product|derivative|impurity)|process\s+impurit",
    re.I)


def _form_kind(name):
    name = str(name or "")
    if _HYDRATE_NAME.search(name):
        return "hydrate"
    if _CRYSTAL_NAME.search(name):
        return "crystal"
    if _SALT_NAME.search(name):
        return "salt"
    return "none"


def _inchikey(smiles):
    if not smiles:
        return None
    try:
        from rdkit import Chem
        molecule = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(molecule) if molecule else None
    except Exception:
        return None


@lru_cache(maxsize=8192)
def _parent_inchikey(smiles):
    """Compare salt/solvate parents without discarding stereochemistry."""
    if not smiles:
        return None
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    parent = rdMolStandardize.FragmentParent(mol)
    return Chem.MolToInchiKey(rdMolStandardize.Uncharger().uncharge(parent))


def _name_key(value):
    return re.sub(r"[\s()（）\[\]【】.,;:、，。；：·'\"‘’“”\-]+", "", str(value or "")).casefold()


_ASCII_ALNUM = re.compile(r"[0-9A-Za-z]")
_WHITESPACE = re.compile(r"\s+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _containment_min_length(key):
    """Minimum name-key length for containment: 3 for CJK keys, else 4.

    Three-character CJK drug bases are common (布洛芬/泼尼松), so their salt
    forms must keep matching by containment; short pure-ASCII keys are formula
    labels ("I"/"A"/"C") and must not match a name that merely carries them.
    """
    return 3 if _CJK.search(key) else 4


def _contains_whole_name(shorter, longer):
    """True when *shorter* occurs in *longer* on ASCII token boundaries.

    Containment needs a long enough normalised key (three characters for CJK
    keys, four otherwise) and the occurrence is rejected when an adjacent
    character is an ASCII alphanumeric: a formula label such as ``I``/``A``/``C``
    must not match a drug name that merely carries the letter, while CJK
    neighbours stay allowed so salt prefixes (``盐酸``) keep matching their base
    name.
    """
    short_key, long_key = _name_key(shorter), _name_key(longer)
    if short_key == long_key or short_key not in long_key:
        return False
    if len(short_key) < _containment_min_length(short_key):
        return False
    short_raw = _WHITESPACE.sub(" ", str(shorter or "")).strip()
    long_raw = _WHITESPACE.sub(" ", str(longer or "")).strip()
    pattern = re.compile(r"\s+".join(re.escape(part) for part in short_raw.split(" ")), re.I)
    for match in pattern.finditer(long_raw):
        before = long_raw[match.start() - 1] if match.start() else ""
        after = long_raw[match.end()] if match.end() < len(long_raw) else ""
        if not _ASCII_ALNUM.match(before) and not _ASCII_ALNUM.match(after):
            return True
    return False


def _name_containment_match(target_name, drug_name):
    """Direction-agnostic whole-token containment; the shorter name must match."""
    target_key, drug_key = _name_key(target_name), _name_key(drug_name)
    if not target_key or not drug_key or target_key == drug_key:
        return False
    if len(target_key) <= len(drug_key):
        return _contains_whole_name(target_name, drug_name)
    return _contains_whole_name(drug_name, target_name)


def _route_reaction_compounds(out_root, index_rows, routes, fragments):
    """reaction_id -> business compound rows, for the documents routes touch."""
    needed = {doc for row in routes + fragments
              for doc in [row["document_id"], *(row.get("volumes") or []),
                          *(step.get("document_id") for step in row.get("steps") or [])] if doc}
    cache_dir = Path(out_root) / "scan" / "records"
    by_uuid = {}
    by_reaction = {}
    for row in index_rows:
        if row.get("document_id") not in needed:
            continue
        records = read_gzip_jsonl(cache_dir / (row["uuid"] + ".json.gz"))
        for record in records:
            for reaction in record.get("reactions") or []:
                reaction_identity = business_reaction_id(
                    row["document_id"], {"id": record["id"], "index": reaction["index"]})
                values, _slots, _normalized = business_compounds(reaction)
                by_reaction[reaction_identity] = values
    return by_reaction


def _compound_keys(compound):
    keys = set()
    inchikey = _inchikey(compound.get("canonical_smiles"))
    if inchikey:
        keys.add("structure:" + inchikey)
        keys.add("skeleton:" + inchikey[:14])
    name = _name_key(compound.get("name"))
    if name:
        keys.add("name:" + name)
    return keys


def _relation_for_target(target, drug_row, channels):
    """Classify one route target against one candidate drug row."""
    evidence = {"channels": list(channels), "structure_match": "none", "name_match": None}
    target_smiles = target.get("canonical_smiles")
    target_inchikey = target.get("inchikey") or _inchikey(target_smiles)
    drug_inchikey = drug_row.get("inchikey")
    target_name = target.get("name") or ""
    drug_names = [drug_row.get("name_raw"), drug_row.get("base_name"), drug_row.get("ingredient_en")]
    drug_names = [name for name in drug_names if name]
    degradation_product = bool(_DEGRADATION_NAME.search(target_name))
    if target_inchikey and drug_inchikey:
        if target_inchikey == drug_inchikey:
            evidence["structure_match"] = "full"
            return "药物", _form_kind(target_name) if _form_kind(target_name) != "none" else (
                "salt" if drug_row.get("salt_form") else "none"), "high", evidence
        if target_inchikey[:14] == drug_inchikey[:14]:
            evidence["structure_match"] = "skeleton"
        if degradation_product:
            evidence["relation_guard"] = "degradation_product"
            return "其他", "none", "none", evidence
        parent = _parent_inchikey(target_smiles)
        drug_parent = _parent_inchikey(drug_row.get("smiles"))
        form = _form_kind(target_name)
        if parent and parent == drug_parent and (form != "none" or drug_row.get("salt_form")):
            evidence["structure_match"] = "stereo_preserving_parent"
            return "相关形式", form if form != "none" else "salt", "medium", evidence
        # The second block can encode stereo/isotope differences or unspecified
        # stereo. Neither a name nor a salt word resolves that disagreement.
        evidence["relation_guard"] = ("stereochemistry_mismatch_or_unspecified"
                                      if target_inchikey[:14] == drug_inchikey[:14]
                                      else "structure_mismatch")
        return ("未确认" if evidence["structure_match"] == "skeleton" else "其他"), "none", "none", evidence
    if target_name:
        target_key = _name_key(target_name)
        for name in drug_names:
            drug_key = _name_key(name)
            if not drug_key:
                continue
            if target_key == drug_key:
                evidence["name_match"] = "exact"
                return "药物", _form_kind(target_name), "medium", evidence
            if degradation_product:
                # Keep scanning for an exact name match, but never promote a
                # degradation/hydrolysis product to a salt or crystal form.
                continue
            if _name_containment_match(target_name, name):
                evidence["name_match"] = "contains"
                return "相关形式", _form_kind(target_name), "low", evidence
        if degradation_product:
            evidence["relation_guard"] = "degradation_product"
            return "其他", "none", "none", evidence
    if _DOSAGE_NAME.search(target_name):
        return "剂型", "none", "low", evidence
    if target_smiles or target_name:
        return "其他", "none", "none", evidence
    return "未确认", "none", "none", evidence


def _route_drug_rows(out_root, routes, fragments, doc_index, catalog, reactions):
    rows = []
    # Pass 1: direct drug/form/dosage relations.
    for route in routes + fragments:
        document_id = route["document_id"]
        document = doc_index.get(document_id) or {}
        channels = document.get("match_channels") or {}
        candidate_ids = document.get("drug_ids") or []
        target = dict(route.get("target") or {})
        target["inchikey"] = _inchikey(target.get("canonical_smiles"))
        candidates = []
        for drug_id in candidate_ids:
            drug_row = catalog.get(drug_id)
            if not drug_row:
                continue
            relation, form_kind, confidence, evidence = _relation_for_target(target, drug_row, channels)
            score = {"high": 5, "medium": 3, "low": 1, "none": 0}[confidence]
            candidate = {"drug_id": drug_id, "relation_type": relation, "form_kind": form_kind,
                         "confidence": confidence, "evidence": evidence, "score": score}
            candidates.append(candidate)
        best = max(candidates, key=lambda c: c["score"], default=None)
        extra = [c for c in candidates if best and c is not best
                 and c["score"] == best["score"] and c["score"] > 0]
        route_status = "complete" if "route_id" in route else "fragment"
        if best is None:
            row = {"drug_id": None, "relation_type": "未确认", "form_kind": "none",
                   "confidence": "none",
                   "evidence": {"channels": list(channels), "reason": "no_document_drug_candidate"},
                   "score": 0}
        else:
            row = best
        unresolved = []
        if row["relation_type"] == "未确认":
            unresolved = list(document.get("unresolved_reason") or []) or [
                "no_document_drug" if not candidate_ids else "no_structure_or_name_match"]
        base = {
            "schema_version": "route-drug-v1",
            "route_id": route.get("route_id"),
            "fragment_id": route.get("fragment_id"),
            "route_status": route_status,
            "document_id": document_id,
            "scope_id": route.get("scope_id"),
            "patent_id": route.get("patent_id"),
            "target": {"reaction_id": target.get("reaction_id"),
                       "compound_id": target.get("compound_id"),
                       "name": target.get("name"),
                       "canonical_smiles": target.get("canonical_smiles"),
                       "inchikey": target.get("inchikey"),
                       "inchikey_skeleton": target.get("inchikey")[:14] if target.get("inchikey") else None},
            "candidate_drug_ids": list(candidate_ids),
            "primary_drug_id": document.get("primary_drug_id"),
            "drug_id": row["drug_id"],
            "relation_type": row["relation_type"],
            "form_kind": row["form_kind"],
            "confidence": row["confidence"],
            "evidence": row["evidence"],
            "unresolved_reason": unresolved,
        }
        rows.append({**base, "_score": row["score"]})
        for candidate in extra:
            rows.append({**base, "drug_id": candidate["drug_id"],
                         "relation_type": candidate["relation_type"],
                         "form_kind": candidate["form_kind"],
                         "confidence": candidate["confidence"],
                         "evidence": candidate["evidence"], "_score": candidate["score"]})
    # Pass 2: materials on a confirmed drug route become precursors of that drug.
    from .routes import _material_form
    drug_targets = defaultdict(lambda: defaultdict(set))
    route_by_id = {item.get("route_id") or item.get("fragment_id"): item for item in routes + fragments}
    for row in rows:
        if row["relation_type"] not in {"药物", "相关形式"}:
            continue
        scope = row.get("scope_id") or ""
        route = route_by_id.get(row.get("route_id") or row.get("fragment_id"))
        if route is None:
            continue
        for reaction_id in route.get("reaction_ids") or []:
            for compound in reactions.get(reaction_id) or []:
                roles = set(compound.get("roles") or [])
                # Only material that actually feeds the route can be a
                # precursor: reactants/participants, never products, solvents or
                # reagents (a merged ["reagent", "participant"] row stays a
                # reagent and is excluded).
                if not roles & {"reactant", "participant"}:
                    continue
                if roles & {"product", "solvent", "reagent"}:
                    continue
                for key in _compound_keys(compound):
                    if key.startswith("structure:") or key.startswith("name:"):
                        drug_targets[(scope, key)][row["drug_id"]].add((
                            row.get("route_id") or row.get("fragment_id"),
                            _inchikey(compound.get("canonical_smiles")), _material_form(compound)))
    linked_rows = []
    for row in rows:
        if row["relation_type"] not in {"其他", "未确认"} or not row.get("target"):
            linked_rows.append(row)
            continue
        target = row["target"]
        keys = set()
        if target.get("inchikey"):
            keys.add("structure:" + target["inchikey"])
        if target.get("name"):
            keys.add("name:" + _name_key(target["name"]))
        matches = defaultdict(set)
        for key in keys:
            for drug_id, materials in drug_targets.get((row.get("scope_id") or "", key), {}).items():
                for path, structure, form in materials:
                    if target.get("inchikey") and structure and target["inchikey"] != structure:
                        continue
                    if _material_form(target) != form:
                        continue
                    matches[drug_id].add(path)
        matches = {drug_id: paths for drug_id, paths in matches.items() if paths}
        if matches:
            for drug_id, paths in sorted(matches.items()):
                linked_rows.append({**row, "drug_id": drug_id, "relation_type": "前体",
                                    "confidence": "medium", "unresolved_reason": [],
                                    "evidence": dict(row.get("evidence") or {}, precursor_path=sorted(paths))})
        else:
            linked_rows.append(row)
    for row in linked_rows:
        row.pop("_score", None)
    return linked_rows


def run_drug_tables(out_root, source_paths=None):
    """Write the drug catalog/patent/document/route relation tables."""
    from .drug_links import (build_document_index, document_drugs, drug_catalog,
                             drug_patents, load_drug_sources)

    out_root = Path(out_root).resolve()
    paths = dict(DEFAULT_DRUG_SOURCES)
    paths.update(source_paths or {})
    sources = load_drug_sources(paths["fda"], paths["cn_metadata"],
                                paths["cn_registrations"], paths["chembl"])
    index_rows = read_jsonl(out_root / "scan" / "documents.jsonl")
    documents = [{"source": row.get("source"), "uuid": row["uuid"],
                  "document_id": row.get("document_id"),
                  "output_dir": str(Path(row["delivery_dir"]).parent)}
                 for row in index_rows if row.get("delivery_dir")]
    doc_index = build_document_index(sources, None, documents)
    catalog_rows = list(drug_catalog(sources))
    catalog = {row["drug_id"]: row for row in catalog_rows}
    drugs_dir = out_root / "drugs"
    write_jsonl(drugs_dir / "drug_catalog.jsonl", catalog_rows,
                sort_key=lambda row: row["drug_id"])
    write_jsonl(drugs_dir / "drug_patent.jsonl", list(drug_patents(sources, doc_index)),
                sort_key=lambda row: (row["drug_id"], row["patent_key"]))
    write_jsonl(drugs_dir / "document_drug.jsonl", list(document_drugs(doc_index)),
                sort_key=lambda row: row["document_id"])
    routes = read_jsonl(out_root / "routes.jsonl")
    fragments = read_jsonl(out_root / "route_fragments.jsonl")
    reactions = _route_reaction_compounds(out_root, index_rows, routes, fragments)
    drug_rows = _route_drug_rows(out_root, routes, fragments, doc_index, catalog, reactions)
    write_jsonl(drugs_dir / "route_drug.jsonl", drug_rows,
                sort_key=lambda row: (row.get("route_id") or row.get("fragment_id"),
                                      row.get("drug_id") or ""))
    relations = Counter(row["relation_type"] for row in drug_rows)
    complete = [row for row in drug_rows if row["route_status"] == "complete"]
    confirmed = [row for row in complete
                 if row["relation_type"] in {"药物", "相关形式", "前体", "剂型"}]
    matched_docs = sum(1 for row in doc_index.values() if row.get("drug_ids"))
    summary = {
        "drug_catalog": len(catalog_rows),
        "drug_patent": sum(1 for _ in drug_patents(sources, doc_index)),
        "documents": len(doc_index),
        "documents_with_drug": matched_docs,
        "document_drug_coverage": round(matched_docs / len(doc_index), 4) if doc_index else 0,
        "route_drug_rows": len(drug_rows),
        "route_relations": dict(relations),
        "complete_routes": len({row["route_id"] for row in complete}),
        "complete_routes_with_chemical_relation": len({row["route_id"] for row in confirmed}),
        "complete_route_coverage": round(len({row["route_id"] for row in confirmed}) /
                                         len({row["route_id"] for row in complete}), 4) if complete else 0,
        "unresolved": sum(1 for row in drug_rows if row["relation_type"] == "未确认"),
    }
    write_json(drugs_dir / "summary.json", summary)
    return summary

