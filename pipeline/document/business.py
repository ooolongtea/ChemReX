"""PDF-scoped reaction-v1 / route-v1 business contract, without model calls."""

from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
import re

from .routes import _canonical, compound_lists, eligible, details_record_kind, parent_processing_complete
from pipeline.validation.quality import QUALITY_VERSION

EXPORTER_VERSION = "2026-10-13-quality-repair"
ROLES = {"reactants": "reactant", "products": "product", "reagents": "reagent",
         "solvents": "solvent", "participants": "participant"}
KINDS = {"synthesis", "formulation", "purification", "unknown"}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def filename_ids(documents):
    """Use PDF filenames; qualify duplicates with their existing result folder."""
    counts = Counter(filename for label, filename in documents)
    return {label: filename if counts[filename] == 1 else f"{label}/{filename}"
            for label, filename in documents}


def readable_part(value):
    # Keep Unicode/filename text readable while escaping the ID separator.
    return str(value).replace("%", "%25").replace(":", "%3A")


def text(value):
    if value is None:
        return None
    if isinstance(value, list):
        values = [text(v) for v in value]
        return "; ".join(v for v in values if v) or None
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    return str(value) if str(value).strip() else None


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def details(reaction):
    """Pick the latest layer with business content, respecting explicit empties."""
    return next((reaction[k] for k in ("validation", "coref", "extraction")
                 if isinstance(reaction.get(k), dict) and "ner" in reaction[k]), {})


def process_kind(record, reaction):
    value = details(reaction).get("process_kind") or (record.get("source") or {}).get("process_kind")
    return value if value in KINDS else "unknown"


def step_key(ref):
    return ref["id"], ref["index"]


def reaction_id(document_id, ref):
    return f"{readable_part(document_id)}::{readable_part(ref['id'])}::{ref['index']}"


def _structure(compound, normalized, conflict=False):
    if conflict or "conflict" in str(compound.get("structure_source", "")):
        return None
    if compound.get("status") not in (None, "valid"):
        return None
    candidate = compound.get("canonical_smiles")
    if not normalized:
        candidate = candidate or compound.get("smiles")
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    return _canonical(candidate)


def compounds(reaction):
    rv = (reaction.get("validation") or {}).get("rxn_validation") or {}
    normalized = isinstance(rv.get("normalized_compounds"), dict)
    lists = compound_lists(reaction)
    quality = reaction.get("quality") or (reaction.get("validation") or {}).get("quality") or {}
    result, slots, occurrences = [], {}, {}
    for role, singular in ROLES.items():
        for index, value in enumerate(lists.get(role) or []):
            value = value if isinstance(value, dict) else {"name": value}
            name = text(value.get("name")) or text(value.get("standard_name"))
            canonical = _structure(value, normalized,
                                   role == "products" and quality.get("structure_status") == "conflict")
            if name is None and canonical is None:
                continue
            # A mapping patch can list the same source reagent as participant.
            # Equal names/structures alone never merge different occurrences.
            provenance = (value.get("source_field"), value.get("source_index"), name, canonical)
            prior = occurrences.get(provenance) if provenance[0] is not None and provenance[1] is not None else None
            merge = prior is not None and {singular, *prior["roles"]} <= {"reagent", "participant"}
            if merge and singular not in prior["roles"]:
                item = prior
                item["roles"].append(singular)
            else:
                item = {"compound_id": f"c{len(result) + 1}", "roles": [singular],
                        "name": name, "canonical_smiles": canonical}
                if canonical and text(value.get("cxsmiles")):
                    item["cxsmiles"] = value["cxsmiles"]
                if isinstance(value.get("stereochemistry"), dict) and value["stereochemistry"]:
                    item["stereochemistry"] = deepcopy(value["stereochemistry"])
                for diagnostic in ('resolution_status', 'resolution_trace', 'discrete_form_evidence'):
                    if value.get(diagnostic) is not None:
                        item[diagnostic] = deepcopy(value[diagnostic])
                result.append(item)
                occurrences.setdefault(provenance, item)
            slots[(role, index)] = item["compound_id"]
    return result, slots, normalized


def _yield(info, issues):
    from pipeline.extraction.yield_values import parse_yield_value, yield_equal
    if not isinstance(info, dict) or info.get("kind") not in {
            "explicit", "derived", "not_reported", "not_extracted", "unattributed"}:
        return None
    kind = info["kind"]
    result = {"kind": kind, "value": None, "unit": "%"}
    if kind in {"explicit", "derived"}:
        parsed = parse_yield_value(info.get('value'))
        if not parsed:
            issues.append("yield_value_uninterpretable")
            return None
        value = parsed['value']
        if kind == "explicit":
            if not info.get("source_text") or not any(yield_equal(v, info.get('value')) for v in info.get('reported_values') or []):
                issues.append("yield_reported_evidence_missing")
                return None
            if parsed['range']:
                low, high = parsed['range']['lower'], parsed['range']['upper']
                if not 0 <= low <= high <= 100:
                    issues.append('yield_range_invalid')
                result['range'] = parsed['range']
            if parsed['range'] or info.get('approximate') or parsed['approximate']:
                result.update(raw=info.get('raw') or str(info.get('value')),
                              approximate=bool(info.get('approximate') or parsed['approximate']))
        else:
            if value is None:
                issues.append('yield_calculation_basis_invalid')
                return None
            basis = deepcopy(info.get("basis"))
            if not isinstance(basis, dict) or not basis:
                issues.append("yield_calculation_basis_missing")
                return None
            for key in ("substrate_mmol", "product_mmol", "decimals"):
                if key in basis:
                    basis[key] = number(basis[key])
                    if basis[key] is None:
                        issues.append("yield_basis_uninterpretable")
                        return None
            a, b, decimals = (basis.get(k) for k in ("substrate_mmol", "product_mmol", "decimals"))
            if (a is None or b is None or decimals is None or a <= 0 or b < 0 or b > a
                    or decimals != int(decimals) or not 0 <= decimals <= 12
                    or basis.get("formula") != "100 * product_mmol / substrate_mmol"
                    or abs(round(100 * b / a, int(decimals)) - value) > .02):
                issues.append("yield_calculation_basis_invalid")
                return None
            result["basis"] = basis
        result["value"] = value
    elif kind == "not_extracted":
        issues.append("source_yield_not_extracted")
        reported = [parse_yield_value(v) for v in info.get("reported_values") or []]
        result["reported_values"] = [v['value'] for v in reported if v and v['value'] is not None]
        ranges = [v['range'] for v in reported if v and v['range']]
        if ranges:
            result['reported_ranges'] = ranges
        if any(v is None for v in reported):
            issues.append("reported_yield_uninterpretable")
    elif kind == "unattributed":
        issues.append("yield_unattributed")
    return result


def _pages(parts):
    # MinerU page_idx is zero-based; chunk.page_start is one-based and is
    # deliberately not used as a fallback for an individual reaction.
    return sorted({p["page_idx"] + 1 for p in parts if isinstance(p, dict)
                   and type(p.get("page_idx")) is int and p["page_idx"] >= 0})


def _source(record, reaction, issues):
    context = reaction.get("step_source") or details(reaction).get("step_source") or {}
    source = {"chunk_id": record["id"], "reaction_index": reaction["index"],
              "pages": _pages(context.get("source_parts") or []),
              "text": text(context.get("local_text")) or text(reaction.get("text"))}
    if not source["text"]:
        issues.append("step_source_text_missing")
    references = []
    for reference in context.get("procedure_references") or []:
        if reference.get('relation') == 'precursor_preparation':
            references.append({'kind': 'precursor_preparation', 'reference_id': reference.get('id'),
                               'status': reference.get('status'), 'pages': _pages(reference.get('blocks') or [])})
            continue
        ref_id = text(reference.get("id"))
        if not ref_id:
            issues.append("procedure_reference_id_missing")
            continue
        status = text(reference.get("status")) or "missing"
        references.append({"kind": "procedure", "reference_id": ref_id, "status": status,
                           "pages": _pages(reference.get("blocks") or [])})
        procedure_type = reference.get('heading_type') or reference.get('kind') or (
            'general' if '_' not in ref_id else ref_id.rsplit('_', 1)[0])
        if procedure_type != 'procedure':
            references[-1]['procedure_type'] = procedure_type
        if status == 'resolved':
            references[-1]['target'] = {
                'chunk_id': reference.get('chunk_id'), 'parent_example_id': reference.get('parent_id'),
                'blocks': [{k: b.get(k) for k in ('block_idx', 'page_idx', 'bbox')}
                           for b in reference.get('blocks') or []]}
        if status != "resolved":
            issues.append("procedure_reference_" + status)
    if references:
        source["references"] = references
    return source


def _expression(rv, issues):
    value = rv.get("effective_expression") if "effective_expression" in rv else (
        rv.get("best_expression") if rv.get("patch_accepted") is True else rv.get("base_expression"))
    value = text(value)
    if not value:
        return None
    sides = value.split(">")
    if len(sides) != 3 or not sides[0] or not sides[2] or any(s and _canonical(s) is None for s in sides):
        issues.append("reaction_expression_invalid")
        return None
    return value


def business_reaction(document_id, record, reaction):
    data = details(reaction)
    ner = data.get("ner") or {}
    rv = (reaction.get("validation") or {}).get("rxn_validation") or {}
    quality = reaction.get("quality") or (reaction.get("validation") or {}).get("quality") or {}
    issues = list(quality.get("blocking_issues") or [])
    if not quality:
        issues.append("quality_evidence_missing")
    elif quality.get("version") != QUALITY_VERSION:
        issues.append("quality_policy_version_unsupported")
    if not parent_processing_complete(record):
        issues.append("chunk_requires_review")
    if details_record_kind(reaction) == 'procedure_template':
        issues.append('procedure_template_without_specific_instance')
    if reaction.get("status") != "ok":
        issues.append("reaction_record_requires_review")
    if rv.get("status") not in {"passed", "corrected"} or rv.get("needs_human_review", True) is not False:
        issues.append("mapping_or_structure_requires_review")
    values, slots, normalized = compounds(reaction)
    products = [c for c in values if "product" in c["roles"]]
    if not products or any(c["canonical_smiles"] is None for c in products):
        issues.append("product_structure_unavailable")
    if not normalized:
        issues.append("normalized_compounds_missing")
    kind = process_kind(record, reaction)
    if kind != "synthesis":
        issues.append("process_policy_not_supported")
    source = _source(record, reaction, issues)
    yield_value = _yield(data.get("yield_info"), issues)
    stereo = data.get("stereochemistry")
    stereo = {"kind": text(stereo.get("kind")), "descriptors": list(stereo.get("descriptors") or []),
              "structure_coarsened": stereo.get("structure_coarsened") is True} if isinstance(stereo, dict) and stereo else None
    if stereo and stereo["structure_coarsened"]:
        issues.append("stereochemistry_coarsened")
    expression = None if 'source_structure_conflict' in issues else _expression(rv, issues)
    if not eligible(record, reaction) and not issues:
        issues.append("quality_not_admitted")
    issues = list(dict.fromkeys(issues))
    result = {"schema_version": "reaction-v1", "document_id": document_id,
              "reaction_id": reaction_id(document_id, {"id": record["id"], "index": reaction["index"]}),
              "record_kind": details_record_kind(reaction),
              "process_kind": kind, "compounds": values,
              "conditions": {"temperature": text(ner.get("TEMPERATURE")), "time": text(ner.get("TIME")), "yield": yield_value},
              "actions": [v for action in data.get("action") or [] if (v := text(action))],
              "reaction_smiles": expression, "stereochemistry": stereo, "source": source,
              "quality": {"admission_eligible": eligible(record, reaction) and not issues,
                          "blocking_issues": issues, "policy_version": QUALITY_VERSION}}
    extraction_config = (reaction.get("extraction") or {}).get("extraction_configuration") or {}
    if extraction_config:
        # Additive provenance: which M1 model/preset produced this reaction.
        result["source_models"] = {"m1": {
            "model": extraction_config.get("model"),
            "preset": extraction_config.get("preset"),
        }}
    return result, slots


def _route_issue(issue, document_id):
    result = {"code": issue["reason"]}
    if issue.get("step"):
        result["reaction_ids"] = [reaction_id(document_id, issue["step"])]
    if issue.get("steps"):
        result["reaction_ids"] = list(dict.fromkeys(reaction_id(document_id, s) for s in issue["steps"]))
    if issue.get("candidates"):
        result["candidate_reaction_ids"] = list(dict.fromkeys(reaction_id(document_id, s) for s in issue["candidates"]))
    if issue.get("id"):
        result["reference_id"] = issue["id"]
    return result


def business_records(document_id, records, groups):
    reactions, by_step, slots = [], {}, {}
    filtered, id_map = [], []
    required = {step_key(s) for g in groups for m in g["methods"] for s in m["steps"]}
    required |= {step_key(s) for g in groups for m in g["methods"] for i in m["issues"] for s in i.get("candidates", [])}
    for record in records:
        for reaction in record.get("reactions") or []:
            key = record["id"], reaction["index"]
            row, compound_slots = business_reaction(document_id, record, reaction)
            raw_kind = (record.get("source") or {}).get("process_kind")
            meaningful = bool(row["compounds"] or row["actions"] or row["conditions"]["temperature"] or row["conditions"]["time"])
            if key not in required and (raw_kind in {"assay", "analysis"} or not meaningful):
                filtered.append({"chunk_id": key[0], "reaction_index": key[1],
                                 "reason": "analysis_only" if raw_kind in {"assay", "analysis"} else "failed_placeholder_without_business_content"})
                continue
            if key in by_step:
                raise ValueError(f"Duplicate reaction source identity: {key}")
            reactions.append(row)
            by_step[key], slots[key] = row, compound_slots
            id_map.append({"chunk_id": key[0], "reaction_index": key[1], "step_id": reaction.get("step_id"),
                           "chunk_fingerprint": record.get("chunk_fingerprint"), "reaction_id": row["reaction_id"],
                           "compound_slots": [{"role": role, "index": index, "compound_id": cid} for (role, index), cid in compound_slots.items()]})
    order = {r["reaction_id"]: i for i, r in enumerate(reactions)}
    routes, route_map = [], []
    for group in groups:
        anchor = group["methods"][0]
        product_number = (anchor["target_product_index"] + 1) if anchor["target_product_index"] is not None else "title"
        group_id = reaction_id(document_id, anchor["target_step"]) + f"::target::{product_number}"
        for method in group["methods"]:
            terminal = method["target_step"]
            terminal_key = step_key(terminal)
            target_id = slots[terminal_key].get(("products", method["target_product_index"]))
            target = {"reaction_id": by_step[terminal_key]["reaction_id"], "compound_id": target_id}
            if target_id is None:
                target.update(name=method["target_name"], basis="source_title")
            product_number = (method["target_product_index"] + 1) if method["target_product_index"] is not None else "title"
            rid = reaction_id(document_id, terminal) + f"::route::{product_number}"
            steps = sorted({by_step[step_key(s)]["reaction_id"] for s in method["steps"]}, key=order.get)
            connections, issues = [], [_route_issue(i, document_id) for i in method["issues"]]
            for edge in method["connections"]:
                from_key, to_key = step_key(edge["from"]), step_key(edge["to"])
                product_id = slots[from_key].get(("products", edge["product_index"]))
                input_id = slots[to_key].get((edge["input_role"], edge["input_index"]))
                if product_id is None or input_id is None:
                    issues.append({"code": "connection_material_unavailable",
                                   "reaction_ids": [by_step[from_key]["reaction_id"], by_step[to_key]["reaction_id"]]})
                    continue
                evidence = edge["evidence"]
                connection = {"from": by_step[from_key]["reaction_id"], "to": by_step[to_key]["reaction_id"],
                              "product_id": product_id, "input_id": input_id,
                              "basis": "material_reference" if evidence.get("material_reference") else evidence["kind"]}
                if evidence.get("material_reference"):
                    connection["evidence_text"] = evidence["material_reference"]
                if connection not in connections:
                    connections.append(connection)
            blocked = [s for s in steps if not reactions[order[s]]["quality"]["admission_eligible"]]
            if blocked:
                issues.insert(0, {"code": "step_quality_blocked", "reaction_ids": blocked})
            if target_id is None and not any(i["code"] == "target_from_title_without_extracted_product" for i in issues):
                issues.append({"code": "target_from_title_without_extracted_product", "reaction_ids": [target["reaction_id"]]})
            routes.append({"schema_version": "route-v1", "document_id": document_id,
                           "route_id": rid, "target_group_id": group_id,
                           "target_group_basis": method["target_group_basis"], "title": text(method["title"]),
                           "process_kind": method["process_kind"] if method["process_kind"] in KINDS else "unknown",
                           "target": target, "steps": steps, "connections": connections,
                           "quality": {"admission_eligible": not issues, "blocking_issues": issues, "policy_version": QUALITY_VERSION}})
            route_map.append({"old_route_id": group["route_id"], "old_method_id": method["method_id"], "route_id": rid})
    routes.sort(key=lambda r: (order[r["target"]["reaction_id"]], r["target"]["compound_id"] or "", r["route_id"]))
    validate_snapshot(document_id, reactions, routes)
    return reactions, routes, {"reactions": id_map, "routes": route_map, "filtered_reactions": filtered}


def validate_snapshot(document_id, reactions, routes):
    """Reject incomplete references, bad roles/types/IDs, and cyclic graphs."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    by_id, material = {}, {}
    for row in reactions:
        require(row["schema_version"] == "reaction-v1" and row["document_id"] == document_id, "Reaction schema/document mismatch")
        rid = row["reaction_id"]
        require(rid not in by_id, "Duplicate reaction ID")
        by_id[rid] = row
        ids = set()
        for c in row["compounds"]:
            cid = c["compound_id"]
            require(cid not in ids and c["roles"] and set(c["roles"]) <= set(ROLES.values()), "Invalid compound ID/roles")
            require(c["name"] is not None or c["canonical_smiles"] is not None, "Empty compound placeholder")
            ids.add(cid)
            material[(rid, cid)] = c
        require(row["process_kind"] in KINDS, "Invalid process kind")
        require(all(type(p) is int and p >= 1 for p in row["source"]["pages"]), "Invalid PDF page index")
        require(all(isinstance(a, str) and a for a in row["actions"]), "Invalid action sequence")
        require(all(row["conditions"][k] is None or isinstance(row["conditions"][k], str) for k in ("temperature", "time")), "Invalid condition type")
        value = row["conditions"]["yield"]
        if value:
            require(value["value"] is None or type(value["value"]) in (int, float), "Invalid yield type")
            if value["kind"] in {"not_reported", "not_extracted", "unattributed"}:
                require(value["value"] is None, "Unconfirmed yield cannot be numeric")
            if value["kind"] == "derived":
                require(bool(value.get("basis")), "Derived yield lacks basis")
            if value.get('range'):
                require(value['value'] is None and value['kind'] == 'explicit', 'Yield interval cannot be a scalar')
                require(set(value['range']) == {'lower', 'upper'} and all(type(v) in (float, int) for v in value['range'].values()), 'Invalid yield interval')
        quality = row["quality"]
        require(type(quality["admission_eligible"]) is bool and all(isinstance(i, str) for i in quality["blocking_issues"]), "Invalid reaction quality types")
        if quality["admission_eligible"]:
            require(not quality["blocking_issues"] and quality["policy_version"] == QUALITY_VERSION, "Unqualified reaction admitted")
    route_ids = set()
    for row in routes:
        require(row["schema_version"] == "route-v1" and row["document_id"] == document_id, "Route schema/document mismatch")
        require(row["route_id"] not in route_ids, "Duplicate route ID")
        route_ids.add(row["route_id"])
        steps = row["steps"]
        require(bool(steps) and len(steps) == len(set(steps)) and set(steps) <= by_id.keys(), "Route steps not closed/unique")
        target = row["target"]
        require(target["reaction_id"] in steps, "Target outside route")
        if target["compound_id"] is None:
            require(target.get("name") and target.get("basis") == "source_title" and any(i["code"] == "target_from_title_without_extracted_product" for i in row["quality"]["blocking_issues"]), "Title target lacks source/blocker")
        else:
            c = material.get((target["reaction_id"], target["compound_id"]))
            require(c is not None and "product" in c["roles"], "Target does not reference product")
        adjacency = {s: set() for s in steps}
        indegree = {s: 0 for s in steps}
        for edge in row["connections"]:
            require(edge["from"] in steps and edge["to"] in steps, "Connection outside route")
            a = material.get((edge["from"], edge["product_id"]))
            b = material.get((edge["to"], edge["input_id"]))
            require(a is not None and "product" in a["roles"] and b is not None and bool(set(b["roles"]) & {"reactant", "participant"}), "Connection material roles invalid")
            if edge["to"] not in adjacency[edge["from"]]:
                adjacency[edge["from"]].add(edge["to"])
                indegree[edge["to"]] += 1
        pending = [s for s, count in indegree.items() if count == 0]
        visited = 0
        while pending:
            step = pending.pop()
            visited += 1
            for downstream in adjacency[step]:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    pending.append(downstream)
        require(visited == len(steps), "Cyclic route graph")
        quality = row["quality"]
        require(type(quality["admission_eligible"]) is bool, "Invalid route admission type")
        for issue in quality["blocking_issues"]:
            require(isinstance(issue, dict) and isinstance(issue.get("code"), str), "Invalid route blocker")
            require(set(issue.get("reaction_ids", [])) <= set(steps), "Route blocker points outside steps")
            require(set(issue.get("candidate_reaction_ids", [])) <= by_id.keys(), "Unclosed ambiguity candidate")
        if quality["admission_eligible"]:
            require(not quality["blocking_issues"] and quality["policy_version"] == QUALITY_VERSION and all(by_id[s]["quality"]["admission_eligible"] for s in steps), "Unqualified route admitted")
