"""Conservative, document-local route assembly. No models or new structures."""

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import re
from functools import lru_cache


VERSION = "2026-10-13-quality-repair"
COMPOUND_FIELDS = {"reactants": "STARTING_MATERIAL", "products": "REACTION_PRODUCT",
                   "reagents": "REAGENT_CATALYST", "solvents": "SOLVENT"}
EXCLUDED_KINDS = {"assay", "analysis"}
FORM = re.compile(r"结晶形式|晶型|无定形|溶剂合物|水合物|(?:hydrochloride|hydrobromide|mesylate|maleate|sulfate|citrate|polymorph|crystal(?:line)? form|solvate|hydrate)\b|(?:盐酸|氢溴酸|甲磺酸|马来酸|硫酸|柠檬酸|磷酸|富马酸|酒石酸)盐", re.I)
FORMULA = re.compile(r"(?:式\s*([0-9]+|[IVXⅠⅡⅢ]+)\s*(?:的)?化合物|compound\s+(?:of\s+)?formula\s*([0-9]+|[IVX]+))", re.I)
GENERIC = re.compile(r"^(?:标题化合物|标题产物|目标产物|产物|所需化合物|化合物|粗产物|粗羰基|粗酰胺|title compound|the title compound|target product|product|desired compound|unknown|n/a)$", re.I)
METHOD_REFERENCE = re.compile(r"类似.{0,12}(?:方法|程序)|参照|参考|按照.{0,30}(?:方法|程序)|采用.{0,30}(?:类似|方法)|similar|procedure (?:of|described|disclosed)|according to", re.I)


def pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def compound_lists(reaction):
    """Keep normalization authoritative; support failed/legacy extraction."""
    validation = reaction.get("validation") or {}
    rv = validation.get("rxn_validation") or {}
    normalized = rv.get("normalized_compounds")
    if isinstance(normalized, dict):
        return normalized
    details = next((reaction.get(k) for k in ("validation", "coref", "extraction")
                    if (reaction.get(k) or {}).get("ner")), validation or reaction.get("coref") or reaction.get("extraction") or {})
    ner = details.get("ner") or {}
    result = {}
    for role, key in COMPOUND_FIELDS.items():
        values = ner.get(key) or []
        if not isinstance(values, list):
            values = [values]
        result[role] = [v if isinstance(v, dict) else {"name": v} for v in values]
    result["participants"] = details.get("reactant_side_participants") or []
    return result


def eligible(record, reaction):
    from pipeline.validation.quality import QUALITY_VERSION
    quality = reaction.get("quality") or (reaction.get("validation") or {}).get("quality") or {}
    rv = (reaction.get("validation") or {}).get("rxn_validation") or {}
    return bool(quality.get("admission_eligible") is True and quality.get("version") == QUALITY_VERSION
                and not quality.get("blocking_issues") and parent_processing_complete(record)
                and details_record_kind(reaction) != 'procedure_template'
                and reaction.get("status") == "ok" and rv.get("status") in {"passed", "corrected"}
                and rv.get("needs_human_review") is False)


def parent_processing_complete(record):
    from .step_context import CONTEXT_VERSION
    # Legacy parent-level review may carry evidence not attributed to a step.
    return record.get('status') == 'ok' or (record.get('context_version') == CONTEXT_VERSION
            and record.get('status') in {'needs_review', 'non_reaction'})


def details_record_kind(reaction):
    return (reaction.get('step_source') or {}).get('record_kind') or next(
        ((reaction.get(k) or {}).get('record_kind') for k in ('validation', 'coref', 'extraction')
         if (reaction.get(k) or {}).get('record_kind')), 'reaction_candidate')


# A role prefix ("中间体006-8", "Intermediate 8", "产物2") is not part of a
# numbered compound code's identity: the same batch is written with and
# without it across the steps of one procedure. Letter codes ("化合物X")
# keep the prefix — stripping them would leave colliding single-letter keys.
_ROLE_PREFIX = re.compile(r"^(?:中间体|化合物|产物|产品|intermediate|compound)(?=\d)", re.I)


def _name(value):
    # Formatting only: no punctuation/stereo removal or fuzzy chemical matching.
    return _ROLE_PREFIX.sub("", re.sub(r"\s+", "", str(value or "")), count=1).casefold()


# Sentence boundaries are shared by every reference check. Chinese/full-width
# punctuation always ends a sentence; an English ".", "!" or "?" ends one when
# whitespace follows, when the next sentence starts lowercase, or when OCR
# dropped the space after a word/number ("Example 7.The mixture"). Decimals
# ("2.5 g") and common abbreviations ("e.g.", "No.", "m.p.") stay inside their
# sentence.
_CN_SENTENCE_END = re.compile(r"[。；;！？\n]+")
_EN_SENTENCE_BREAK = re.compile(
    r"[.!?](?=\s+\S)|(?<=[a-z0-9)\]])[.!?](?=[A-Z])")


# A period is not a boundary after one of these abbreviations. The tail match
# covers both one-part ("No.") and dotted ("e.g.", "m.p.") abbreviations.
_ABBREVIATION_TAIL = re.compile(r"((?:[A-Za-z]\.)*[A-Za-z]{1,6})$")
_SENTENCE_ABBREVIATIONS = frozenset({
    "e.g", "i.e", "no", "vs", "approx", "ref", "prep", "eq", "fig",
    "m.p", "b.p", "etc", "cf", "ca", "u.s", "u.s.a", "u.s.p", "inc",
    "ltd", "co", "st", "dr", "mr", "mrs", "ms", "prof", "vol", "pp", "p",
})


def _protected_period(text, position):
    match = _ABBREVIATION_TAIL.search(text[:position])
    return bool(match and match.group(1).casefold() in _SENTENCE_ABBREVIATIONS)


@lru_cache(maxsize=8192)
def _split_sentences(text):
    sentences = []
    for chunk in _CN_SENTENCE_END.split(text):
        start = 0
        for match in _EN_SENTENCE_BREAK.finditer(chunk):
            if chunk[match.start()] == "." and _protected_period(chunk, match.start()):
                continue
            sentence = chunk[start:match.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = match.end()
        tail = chunk[start:].strip()
        if tail:
            sentences.append(tail)
    return tuple(sentences)


def _smiles(compound):
    if compound.get("ignored") or compound.get("status") not in (None, "valid"):
        return None
    if "conflict" in str(compound.get("structure_source", "")):
        return None
    value = compound.get("canonical_smiles")
    if not value:
        return None
    return _canonical(value)


@lru_cache(maxsize=8192)
def _canonical(value):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def _match(left, right):
    if left.get("ignored") or right.get("ignored"):
        return None
    if any("conflict" in str(c.get("structure_source", "")) for c in (left, right)):
        return None
    a, b = _smiles(left), _smiles(right)
    if a and b:
        stereo_left, stereo_right = left.get("stereochemistry"), right.get("stereochemistry")
        if stereo_left and stereo_right and stereo_left != stereo_right:
            return None
        return "canonical_smiles" if a == b else None
    a, b = _name(left.get("name")), _name(right.get("name"))
    if a and a == b and not GENERIC.fullmatch(str(left.get("name", "")).strip()):
        return "exact_name"
    return None


def _formula(value):
    match = FORMULA.search(value or "")
    if not match:
        return None
    number = (match[1] or match[2]).upper()
    # A local label spelling convention; only the numeral one is ambiguous in OCR.
    return "1" if number in {"I", "Ⅰ", "1"} else number


def _base_name(value):
    value = re.sub(r"(?:的)?(?:制备|合成)\s*[:：]?\s*$", "", str(value or "")).strip()
    formula = FORMULA.match(value)
    if formula:
        return formula.group(0)
    match = FORM.search(value)
    if match:
        before = value[:match.start()].rstrip("的 -,")
        # A solvent name alone is not a parent compound.
        if "溶剂合物" in match.group(0):
            return ""
        return before
    return value


@dataclass(eq=False)
class Unit:
    record: dict
    reaction: dict
    order: int
    ref: dict
    compounds: dict
    edges: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    method_references: list = field(default_factory=list)

    @property
    def source(self):
        return self.record.get("source") or {}

    @property
    def context(self):
        return self.reaction.get("step_source") or {}

    @property
    def kind(self):
        return (self.reaction.get("validation") or {}).get("process_kind") or self.source.get("process_kind") or "unverified"


def _reference_evidence(unit, other, ingredient):
    """A mentioned example alone is insufficient; require material provenance."""
    source = unit.source
    if other.source.get("reference_id", other.record["id"]) not in (source.get("references") or []):
        return None
    text = unit.context.get("local_text") or unit.reaction.get("text") or source.get("procedure") or ""
    number = re.escape(str(other.source.get("number", "")))
    label = rf"(?:实施例|中间体|步骤|Example|Intermediate|Preparation|Step)\s*{number}(?!\d)"
    for sentence in _split_sentences(text):
        if not re.search(label, sentence, re.I):
            continue
        if METHOD_REFERENCE.search(sentence):
            continue
        if (re.search(r"(?:由|从|来自|所得|制得|得到|obtained|prepared|from)", sentence, re.I)
                and _name(ingredient.get("name")) in _name(sentence)):
            return {"kind": "material_reference", "text": sentence}
    return None


def _identity(product, unit, final_product):
    name = str(product.get("name") or "").strip()
    titles = [unit.context.get("title") or "", unit.source.get("title") or ""]
    display = pick(product, ("name", "canonical_smiles", "cxsmiles", "stereochemistry", "status", "reason"))
    if not name:
        canonical = _smiles(product) if unit.kind != "formulation" else None
        if canonical:
            display["canonical_smiles"] = canonical
            return "structure:" + canonical, display, "canonical_smiles"
        return None
    if GENERIC.fullmatch(name):
        return None
    if "conflict" in str(product.get("structure_source", "")):
        return None
    if FORM.search(name) or any(FORM.search(title) for title in titles):
        # Resolve a shortened form name against its own source title, never the abstract alone.
        label = _formula(name) or next((_formula(title) for title in titles if _formula(title)), None)
        if label:
            display = {"name": f"式{label}化合物"}
            anchor = final_product or {}
            if _formula(anchor.get("reference")) == label and anchor.get("product"):
                display["name"] = anchor["product"]
            return "formula:" + label, display, "source_formula_label"
        base = _base_name(name)
        if base and not GENERIC.fullmatch(base):
            return "name:" + _name(base), {"name": base}, "parent_name"
        return None
    # A formulation's active ingredient SMILES does not identify the formulation.
    canonical = None if unit.kind == "formulation" else _smiles(product)
    if canonical:
        display["canonical_smiles"] = canonical
        return "structure:" + canonical, display, "canonical_smiles"
    label = _formula(name)
    if label:
        return "formula:" + label, display, "source_formula_label"
    return "name:" + _name(name), display, "exact_name"


def _stereo_identity(unit, product):
    stereo = product.get("stereochemistry") or (unit.reaction.get("validation") or {}).get("stereochemistry") or {}
    if isinstance(stereo, dict) and stereo.get("kind") in {"relative", "mixture", "unassigned_isomer"}:
        return ":stereo:" + json.dumps(stereo, sort_keys=True, ensure_ascii=False)
    if product.get("cxsmiles"):
        return ":cx:" + product["cxsmiles"]
    return ""


def assemble_routes(records, final_product=None):
    """Return route groups and the disposition of every current reaction."""
    units = []
    unassigned = []
    for record in records:
        for reaction in record.get("reactions") or []:
            ref = {"id": record["id"], "index": reaction["index"], "step_id": reaction.get("step_id")}
            unit = Unit(record, reaction, len(units), ref, compound_lists(reaction))
            units.append(unit)
    active = [u for u in units if u.kind not in EXCLUDED_KINDS and
              (u.reaction.get("status") != "non_reaction" or u.kind == "formulation"
               or FORM.search(u.context.get("title", "")))]
    # A named salt/form may resolve to a single explicitly named covalent parent.
    parents = defaultdict(set)
    for unit in active:
        for product in unit.compounds.get("products", []):
            canonical = _smiles(product) if unit.kind != "formulation" else None
            if canonical and not FORM.search(str(product.get("name", ""))):
                parents[_name(product.get("name"))].add(canonical)
    for unit in active:
        text = unit.context.get("local_text") or unit.reaction.get("text") or ""
        if METHOD_REFERENCE.search(text):
            unit.method_references = list(dict.fromkeys(unit.source.get("references") or []))
        quality = unit.reaction.get("quality") or (unit.reaction.get("validation") or {}).get("quality") or {}
        if quality.get("structure_status") == "conflict":
            unit.issues.append({"reason": "structure_conflict"})
        inputs = [(role, index, value) for role in ("reactants", "participants")
                  for index, value in enumerate(unit.compounds.get(role, []))]
        for input_role, input_index, ingredient in inputs:
            matches = []
            for prior in active:
                if prior.order >= unit.order:
                    break
                prior_quality = prior.reaction.get("quality") or (prior.reaction.get("validation") or {}).get("quality") or {}
                if quality.get("structure_status") == "conflict" or prior_quality.get("structure_status") == "conflict":
                    continue
                for product_index, product in enumerate(prior.compounds.get("products", [])):
                    match = _match(product, ingredient)
                    if match:
                        matches.append((prior, match, product_index))
            local = [m for m in matches if m[0].record["id"] == unit.record["id"]]
            referenced = [(p, match, index, _reference_evidence(unit, p, ingredient)) for p, match, index in matches]
            referenced = [m for m in referenced if m[3]]
            if local:
                candidates = local
            elif referenced:
                candidates = [(p, match, index) for p, match, index, evidence in referenced]
            else:
                candidates = matches
                # Method reference to an example is not an input from that example.
                candidates = [m for m in candidates if m[0].source.get("reference_id") not in unit.method_references]
            unique = {(p.order, index): (p, match, index) for p, match, index in candidates}
            if len(unique) == 1:
                prior, match, product_index = next(iter(unique.values()))
                edge = {"from": prior.ref, "to": unit.ref,
                        "product_index": product_index, "input_role": input_role, "input_index": input_index,
                        "evidence": {"kind": match, "name": ingredient.get("name")}}
                evidence = _reference_evidence(unit, prior, ingredient)
                if evidence:
                    edge["evidence"]["material_reference"] = evidence["text"]
                unit.edges.append((prior.order, edge))
            elif len(unique) > 1:
                unit.issues.append({"reason": "ambiguous_upstream", "name": ingredient.get("name"),
                                    "candidates": [p.ref for p, _, _ in unique.values()]})
        references = unit.context.get("procedure_references") or []
        for reference in references:
            if reference.get('relation') == 'precursor_preparation':
                continue
            if reference.get("status") != "resolved":
                unit.issues.append({"reason": "procedure_reference_" + str(reference.get("status", "missing")),
                                    "id": reference.get("id")})
        if unit.context.get("unit_kind") == "telescoped":
            unit.issues.append({"reason": "telescoped_without_reported_intermediates"})

    consumed = {prior for u in active for prior, _ in u.edges}
    terminal_steps = {record["id"]: (record.get("reactions") or [{}])[-1].get("step_id")
                      for record in records if record.get("terminal_product_coverage") in {"supported", "unknown"}}
    groups = {}
    assigned = set()
    for terminal in active:
        if terminal.order in consumed:
            continue
        if (terminal.record["id"] in terminal_steps
                and terminal.reaction.get("step_id") != terminal_steps[terminal.record["id"]]):
            # The extractor identifies an example's final source step. Broken
            # links do not turn its earlier intermediates into new target products.
            continue
        products = terminal.compounds.get("products") or []
        identities = [(identity[0] + _stereo_identity(terminal, p), identity[1], identity[2], index)
                      for index, p in enumerate(products) if (identity := _identity(p, terminal, final_product))]
        title_target = False
        # Retain a specific formulation/preparation title when extraction has no product.
        if not identities and not products:
            title = terminal.context.get("title") or terminal.source.get("title") or ""
            if title and (terminal.kind == "formulation" or FORM.search(title)):
                candidate = {"name": re.sub(r"[:：]\s*$", "", title)}
                identity = _identity(candidate, terminal, final_product)
                if identity:
                    identities = [(*identity, None)]
                    title_target = True
        if not identities:
            continue
        ancestors = set()
        pending = [terminal.order]
        while pending:
            order = pending.pop()
            if order in ancestors:
                continue
            ancestors.add(order)
            pending.extend(p for p, _ in units[order].edges)
        disconnected = [u for u in active if u.record["id"] == terminal.record["id"]
                        and u.order < terminal.order and u.order not in ancestors]
        if terminal.record["id"] in terminal_steps:
            # Keep the procedure fragments under their actual method, with a
            # gap flag, rather than hiding them or manufacturing a connection.
            pending = [u.order for u in disconnected]
            while pending:
                order = pending.pop()
                if order not in ancestors:
                    ancestors.add(order)
                    pending.extend(p for p, _ in units[order].edges)
        selected = [units[i] for i in sorted(ancestors)]
        edges = [edge for u in selected for _, edge in u.edges]
        issues = [dict(issue, step=u.ref) for u in selected for issue in u.issues]
        if title_target:
            issues.append({"reason": "target_from_title_without_extracted_product", "step": terminal.ref})
        # Earlier units under this procedure title may be disconnected extraction fragments.
        if disconnected and terminal.record["id"] in terminal_steps:
            issues.append({"reason": "disconnected_procedure_steps", "steps": [u.ref for u in disconnected]})
        if terminal.record.get('terminal_product_coverage') == 'unknown':
            issues.append({'reason': 'terminal_product_coverage_unknown', 'step': terminal.ref})
        method_eligible = not issues and all(eligible(u.record, u.reaction) for u in selected)
        for identity, display, basis, product_index in identities:
            if basis == "parent_name" and len(parents[_name(display["name"])]) == 1:
                canonical = next(iter(parents[_name(display["name"])]))
                identity = "structure:" + canonical
                display = dict(display, canonical_smiles=canonical)
            route = groups.setdefault(identity, {"product": display, "identity_basis": basis, "methods": []})
            route["identity_key"] = identity
            method_id = "method_" + hashlib.sha256(json.dumps([identity, terminal.ref, product_index], sort_keys=True).encode()).hexdigest()[:12]
            if any(m["method_id"] == method_id for m in route["methods"]):
                continue
            route["methods"].append({"method_id": method_id,
                "title": terminal.source.get("title") or terminal.context.get("title") or terminal.record["id"],
                "target_step": terminal.ref, "target_product_index": product_index,
                "target_name": candidate["name"] if title_target else products[product_index].get("name"),
                "target_group_basis": "source_title" if title_target else basis,
                "process_kind": terminal.kind, "steps": [u.ref for u in selected], "connections": edges,
                "method_references": list(dict.fromkeys(ref for u in selected for ref in u.method_references)),
                "issues": issues, "connection_status": "needs_review" if issues else "assembled",
                "admission_eligible": method_eligible})
            assigned.update(ancestors)
    routes = []
    for identity, route in groups.items():
        route["route_id"] = "route_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
        routes.append(route)
    for unit in units:
        if unit.order not in assigned:
            reason = ("excluded_process" if unit.kind in EXCLUDED_KINDS else
                      "non_reaction" if unit not in active else "target_or_dependency_unresolved")
            unassigned.append({**unit.ref, "reason": reason})
    return routes, unassigned


# ---------------------------------------------------------------------------
# Patent-scope assembly for the route-repair task (route-v2 candidates).
#
# The v1 assembler above is document-local and only accepts upstream records
# that appear earlier in the source order. Patent volumes can describe the
# preparation of a material after the step that consumes it, and one patent
# can be split into several PDF volumes. This scope assembler indexes the whole
# confirmed patent, resolves material dependencies in both text orders, keeps
# alternative preparations as alternatives (never serialised into one chain),
# and reports completeness separately from quality admission.
# ---------------------------------------------------------------------------

ROUTE_V2_VERSION = "2026-10-27-reference-edges"
ROUTE_SCOPE_VERSION = "patent-scope-v1"

_START_CUES = re.compile(
    r"commercially available|purchased|bought|supplier|catalog(?:ue)?"
    r"|市售|购自|商购|购得|购买|外购", re.I)
_EXTERNAL_DOCUMENT = re.compile(
    r"\b(?:in|of|according to)\s+(?:WO|U\.?S\.?|CN|EP)\s*(?:Patent\s*(?:No\.?\s*)?)?\d{5,}", re.I)
_INTERNAL_OBTAIN = re.compile(
    r"(?:由|从|来自|所得|制得|得到|obe?tained|prepared|produc(?:ed|ing)|from)", re.I)


def _choice_order(candidate):
    producer = candidate[0]
    return (producer.document_id, producer.record["id"], producer.reaction["index"], candidate[2])


def _record_key(unit):
    return unit.document_id, unit.record["id"]


_CRYSTAL_LABEL = re.compile(
    r"\b(?:crystal(?:line)?\s+)?form\s*[-:]?\s*([A-Z0-9]+)\b"
    r"|\bpolymorph\s*[-:]?\s*([A-Z0-9]+)\b|晶型\s*([A-Z0-9ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)"
    r"|([A-Z0-9ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\s*型", re.I)
_MATERIAL_FORM = re.compile(
    r"(?:mono|di|tri|tetra)?hydrochloride|hydrobromide|mesylate|besylate|maleate|fumarate|sulfate|phosphate"
    r"|tartrate|citrate|succinate|(?:mono|di|tri|tetra|hemi)?hydrate|solvate|amorphous|free\s+base"
    r"|盐酸(?:盐)?|氢溴酸(?:盐)?|甲磺酸(?:盐)?|苯磺酸(?:盐)?|马来酸(?:盐)?|富马酸(?:盐)?|硫酸(?:盐)?|磷酸(?:盐)?"
    r"|酒石酸(?:盐)?|柠檬酸(?:盐)?|(?:一|二|三|四|半)?水合物|溶剂合物|无定形|游离碱", re.I)


def _material_form(compound):
    name = str(compound.get("name") or "")
    labels = ["crystal:" + _name(next(v for v in match.groups() if v))
              for match in _CRYSTAL_LABEL.finditer(name)]
    aliases = {"盐酸": "hydrochloride", "盐酸盐": "hydrochloride",
               "monohydrochloride": "hydrochloride", "氢溴酸": "hydrobromide", "氢溴酸盐": "hydrobromide",
               "甲磺酸": "mesylate", "甲磺酸盐": "mesylate", "苯磺酸": "besylate", "苯磺酸盐": "besylate",
               "马来酸": "maleate", "马来酸盐": "maleate", "富马酸": "fumarate", "富马酸盐": "fumarate",
               "硫酸": "sulfate", "硫酸盐": "sulfate", "磷酸": "phosphate", "磷酸盐": "phosphate",
               "酒石酸": "tartrate", "酒石酸盐": "tartrate", "柠檬酸": "citrate", "柠檬酸盐": "citrate",
               "水合物": "hydrate", "一水合物": "monohydrate", "二水合物": "dihydrate",
               "三水合物": "trihydrate", "四水合物": "tetrahydrate", "半水合物": "hemihydrate",
               "无定形": "amorphous", "溶剂合物": "solvate", "游离碱": "freebase"}
    labels.extend(aliases.get(_name(match.group()), _name(match.group()))
                  for match in _MATERIAL_FORM.finditer(name))
    if re.search(r"tablet|capsule|ointment|formulation|片剂|胶囊|软膏|制剂", name, re.I):
        labels.append("formulation:" + _name(name))
    return tuple(sorted(set(labels)))


def _scope_match(product, ingredient):
    # Molecular graphs cannot distinguish polymorphs and may have been resolved
    # to the free parent for a named salt. Require the disclosed form as well.
    if _material_form(product) != _material_form(ingredient):
        return None
    return _match(product, ingredient)


def _scope_units(documents):
    """Flatten (document_id, records) into ordered scope units."""
    units = []
    for document in documents:
        document_id = document["document_id"]
        for record in document.get("records") or []:
            for reaction in record.get("reactions") or []:
                ref = {"id": record["id"], "index": reaction["index"],
                       "step_id": reaction.get("step_id"), "document_id": document_id}
                unit = Unit(record, reaction, len(units), ref, compound_lists(reaction))
                unit.document_id = document_id
                unit.volume_order = document.get("volume_order", 0)
                unit.upstream = []
                unit.boundaries = []
                unit.ambiguities = []
                unit.alternative_materials = []
                unit.preparation_choices = []
                unit.form_mismatches = []
                units.append(unit)
    return units


def _scope_keys(compound):
    keys = []
    canonical = _smiles(compound)
    if canonical:
        stereo = compound.get("stereochemistry")
        suffix = (":stereo:" + json.dumps(stereo, sort_keys=True, ensure_ascii=False)
                  if isinstance(stereo, dict) and stereo else "")
        keys.append("structure:" + canonical + suffix)
        keys.append("structure:" + canonical)
    name = str(compound.get("name") or "").strip()
    normalized = _name(name)
    if normalized and not GENERIC.fullmatch(name):
        keys.append("name:" + normalized)
    return list(dict.fromkeys(keys))


def _producer_identity(candidate):
    producer, _basis, product_index = candidate
    product = (producer.compounds.get("products") or [])[product_index]
    canonical = _smiles(product)
    if canonical:
        return "structure:" + canonical
    normalized = _name(product.get("name"))
    return "name:" + normalized if normalized else "order:" + str(producer.order)


def _active_scope_units(units):
    active = []
    for unit in units:
        if unit.kind in EXCLUDED_KINDS:
            continue
        # A general-procedure template is not a specific preparation and must
        # never provide a material or become a target (same rule as the v1
        # ``eligible``/``details_record_kind`` guard).
        if unit.context.get("record_kind") == "procedure_template":
            continue
        if (unit.reaction.get("status") != "non_reaction" or unit.kind == "formulation"
                or FORM.search(unit.context.get("title", ""))):
            active.append(unit)
    return active


def _producer_index(active):
    index = defaultdict(list)
    for unit in active:
        products = unit.compounds.get("products") or []
        for product_index, product in enumerate(products):
            for key in _scope_keys(product):
                index[key].append((unit, product_index))
        # Reference-as-edge: a reactant literally named after a numbered
        # preparation ("实施例B", "product of Preparation 8") addresses that
        # record's terminal product, which structure/name keys can never
        # reach. Key the terminal step's products by the printed heading id.
        reference_id = unit.source.get("reference_id")
        if reference_id and products and unit.reaction is (unit.record.get("reactions") or [unit.reaction])[-1]:
            for product_index in range(len(products)):
                index.setdefault("ref:" + reference_id, []).append((unit, product_index))
    return index


def _structure_conflict(unit):
    quality = unit.reaction.get("quality") or (unit.reaction.get("validation") or {}).get("quality") or {}
    return quality.get("structure_status") == "conflict"


def _input_candidates(index, unit, ingredient):
    found = {}
    for key in _scope_keys(ingredient):
        for producer, product_index in index.get(key, []):
            if producer is unit:
                continue
            if _structure_conflict(unit):
                continue
            product = (producer.compounds.get("products") or [])[product_index]
            basis = _scope_match(product, ingredient)
            if _structure_conflict(producer) and basis != "canonical_smiles":
                # A conflicted producer still proves this material when the
                # consumer carries the identical structure: the conflict
                # questions the product's other evidence, not the identity
                # both steps print. Name-only matches stay vetoed.
                continue
            if basis:
                found[(producer.order, product_index)] = (producer, basis, product_index)
            elif _match(product, ingredient) and _material_form(product) != _material_form(ingredient):
                unit.form_mismatches.append((ingredient, producer.ref))
    return list(found.values())


_LABELLED_REFERENCE = re.compile(
    r"(?:实施例|Example|中间体|Intermediate|制备例|Preparation|步骤|Step)\s*[\(（]?"
    r"\s*\d+[A-Za-z]?(?:[-–—.]\d+)*", re.I)
# "与制备例 7 同样" / "按照制备例 141-1 记载的方法" are method references, not
# material provenance: the named materials are supplied in the same sentence.
_METHOD_LIKE = re.compile(
    r"与.{0,24}(?:同样|相同)|按照.{0,30}(?:方法|步骤|记载)|参照.{0,30}(?:方法|记载)"
    r"|根据.{0,30}(?:方法|记载)|同样的方法|相同的方法|same manner|in analogy"
    r"|following the (?:procedure|method)|according to|similarly|as described in"
    r"|按照下述|按下述", re.I)


# A numbered reference ("Step 3 of Preparation 9", "Reference Example 55",
# "制备例 24") as written inside a sentence.
_REFERENCE_PHRASE = (
    r"(?:(?:Step|步骤)\s*[\(（]?\s*\d+[A-Za-z]?(?:[-–—.]\d+)*\s*(?:of|in|中)\s+)?"
    r"(?:(?:Reference|参考)\s+)?"
    r"(?:Example|Preparation|Preparative\s+Example|Intermediate|实施例|制备例|中间体)"
    r"\s*[\(（]?\s*\d+[A-Za-z]?(?:[-–—.]\d+)*")
# "<NAME> from Step 3 of Preparation 9" / "<NAME> produced in the preparation
# process of Reference Example 55": the numbered reference follows and directly
# modifies the material name. "used in Step 1" and method usages do not match.
_EN_ATTACHED = re.compile(
    r"\s*,?\s*(?:\([^()]{0,40}\)\s*)?"
    r"(?:which\s+)?(?:was|were|is|are|had\s+been)?\s*"
    r"(?:from\b|obtained\s+(?:from|in|by)|prepared\s+(?:from|in|by)"
    r"|produced\s+(?:in|from|by)|isolated\s+from|made\s+(?:from|in|by)"
    r"|synthesi[sz]ed\s+(?:in|from|by))\s+(?:the\s+)?"
    r"(?:(?:preparation|synthesis|preparative)\s+(?:process|procedure|method)\s+of\s+"
    r"|process\s+of\s+|procedure\s+of\s+|method\s+of\s+)?"
    + _REFERENCE_PHRASE, re.I)
# "在制备例 24 记载的<NAME>中加入" / "由制备例 N 得到的<NAME>" /
# "按照制备例 N 制备的<NAME>": the reference precedes and modifies the name.
_ZH_ATTACHED = re.compile(
    r"(?:在|由|从|按照|根据|依据|如)?\s*(?:上述|前述)?\s*"
    r"(?:制备例|实施例|中间体|步骤|参考实施例)\s*[\(（]?\s*\d+[A-Za-z]?(?:[-–—.]\d+)*"
    r"\s*(?:中|里)?\s*(?:所)?\s*"
    r"(?:记载的|描述的|得到的|制得的|制备的|合成的|获得的|所得)\s*$")
# A material name that is itself a numbered-reference placeholder, such as
# "product of Preparation 8" or "制备例 8 的产物".
_PLACEHOLDER_EN = re.compile(
    r"(?:product|compound|intermediate|material|substance|标题化合物|产物|产品|化合物|中间体)"
    r"\s+(?:of|from|in|obtained\s+from|produced\s+in)\s+(?:the\s+)?"
    r"(?:(?:preparation|synthesis)\s+(?:process|procedure|method)\s+of\s+)?"
    r"(?:Reference\s+)?(?:Example|Preparation|Preparative\s+Example|Step)\s*[\(（]?\s*\d+",
    re.I)
_PLACEHOLDER_ZH = re.compile(
    r"(?:制备例|实施例|中间体|步骤)\s*[\(（]?\s*\d+[A-Za-z]?(?:[-–—.]\d+)*\s*(?:中|里)?\s*"
    r"(?:记载的|所述|所得|得到|制备)?的?\s*(?:产物|产品|化合物|中间体)")


@lru_cache(maxsize=8192)
def _material_name_pattern(name):
    """Whitespace-tolerant literal pattern for a material name (cached)."""
    tokens = [re.escape(token) for token in str(name or "").split()]
    return re.compile(r"\s*".join(tokens), re.I) if tokens else None


def _placeholder_reference_name(name):
    """True when a material name is itself a numbered-reference placeholder."""
    value = str(name or "").strip()
    return bool(value and (_PLACEHOLDER_EN.search(value) or _PLACEHOLDER_ZH.search(value)))


def _direct_reference(sentence, ingredient):
    """True when a numbered reference directly modifies the material name."""
    pattern = _material_name_pattern(ingredient.get("name"))
    if not pattern:
        return False
    for match in pattern.finditer(sentence):
        if _EN_ATTACHED.match(sentence, match.end()):
            return True
        if _ZH_ATTACHED.search(sentence[:match.start()]):
            return True
    return False


# Cross-sentence provenance (R1): "Intermediate X was dissolved in THF. The
# material was obtained from Example 7." The anaphor list is deliberately narrow
# (no bare "it"/"this"/"the formulation"); the anaphor must also sit close to a
# provenance verb so an unrelated "compound"/"product" mention does not count.
_EN_MATERIAL_ANAPHOR = re.compile(
    r"\bthe\s+(?:material|intermediate|compound|product|residue|substance)\b", re.I)
_ZH_MATERIAL_ANAPHOR = re.compile(r"(?:该|此|上述|所述)\s*(?:物质|中间体|化合物|产物|原料)")
_ANAPHOR_OBTAIN = re.compile(
    r"(?:由|从|来自|所得|制得|得到|获得|制备|obe?tained|prepared|produc(?:ed|ing)|from)",
    re.I)
_ANAPHOR_WINDOW = 60


def _material_anaphor(sentence):
    """True for an explicit material anaphor close to a provenance verb."""
    verbs = [match.start() for match in _ANAPHOR_OBTAIN.finditer(sentence)]
    if not verbs:
        return False
    for pattern in (_EN_MATERIAL_ANAPHOR, _ZH_MATERIAL_ANAPHOR):
        for match in pattern.finditer(sentence):
            if any(abs(verb - match.start()) <= _ANAPHOR_WINDOW for verb in verbs):
                return True
    return False


def _reference_sentence(unit, ingredient):
    text = unit.context.get("local_text") or unit.reaction.get("text") or ""
    normalized = _name(ingredient.get("name"))
    if not normalized:
        return None
    placeholder = _placeholder_reference_name(ingredient.get("name"))
    sentences = _split_sentences(text)
    for position, sentence in enumerate(sentences):
        if not _LABELLED_REFERENCE.search(sentence):
            continue
        if normalized in _name(sentence):
            # A reference attached to the material name is provenance in itself
            # (D1/D2): no obtain verb is needed, and a method phrase elsewhere in
            # the same sentence must not veto it.
            if placeholder or _direct_reference(sentence, ingredient):
                return sentence
            # A reference merely mentioned in the sentence is provenance only
            # with an obtain/provenance verb, and never for a method usage
            # ("same manner as Preparative Example 7").
            if _METHOD_LIKE.search(sentence) or METHOD_REFERENCE.search(sentence):
                continue
            if _INTERNAL_OBTAIN.search(sentence):
                return sentence
            continue
        # R1: the previous sentence names the material, and this sentence carries
        # a labelled reference plus an explicit material anaphor.
        if position == 0 or normalized not in _name(sentences[position - 1]):
            continue
        if not _INTERNAL_OBTAIN.search(sentence):
            continue
        if _METHOD_LIKE.search(sentence) or METHOD_REFERENCE.search(sentence):
            continue
        if _material_anaphor(sentence):
            return sentences[position - 1] + " " + sentence
    return None


def _named_reference_keys(value):
    """Reference keys for every labelled reference written inside ``value``."""
    from .step_context import _NAMED_REFERENCE, reference_key
    keys = []
    for match in _NAMED_REFERENCE.finditer(str(value or "")):
        key = reference_key(match)
        if key:
            keys.append(key)
    return keys


TEMPLATE_ROUTE_VERSION = "2026-10-27-reference-route"


def _template_code(compound):
    """The code a scheme-template compound carries: bracketed ("(D)", "胺(H)",
    "compound (9)") or formula-style ("compound of formula V", "formula II-a"
    — the -a suffix is part of the identity, II-a is not II)."""
    from .segmenter import scheme_codes, _SCHEME_CODE, _de_latex
    for value in (compound.get("dict_key"), compound.get("name")):
        text = str(value or "")
        codes = scheme_codes(text)
        if not codes and re.fullmatch(r'[A-Za-z0-9′\'-]+', text):
            codes = scheme_codes('(' + text + ')')
        if codes:
            # An explicit compound label takes precedence over a family name
            # such as "式一化合物(K)". Both identities still use the same parser.
            bracket = [m['bracket'].upper() for m in _SCHEME_CODE.finditer(_de_latex(text))
                       if m['bracket'] and m['bracket'].upper() in codes]
            if bracket:
                return bracket[0]
            return codes[0]
    return None


def _template_targets(unit):
    """Template codes identify generic salts even without a concrete graph."""
    targets = []
    for index, product in enumerate(unit.compounds.get('products') or []):
        code = _template_code(product)
        if code:
            targets.append({'identity': f"template_code:{unit.record['id']}:{code}",
                            'product_index': index, 'display': pick(product, (
                                'name', 'canonical_smiles', 'cxsmiles', 'status', 'reason')),
                            'basis': 'template_code'})
    return targets or _route_targets(unit, None)


def assemble_template_routes(documents):
    """Assemble scheme-template steps into reference-route candidates.

    Template records (record_kind == "procedure_template") describe a general
    synthetic strategy in one narrative; their steps chain by letter-code
    continuity — one step's product code reappears as a later step's reactant
    code. Returned rows mirror assemble_routes_v2 candidates with
    kind="reference_route"; the caller converts them separately."""
    routes = []
    for document in documents:
        document_id = document["document_id"]
        for record in document.get("records") or []:
            template = [r for r in record.get("reactions") or []
                        if (r.get("step_source") or {}).get("record_kind") == "procedure_template"]
            if not template:
                continue
            units = []
            for reaction in sorted(template, key=lambda r: r.get("index") or 0):
                ref = {"id": record["id"], "index": reaction["index"],
                       "step_id": reaction.get("step_id"), "document_id": document_id}
                unit = Unit(record, reaction, len(units), ref, compound_lists(reaction))
                unit.document_id = document_id
                unit.volume_order = document.get("volume_order", 0)
                units.append(unit)
            edges = []
            for later in range(len(units)):
                for role in ("reactants", "participants"):
                    for input_index, ingredient in enumerate(
                            (units[later].compounds.get(role) or [])):
                        code = _template_code(ingredient)
                        if not code:
                            continue
                        for earlier in range(later):
                            for product_index, product in enumerate(
                                    (units[earlier].compounds.get("products") or [])):
                                if _template_code(product) == code:
                                    edges.append({"from": units[earlier].ref,
                                                  "to": units[later].ref,
                                                  "product_index": product_index,
                                                  "input_role": role,
                                                  "input_index": input_index,
                                                  "basis": "template_code_continuity",
                                                  "code": code})
            # The route target is a chain end, not the narrative's last
            # sentence: a scheme often closes with alternative-activation
            # steps whose products are uncoded auxiliaries ("activated
            # ester"), while the real chain ends on a coded compound.
            producers = {edge["from"]["index"] for edge in edges}
            end_units = [unit for unit in units
                         if unit.ref["index"] not in producers]
            coded_ends = [unit for unit in end_units
                          if any(_template_code(product)
                                 for product in (unit.compounds.get("products") or []))]
            for terminal in coded_ends or end_units:
                selected_indices = {terminal.ref['index']}
                pending = [terminal.ref['index']]
                while pending:
                    current = pending.pop()
                    for edge in edges:
                        parent = edge['from']['index']
                        if edge['to']['index'] == current and parent not in selected_indices:
                            selected_indices.add(parent)
                            pending.append(parent)
                connected_indices = set(selected_indices)
                # Retain the existing evidence gaps for uncoded activation
                # auxiliaries accompanying a single main chain. Other coded
                # terminal branches belong only to their own target routes.
                if len(coded_ends) == 1:
                    selected_indices.update(unit.ref['index'] for unit in end_units
                                            if unit not in coded_ends)
                selected = [unit for unit in units if unit.ref['index'] in selected_indices]
                connections = [edge for edge in edges if edge['from']['index'] in selected_indices
                               and edge['to']['index'] in selected_indices]
                gaps = [{"code": "template_step_unconnected", "reaction_ids": [dict(unit.ref)]}
                        for unit in selected if unit.ref['index'] not in connected_indices]
                choices = defaultdict(list)
                for edge in connections:
                    choices[(edge['to']['index'], edge['input_role'], edge['input_index'], edge['code'])].append(edge)
                alternatives = [{'consumer': group[0]['to'], 'input_role': role, 'input_index': index,
                                 'code': code, 'producers': [edge['from'] for edge in group]}
                                for (_consumer, role, index, code), group in choices.items() if len(group) > 1]
                for target in _template_targets(terminal):
                    routes.append({"kind": "reference_route", "target": target,
                                   "terminal": terminal, "selected": selected,
                                   "connections": connections, "boundaries": [], "alternatives": alternatives,
                                   "gaps": gaps,
                                   "completeness": "complete" if not gaps else "incomplete",
                                   "volumes": [document_id]})
    return routes


def _bare_reference_name(name):
    """True when a material name is exactly a numbered reference."""
    from .step_context import _NAMED_REFERENCE
    value = str(name or "").strip()
    return bool(value and _NAMED_REFERENCE.fullmatch(value))


def _ingredient_evidence_sentence(unit, ingredient):
    """First sentence naming the ingredient; the name itself as fallback."""
    name = str(ingredient.get("name") or "").strip()
    text = unit.context.get("local_text") or unit.reaction.get("text") or ""
    for sentence in _split_sentences(text):
        if _name(name) in _name(sentence):
            return sentence
    return name or None


def _ingredient_reference_keys(unit, ingredient):
    """Reference keys a reactant name or its provenance sentence addresses.

    Two shapes count. The name itself may be a bare or placeholder numbered
    reference ("实施例B", "制备例 8 的产物"): the name is the provenance and no
    obtain verb is required. Otherwise a provenance sentence
    (``_reference_sentence``) may attach a labelled reference to an ordinary
    chemical name; those keys are constrained to the chunk's cited references.
    """
    name = str(ingredient.get("name") or "").strip()
    if not name or GENERIC.fullmatch(name):
        return []
    keys = []
    if _bare_reference_name(name) or _placeholder_reference_name(name):
        text = unit.context.get("local_text") or unit.reaction.get("text") or ""
        naming = [sentence for sentence in _split_sentences(text)
                  if _name(name) in _name(sentence)]
        # A pure method citation ("与实施例8...基本相同") is not material
        # provenance; only veto when every naming sentence is method-like.
        if not naming or not all(_METHOD_LIKE.search(sentence) or METHOD_REFERENCE.search(sentence)
                                 for sentence in naming):
            keys.extend(_named_reference_keys(name))
    sentence = _reference_sentence(unit, ingredient)
    if sentence:
        cited = unit.source.get("references") or []
        keys.extend(key for key in _named_reference_keys(sentence)
                    if not cited or key in cited)
    return list(dict.fromkeys(keys))


def _reference_candidates(index, unit, ingredient):
    """Producers addressed by the ingredient's reference keys, if any."""
    found = {}
    for key in _ingredient_reference_keys(unit, ingredient):
        for producer, product_index in index.get("ref:" + key, []):
            if producer is unit or _structure_conflict(unit) or _structure_conflict(producer):
                continue
            found[(producer.order, product_index)] = (producer, "material_reference", product_index)
    return list(found.values())


def _explicit_start_reason(text):
    """Explicit external-document or commercial evidence in the given text."""
    if _EXTERNAL_DOCUMENT.search(text) or re.search(r"(?:WO|US|CN|EP)\s*\d{6,}", text, re.I):
        return "external_document_reference"
    if _START_CUES.search(text):
        return "commercial_material"
    return None


def _start_boundary(unit, ingredient):
    text = unit.context.get("local_text") or unit.reaction.get("text") or ""
    name = ingredient.get("name")
    sentence = _reference_sentence(unit, ingredient)
    if sentence:
        # R7: only a cue inside the same evidence sentence can belong to this
        # material; an unrelated purchase/patent mention elsewhere in the step
        # must not suppress the internal-reference gap.
        explicit = _explicit_start_reason(sentence)
        if explicit:
            return {"reason": explicit, "name": name}
        # A mentioned internal preparation is a real dependency, not a start
        # boundary; not finding it is a gap, not evidence the material is bought.
        return {"reason": "internal_reference_missing", "name": name,
                "evidence_text": sentence}
    if _placeholder_reference_name(name) or _bare_reference_name(name):
        # The material is named after a numbered preparation/example, so its
        # producer exists in the patent and is merely missing from the scope:
        # an internal reference gap, never an undisclosed method. Only the
        # sentence that actually names the placeholder may supply a start cue.
        evidence = next((candidate for candidate in _split_sentences(text)
                         if _name(name) in _name(candidate)), None)
        explicit = _explicit_start_reason(evidence) if evidence else None
        if explicit:
            return {"reason": explicit, "name": name}
        return {"reason": "internal_reference_missing", "name": name,
                "evidence_text": evidence or str(name or "")}
    # No internal reference and no placeholder: the usual start-boundary order.
    explicit = _explicit_start_reason(text)
    if explicit:
        return {"reason": explicit, "name": name}
    return {"reason": "method_not_disclosed_in_scope", "name": name}


def _method_references(unit):
    text = unit.context.get("local_text") or unit.reaction.get("text") or ""
    if METHOD_REFERENCE.search(text):
        return list(dict.fromkeys(unit.source.get("references") or []))
    return []


def _resolve_scope_units(active):
    index = _producer_index(active)
    for unit in active:
        method_references = _method_references(unit)
        inputs = [(role, position, value) for role in ("reactants", "participants")
                  for position, value in enumerate(unit.compounds.get(role, []))]
        for role, position, ingredient in inputs:
            candidates = _input_candidates(index, unit, ingredient)
            if not candidates:
                # Reference-as-edge: a reactant named after a numbered
                # preparation/example resolves to that record's product.
                ref_candidates = _reference_candidates(index, unit, ingredient)
                if ref_candidates:
                    if len({_producer_identity(c) for c in ref_candidates}) > 1:
                        unit.ambiguities.append({"input_role": role, "input_index": position,
                                                 "name": ingredient.get("name"),
                                                 "candidates": [c[0].ref for c in ref_candidates]})
                        continue
                    chosen = sorted(ref_candidates, key=_choice_order)[0]
                    evidence = (_reference_sentence(unit, ingredient)
                                or _ingredient_evidence_sentence(unit, ingredient))
                    unit.upstream.append({"producer": chosen[0], "product_index": chosen[2],
                                          "input_role": role, "input_index": position,
                                          "basis": "material_reference", "evidence": evidence})
                    unit.preparation_choices.append({"edge": unit.upstream[-1],
                                                     "name": ingredient.get("name"),
                                                     "pool": sorted(ref_candidates, key=_choice_order),
                                                     "referenced": ref_candidates})
                    continue
                mismatched = [ref for value, ref in unit.form_mismatches if value is ingredient]
                unit.boundaries.append({"input_role": role, "input_index": position,
                                        **({"reason": "material_form_mismatch", "name": ingredient.get("name")}
                                           if mismatched else _start_boundary(unit, ingredient))})
                continue
            same_record = [c for c in candidates if _record_key(c[0]) == _record_key(unit)]
            referenced = [c for c in candidates if _reference_sentence(unit, ingredient) and
                          _reference_evidence(unit, c[0], ingredient)]
            pool = referenced or same_record or candidates
            if not (referenced or same_record):
                # A method reference to Example N is not material provenance.
                pool = [c for c in pool if c[0].source.get("reference_id") not in method_references]
                if not pool:
                    unit.boundaries.append({"input_role": role, "input_index": position,
                                            **_start_boundary(unit, ingredient)})
                    continue
            if len({_producer_identity(c) for c in pool}) > 1:
                unit.ambiguities.append({"input_role": role, "input_index": position,
                                         "name": ingredient.get("name"),
                                         "candidates": [c[0].ref for c in candidates]})
                continue
            chosen = sorted(pool, key=_choice_order)[0]
            chosen_basis = "material_reference" if chosen in referenced else chosen[1]
            evidence = _reference_sentence(unit, ingredient) if chosen in referenced else None
            unit.upstream.append({"producer": chosen[0], "product_index": chosen[2],
                                  "input_role": role, "input_index": position,
                                  "basis": chosen_basis, "evidence": evidence})
            unit.preparation_choices.append({"edge": unit.upstream[-1], "name": ingredient.get("name"),
                                             "pool": sorted(pool, key=_choice_order),
                                             "referenced": referenced})


def _choose_preparations(active, siblings, terminal_steps):
    """Prefer complete producer closures; never enumerate combinations."""
    visiting, done = set(), set()

    def assessment(unit):
        selected = _closure(unit)
        status, gaps = _assess_route(unit, selected, siblings, terminal_steps)
        cycle = any(value.order in visiting for value in selected)
        return (0 if status == "complete" and not cycle else
                1 if status == "unconfirmed" and not cycle else 2, len(gaps))

    def visit(unit):
        if unit.order in done or unit.order in visiting:
            return
        visiting.add(unit.order)
        for choice in unit.preparation_choices:
            for producer, _, _ in choice["pool"]:
                visit(producer)
            chosen = min(choice["pool"], key=lambda c: (*assessment(c[0]), _choice_order(c)))
            choice["edge"].update(producer=chosen[0], product_index=chosen[2],
                                  basis="material_reference" if chosen in choice["referenced"] else chosen[1])
        visiting.remove(unit.order)
        done.add(unit.order)

    for unit in sorted(active, key=lambda u: (u.document_id, u.record["id"], u.reaction["index"])):
        visit(unit)
    _break_cycles(active)
    for unit in active:
        for choice in unit.preparation_choices:
            edge = choice["edge"]
            if edge not in unit.upstream:
                continue
            others = [c for c in choice["pool"] if c[0] is not edge["producer"]
                      and assessment(c[0])[0] == 0]
            if others:
                unit.alternative_materials.append({
                    "input_role": edge["input_role"], "input_index": edge["input_index"],
                    "name": choice["name"], "chosen": edge["producer"].ref,
                    "alternatives": [dict(c[0].ref, product_index=c[2], basis=c[1]) for c in others]})


def _break_cycles(active):
    by_order = {unit.order: unit for unit in active}
    producers = {unit.order: [edge["producer"].order for edge in unit.upstream] for unit in active}
    removed = []
    colors = {}

    def visit(order, trail):
        if colors.get(order) in (1, 2):
            return
        colors[order] = 1
        for producer_order in list(producers[order]):
            if producer_order in trail:
                removed.append((order, producer_order))
                producers[order].remove(producer_order)
                continue
            visit(producer_order, trail | {order})
        colors[order] = 2

    for unit in active:
        visit(unit.order, set())
    for consumer_order, producer_order in removed:
        unit = by_order[consumer_order]
        unit.upstream = [edge for edge in unit.upstream
                         if edge["producer"].order != producer_order]
        unit.ambiguities.append({"reason": "dependency_cycle",
                                 "candidates": [by_order[producer_order].ref]})
    return removed


def _closure(terminal):
    """Material dependency closure in execution order (upstream, ..., target).

    Source order is deliberately not used: a patent may describe the
    preparation of a starting material after the step that consumes it.
    """
    ordered, visited = [], set()

    def visit(unit):
        if unit.order in visited:
            return
        visited.add(unit.order)
        for edge in unit.upstream:
            visit(edge["producer"])
        ordered.append(unit)

    visit(terminal)
    return ordered


def _conflict_topology_confirmed(unit, selected):
    """A conflicted step still proves route topology when every selected
    downstream consumer carries its product's identical structure."""
    edges = [edge for consumer in selected
             for edge in consumer.upstream if edge["producer"] is unit]
    return bool(edges) and all(edge["basis"] == "canonical_smiles" for edge in edges)


def _route_targets(terminal, final_product):
    identities = []
    for product_index, product in enumerate(terminal.compounds.get("products") or []):
        identity = _identity(product, terminal, final_product)
        if not identity:
            continue
        key = identity[0] + _stereo_identity(terminal, product)
        form = _material_form(product)
        display = identity[1]
        if form:
            key += ":form:" + json.dumps(form, ensure_ascii=False)
            display = {**display, **pick(product, ("name", "canonical_smiles", "stereochemistry"))}
        identities.append({"identity": key, "product_index": product_index,
                           "display": display, "basis": identity[2]})
    return identities


def _connection_rows(selected):
    selected_orders = {unit.order for unit in selected}
    rows = []
    for unit in selected:
        for edge in unit.upstream:
            producer = edge["producer"]
            if producer.order not in selected_orders:
                continue
            row = {"from_ref": producer.ref, "to_ref": unit.ref,
                   "product_index": edge["product_index"],
                   "input_role": edge["input_role"], "input_index": edge["input_index"],
                   "basis": edge["basis"]}
            if edge.get("evidence"):
                row["evidence_text"] = edge["evidence"]
            rows.append(row)
    return rows


def _assess_route(terminal, selected, siblings_by_record, terminal_steps):
    gaps = []
    incomplete = False
    unconfirmed = False
    for unit in selected:
        for boundary in unit.boundaries:
            if boundary.get("reason") == "material_form_mismatch":
                gaps.append({"code": "material_form_mismatch", "reaction_ids": [dict(unit.ref)],
                             "material": boundary.get("name")})
                incomplete = True
            if boundary.get("reason") == "internal_reference_missing":
                gaps.append({"code": "missing_internal_material",
                             "reaction_ids": [dict(unit.ref)],
                             "material": boundary.get("name"),
                             "evidence_text": boundary.get("evidence_text")})
                incomplete = True
        for ambiguity in unit.ambiguities:
            gaps.append({"code": "ambiguous_upstream", "reaction_ids": [dict(unit.ref)],
                         "material": ambiguity.get("name"),
                         "candidate_reaction_ids": ambiguity.get("candidates") or []})
            incomplete = True
        for reference in unit.context.get("procedure_references") or []:
            if reference.get("relation") == "precursor_preparation":
                continue
            if reference.get("status") != "resolved":
                gaps.append({"code": "procedure_reference_" + str(reference.get("status", "missing")),
                             "reaction_ids": [dict(unit.ref)],
                             "reference_id": reference.get("id")})
                incomplete = True
        if unit.context.get("unit_kind") == "telescoped":
            gaps.append({"code": "telescoped_without_reported_intermediates",
                         "reaction_ids": [dict(unit.ref)]})
    # Every selected preparation contributes its own coverage obligations,
    # including upstream methods with disconnected or unconfirmed steps.
    last_selected = {}
    for unit in selected:
        key = _record_key(unit)
        if key not in last_selected or last_selected[key].order < unit.order:
            last_selected[key] = unit
    for method in last_selected.values():
        if _record_key(method) not in terminal_steps:
            continue
        siblings = siblings_by_record.get(_record_key(method), [])
        selected_orders = {unit.order for unit in selected}
        missing = [unit for unit in siblings
                   if unit.order not in selected_orders and unit.order < method.order]
        if missing:
            gaps.append({"code": "disconnected_procedure_steps",
                         "reaction_ids": [dict(unit.ref) for unit in missing]})
            incomplete = True
        unlinked = [unit for unit in siblings
                    if unit.order in selected_orders and unit.order != method.order
                    and not unit.upstream and not unit.boundaries]
        if unlinked and len(siblings) > 1:
            gaps.append({"code": "unconnected_procedure_steps",
                         "reaction_ids": [dict(unit.ref) for unit in unlinked]})
            incomplete = True
    for method in last_selected.values():
        if method.record.get("terminal_product_coverage") == "unknown":
            # A bare paragraph marker ("[0231]") is not a product title, so
            # the coverage flag below was set by a check that could not be
            # asked; records delivered before that rule carry the vacuous
            # flag and must not unconfirm a route.
            title = str((method.record.get("source") or {}).get("title") or "").strip()
            if re.fullmatch(r"\[\d{1,4}\]", title):
                continue
            gaps.append({"code": "terminal_product_coverage_unknown",
                         "reaction_ids": [dict(method.ref)]})
            unconfirmed = True
    if not terminal.compounds.get("products"):
        gaps.append({"code": "extraction_product_missing", "reaction_ids": [dict(terminal.ref)]})
        unconfirmed = True
    for unit in selected:
        if _structure_conflict(unit):
            if _conflict_topology_confirmed(unit, selected):
                # Both sides print the identical structure, so the conflict
                # questions other evidence for that product, not this route's
                # step-to-step linkage. It stays on the reaction's quality and
                # the admission gate; it must not break route completeness.
                continue
            gaps.append({"code": "structure_conflict",
                         "reaction_ids": [dict(unit.ref)]})
            incomplete = True
    status = "complete" if not gaps else ("incomplete" if incomplete else "unconfirmed")
    return status, gaps


def assemble_routes_v2(documents, final_products=None):
    """Assemble route candidates across a confirmed patent scope.

    ``documents`` is a list of dicts with ``document_id``, ``volume_order`` and
    ``records`` (the compact delivery records of that volume). Returns
    ``(routes, fragments)`` as internal candidate rows; the caller converts
    compound slots to reaction-v1 business identifiers and writes the files.
    """
    final_products = final_products or {}
    units = _scope_units(documents)
    active = _active_scope_units(units)
    siblings_by_record = defaultdict(list)
    for unit in units:
        siblings_by_record[_record_key(unit)].append(unit)
    _resolve_scope_units(active)
    terminal_steps = {}
    for document in documents:
        for record in document.get("records") or []:
            if record.get("terminal_product_coverage") in {"supported", "unknown"}:
                terminal_steps[(document["document_id"], record["id"])] = (record.get("reactions") or [{}])[-1].get("index")
    _choose_preparations(active, siblings_by_record, terminal_steps)
    consumed = {edge["producer"].order for unit in active for edge in unit.upstream}
    routes, fragments = [], []
    for terminal in active:
        if terminal.order in consumed:
            continue
        if (_record_key(terminal) in terminal_steps
                and terminal.reaction.get("index") != terminal_steps[_record_key(terminal)]):
            continue
        final_product = final_products.get(terminal.document_id)
        identities = _route_targets(terminal, final_product)
        title_target = None
        if not identities:
            from .step_context import _product_heading
            source_title = terminal.source.get("title") or ""
            title = (source_title if _product_heading(source_title)
                     else terminal.context.get("title") or source_title)
            if title and not GENERIC.fullmatch(title):
                candidate = {"name": re.sub(r"[:：]\s*$", "", title)}
                identity = _identity(candidate, terminal, final_product)
                if identity:
                    title_target = {"identity": identity[0], "product_index": None,
                                    "display": identity[1], "basis": identity[2]}
                    if _material_form(candidate):
                        title_target["identity"] += ":form:" + json.dumps(_material_form(candidate), ensure_ascii=False)
                        title_target["display"] = {"name": title}
            if title_target is None:
                title_target = {"identity": "unresolved:" + terminal.document_id + "::" + terminal.record["id"],
                                "product_index": None, "display": {}, "basis": "unresolved_target"}
        targets = identities or ([title_target] if title_target else [])
        if not targets:
            continue
        selected = _closure(terminal)
        status, gaps = _assess_route(terminal, selected, siblings_by_record, terminal_steps)
        connections = _connection_rows(selected)
        boundaries = []
        for unit in selected:
            for boundary in unit.boundaries:
                row = {"reaction_id": dict(unit.ref), "name": boundary.get("name"),
                       "reason": boundary.get("reason")}
                if boundary.get("evidence_text"):
                    row["evidence_text"] = boundary["evidence_text"]
                boundaries.append(row)
        alternatives = []
        for unit in selected:
            for alternative in unit.alternative_materials:
                alternatives.append({"reaction_id": dict(unit.ref),
                                     "material": alternative.get("name"),
                                     "chosen": alternative.get("chosen"),
                                     "alternatives": alternative.get("alternatives")})
        for target in targets:
            row = {"target": target, "terminal": terminal, "selected": selected,
                   "connections": connections, "boundaries": boundaries,
                   "alternatives": alternatives, "gaps": gaps,
                   "completeness": status, "title_target": title_target is not None,
                   "volumes": list(dict.fromkeys(unit.document_id for unit in selected))}
            (routes if status == "complete" else fragments).append(row)
    return routes, fragments
