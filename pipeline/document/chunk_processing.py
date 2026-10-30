"""One-chunk extraction contract shared by batch and standalone runners."""

import re
from copy import deepcopy
from .step_context import split_steps, extraction_text
from pipeline.validation.quality import source_metadata, evaluate_quality, _norm, product_title_norm, product_title_matches
from pipeline.chemical_quality import valid_image_smiles
from typing import Any, Dict, List


MAX_INPUT_CHARS = 3500
COMPLETED_STATUSES = {"ok", "needs_review", "non_reaction"}


def extraction_variant(step):
    return 'scheme_template' if step.get('record_kind') == 'procedure_template' else None


def split_chunk_text(chunk: Dict[str, Any]) -> List[str]:
    """Split long procedures at MinerU block boundaries without dropping text."""
    text = chunk["text"]
    if len(text) <= MAX_INPUT_CHARS:
        return [text]
    parts = [str(part).strip() for part in chunk.get("text_parts", []) if str(part).strip()]
    if not parts:
        parts = [text]
    elif chunk.get("title"):
        parts.insert(0, str(chunk["title"]).strip())
    segments: List[str] = []
    pending = ""
    for part in parts:
        while len(part) > MAX_INPUT_CHARS:
            cut = max(part.rfind("。", 1600, MAX_INPUT_CHARS),
                      part.rfind(". ", 1600, MAX_INPUT_CHARS),
                      part.rfind(" ", 1600, MAX_INPUT_CHARS))
            cut = cut + 1 if cut >= 1600 else MAX_INPUT_CHARS
            if pending:
                segments.append(pending)
                pending = ""
            segments.append(part[:cut].strip())
            part = part[cut:].strip()
        if pending and len(pending) + len(part) + 1 > MAX_INPUT_CHARS:
            segments.append(pending)
            pending = ""
        pending = (pending + " " + part).strip()
    if pending:
        segments.append(pending)
    return segments


def _has_ner(ner: Dict[str, Any]) -> bool:
    return any(bool(value) for key, value in (ner or {}).items()
               if key in ("REACTION_PRODUCT", "STARTING_MATERIAL", "REAGENT_CATALYST",
                          "SOLVENT", "TEMPERATURE", "TIME", "YIELD_PERCENT"))


def canonical_image_smiles(value):
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.MolFromSmiles(value))


def backfill_image_label_compounds(coref, image_evidence):
    """Deterministic fallback when the coref mapping misses a label the
    compound name itself carries ("式一化合物(K)" → dict key "(K)").

    Mirrors coref's ``_apply_mapping`` fields with a distinct trace service
    so the provenance of the backfill stays visible."""
    from .image_labels import carries_image_label, image_scope_labels
    labels = image_scope_labels(image_evidence)
    if not labels:
        return
    local_dict = (image_evidence or {}).get("dict_result") or {}
    for field in ("REACTION_PRODUCT", "STARTING_MATERIAL"):
        for comp in (coref.get("ner") or {}).get(field) or []:
            if not isinstance(comp, dict) or comp.get("smiles"):
                continue
            if not carries_image_label(comp.get("name"), labels):
                continue
            for key, value in local_dict.items():
                if not carries_image_label(comp.get("name"), {key}):
                    continue
                if not valid_image_smiles(value, allow_markush=True):
                    continue
                comp["smiles"] = value
                comp["dict_key"] = key
                comp["structure_source"] = "image_identifier"
                comp["resolution_trace"] = [*(comp.get("resolution_trace") or []),
                                            dict(service="image_label_backfill",
                                                 query=comp.get("name"), status="resolved")]
                break


def bind_scheme_antecedents(records, text):
    """Bind functional-group shorthand only to an explicit preceding product."""
    from .routes import _split_sentences, _template_code
    from .segmenter import _de_latex, scheme_codes
    records, repairs = deepcopy(records), {}
    sentences = _split_sentences(_de_latex(text))
    generic = re.compile(r'^(?:硝基苯|硝基芳烃|苯胺|芳胺|nitroarene|nitrobenzene|aniline)$', re.I)
    product_cue = re.compile(r'得到|生成|制得|形成|转化成|转化为|产品|产物|\b(?:give|giving|afford(?:ed)?|yield(?:ed)?|form(?:ed)?)\b', re.I)
    def carries(sentence, code):
        return bool(re.search(r'(?<![A-Za-z0-9])' + re.escape(code) + r'(?![A-Za-z0-9-])', sentence, re.I))
    for number, ner in enumerate(records):
        outputs = {_template_code({'name': p}) for p in ner.get('REACTION_PRODUCT') or [] if isinstance(p, str)} - {None}
        if len(outputs) != 1:
            continue
        code = next(iter(outputs))
        for index, name in enumerate(ner.get('STARTING_MATERIAL') or []):
            if not isinstance(name, str) or not generic.fullmatch(name.strip()):
                continue
            positions = [i for i, sentence in enumerate(sentences) if carries(sentence, code) and name in sentence]
            if len(positions) != 1 or not positions[0]:
                continue
            previous = sentences[positions[0] - 1]
            cues = list(product_cue.finditer(previous))
            if not cues:
                continue
            tail = previous[cues[-1].end():]
            reported = set(scheme_codes('产物' + tail))
            if len(reported) != 1:
                continue
            candidates = {_template_code({'name': p}) for prior in records[:number]
                          for p in prior.get('REACTION_PRODUCT') or [] if isinstance(p, str)} - {None, code}
            candidates = {candidate for candidate in candidates if candidate in reported and carries(tail, candidate)}
            if len(candidates) != 1:
                continue
            antecedent = next(iter(candidates))
            ner['STARTING_MATERIAL'][index] = antecedent
            repairs.setdefault(number, []).append({'field': 'STARTING_MATERIAL', 'index': index,
                'from': name, 'to': antecedent, 'basis': 'unique_preceding_scheme_product',
                'source_text': previous + ' ' + sentences[positions[0]]})
    return records, repairs


def ground_scheme_compounds(coref, image_evidence):
    """A scheme label is not a chemical formula or a literal functional group."""
    from .routes import _template_code
    local = (image_evidence or {}).get('dict_result') or {}
    for field in ('STARTING_MATERIAL', 'REACTION_PRODUCT'):
        for compound in (coref.get('ner') or {}).get(field) or []:
            if not isinstance(compound, dict):
                continue
            code = _template_code({'name': compound.get('name')})
            if not code:
                continue
            values = {smiles for label, smiles in local.items()
                      if _template_code({'name': label}) == code
                      and valid_image_smiles(smiles, allow_markush=True)}
            mapping = coref.setdefault('id_smiles_mapping', {})
            mapping.pop(compound.get('name'), None)
            for key in ('smiles', 'canonical_smiles', 'cxsmiles', 'standard_name', 'stereochemistry',
                        'dict_key', 'image_source', 'resolution_trace', 'atom_mapping'):
                compound.pop(key, None)
            compound.update(identifier_only=True, resolution_status='identifier_only',
                            structure_source='scheme_identifier', status='structure_unavailable',
                            reason='unresolved_structure')
            if len(values) == 1:
                compound.update(smiles=next(iter(values)), dict_key=code, structure_source='image_identifier')
                mapping[compound['name']] = compound['smiles']
            unresolved = coref.setdefault('unresolved_identifiers', [])
            if compound['name'] not in mapping and compound['name'] not in unresolved:
                unresolved.append(compound['name'])
    mapping = coref.get('id_smiles_mapping') or {}
    unresolved = [name for name in coref.get('unresolved_identifiers') or [] if name not in mapping]
    coref['unresolved_identifiers'] = unresolved
    if isinstance(coref.get('summary'), dict):
        coref['summary'].update(total_identifiers=len(set(mapping) | set(unresolved)),
                                resolved=len(mapping), unresolved=len(unresolved))


def _named_image_edge(step, coref, image):
    """Find a unique scheme edge anchored by a locally named, labelled precursor."""
    if step.get('boundary_status') != 'source_grounded' or len(image.get('image_files', [])) != 1:
        return None
    ner = coref.get('ner') or {}
    reactants = ner.get('STARTING_MATERIAL') or []
    if len(reactants) != 1 or not isinstance(reactants[0], dict):
        return None
    precursor = reactants[0]
    key = precursor.get('dict_key')
    if (precursor.get('structure_source') != 'image_identifier' or not key
            or image.get('dict_result', {}).get(key) != precursor.get('smiles')
            or not valid_image_smiles(precursor.get('smiles'))):
        return None
    if not re.search(r'(?<![A-Za-z0-9])' + re.escape(str(key)) + r'(?![A-Za-z0-9])',
                     step.get('local_text', ''), re.I):
        return None
    from pathlib import Path
    filename = Path(str(image['image_files'][0])).name
    if not any(entry.get('image_file_name') == filename
               and entry.get('dict_result', {}).get(key) == precursor['smiles']
               for entry in image.get('per_image_detections') or []):
        return None
    edges = [dict(reaction, reaction_index=i, image_file=entry['image_file_name'])
             for entry in image.get('image_result', [])
             if entry.get('image_file_name') == filename
             for i, reaction in enumerate(entry.get('reactions', []))
             if valid_image_smiles(reaction.get('reactants'))
             and valid_image_smiles(reaction.get('products'))
             and canonical_image_smiles(reaction['reactants']) == canonical_image_smiles(precursor['smiles'])]
    if len({canonical_image_smiles(e['products']) for e in edges}) != 1:
        return None
    return dict(edges[0], precursor_identifier=key)


def process_chunk(pipe: Any, chunk: Dict[str, Any], language: str,
                  merged_dict: Dict[str, str],
                  image_evidence: Dict[str, Any] = None, reuse=None) -> Dict[str, Any]:
    base = {
        "id": chunk["id"], "source": chunk, "reactions": [],
        "position": {key: chunk["position"][key]
                     for key in ("page_idx", "bbox", "match_method", "pages")},
    }
    kind = chunk.get("process_kind", "synthesis")
    if chunk.get('kind') in ('fallback', 'scheme'):
        from .segmenter import has_procedure_operations, is_scheme_description
        chunk_text = chunk.get('procedure') or chunk.get('text') or ''
        if not has_procedure_operations(chunk_text) and not is_scheme_description(chunk_text):
            return dict(base, status='non_reaction', reason='non_procedure_text',
                        extraction=None, coref=None, validation=None)
    if kind in ("assay", "analysis", "purification", "formulation"):
        return dict(base, status="non_reaction", reason=kind,
                    analysis_conditions={"source_text": chunk.get("procedure") or chunk["text"],
                                         "block_indices": chunk.get("block_indices", []),
                                         "condition_type": kind}
                    if kind in ("assay", "analysis", "formulation") else None,
                    extraction=None, coref=None, validation=None)

    reactions = []
    steps = chunk.get("steps") or split_steps(chunk)
    if any(s.get("needs_step_planning") for s in steps):
        from .step_context import refine_steps
        steps = refine_steps(chunk, steps, pipe._get_extraction())
    for segment_index, step in enumerate(steps, 1):
        step_kind = step.get('process_kind') or kind
        if step_kind in ('assay', 'analysis', 'purification', 'formulation'):
            base.setdefault('non_reaction_units', []).append({'step_id': step['step_id'],
                'process_kind': step_kind, 'source_parts': deepcopy(step['source_parts']),
                'source_text': step['text']})
            continue
        if step_kind == 'unverified':
            step_kind = kind
        text = extraction_text(step)
        # A general-scheme paragraph describes several transformations in one
        # narrative: the scheme_template prompt returns one record per
        # transformation, with the action verb inside each record.
        variant = extraction_variant(step)
        from pipeline.extraction.inference import ContextLengthError
        try:
            if variant:
                extraction = pipe._get_extraction().extract(
                    text, language, task='ner', prompt_variant=variant)
            elif reuse:
                extraction = reuse.extract(pipe._get_extraction(), step, language)
            else:
                extraction = pipe._get_extraction().extract(text, language)
        except ContextLengthError as exc:
            failed = dict(language=language, ner={}, action=[], raw={},
                          parse_status={'ner': 'context_length_exceeded'},
                          **source_metadata({}, step['local_text'], step))
            quality = evaluate_quality(failed, 'needs_review')
            quality['blocking_issues'].append('context_length_exceeded')
            reactions.append(dict(step_id=step['step_id'], step_source=deepcopy(step),
                quality=quality, index=len(reactions)+1, segment_index=segment_index,
                record_index=1, text=text, status='needs_review', reason='context_length_exceeded',
                error=str(exc), extraction=failed, coref=None, validation=None))
            continue
        raw = extraction.get("raw", {})
        parsed = extraction.get("ner_records")
        if parsed is None:
            parsed = [extraction.get("ner", {})]
        if not parsed:
            parsed = [{}]
        scheme_repairs = {}
        if variant:
            parsed, scheme_repairs = bind_scheme_antecedents(parsed, step['local_text'])
        for record_index, ner in enumerate(parsed, 1):
            ner = deepcopy(ner)
            from .source_repairs import repair_local_fields, repair_nitrile_structures
            field_repairs = []
            if len(parsed) == 1:
                ner, field_repairs = repair_local_fields(ner, step['local_text'])
            # Bind deictic product mentions before the name-resolution cascade.
            if step.get("title") and step.get("boundary_status") == "source_grounded":
                ner["REACTION_PRODUCT"] = [step["title"] if isinstance(p, str) and re.fullmatch(
                    r"(?:标题化合物|标题产物|目标产物|title compound|target product|the title compound)", p.strip(), re.I)
                    else p for p in ner.get("REACTION_PRODUCT", [])]
            item_extraction = dict(extraction, ner=ner)
            field_repairs.extend(scheme_repairs.get(record_index - 1, []))
            if field_repairs:
                item_extraction['source_field_repairs'] = field_repairs
            item_extraction.pop("ner_records", None)
            if variant:
                # Scheme-template records carry their own action verb and are
                # step-aligned by construction (one record per transformation).
                step_action = ner.pop("action", None)
                if isinstance(step_action, list):
                    item_extraction["action"] = [str(value).strip() for value in step_action if value]
                elif isinstance(step_action, str) and step_action.strip():
                    item_extraction["action"] = [step_action.strip()]
                else:
                    item_extraction["action"] = []
                item_extraction['step_order'] = ner.pop("step_order", None)
                item_extraction["scheme_template"] = True
            elif len(parsed) > 1:
                item_extraction["action"] = []
            parse_status = extraction.get("parse_status") or {}
            invalid_ner = parse_status.get("ner") not in (None, "ok")
            invalid_action = parse_status.get("action") not in (None, "ok")
            if invalid_ner or not _has_ner(ner):
                failed_record = dict(item_extraction, **source_metadata(ner, step["local_text"], step,
                                                                         item_extraction.get("action")))
                quality = evaluate_quality(failed_record, "needs_review")
                quality["blocking_issues"].append("invalid_or_empty_ner")
                reactions.append({
                    "step_id": step["step_id"], "step_source": deepcopy(step), "quality": quality,
                    "index": len(reactions) + 1, "segment_index": segment_index,
                    "record_index": record_index, "text": text,
                    "status": "needs_review", "reason": "invalid_or_empty_ner",
                    "extraction": item_extraction, "coref": None, "validation": None,
                })
                continue
            metadata = source_metadata(ner, step["local_text"], step, item_extraction.get("action"))
            payload = dict(item_extraction, **metadata, text=text,
                           ms_source_text=step["local_text"], dict_result=merged_dict,
                           process_kind=step_kind, record_kind=step.get('record_kind', 'reaction_candidate'))
            if reuse:
                payload['ner'] = reuse.enrich_known_names(ner, step, language, record_index,
                                                          image_evidence=image_evidence)
            coref = pipe._get_coref().resolve(payload)
            from .source_conflicts import reject_source_conflicts
            reject_source_conflicts(coref, chunk.get('source_compound_evidence') or [])
            if coref.get("standardize_status") == "failed":
                detail = coref.get("standardize_error") or "see coref logs"
                raise RuntimeError(f"compound standardization failed: {detail}")
            repair_nitrile_structures(coref.get('ner') or {}, step['local_text'])
            coref["process_kind"] = step_kind
            # Third-party/test resolvers may only return the compound record.
            for key, value in payload.items():
                if key not in coref and key not in ("dict_result", "ner"):
                    coref[key] = deepcopy(value)
            backfill_image_label_compounds(coref, image_evidence)
            if variant:
                ground_scheme_compounds(coref, image_evidence)
            for product in (coref.get("ner") or {}).get("REACTION_PRODUCT") or []:
                if isinstance(product, dict) and product.get("structure_source") == "image_identifier":
                    key = product.get('dict_key')
                    locally_named = (step.get('boundary_status') == 'source_grounded' and step.get('title')
                        and _norm(product.get('name')) == product_title_norm(step['title'])
                        and product.get('smiles') == (image_evidence or {}).get('dict_result', {}).get(key)
                        and any(e.get('dict_result', {}).get(key) == product.get('smiles')
                                for e in (image_evidence or {}).get('per_image_detections') or []))
                    if key in (image_evidence or {}).get("dict_result", {}) and (len(steps) == 1 or locally_named):
                        product["image_source"] = {"images": (image_evidence or {}).get("image_files", []), "dict_key": product["dict_key"]}
                    else:
                        product["image_scope_unverified"] = True
            image_products = {
                product for item in (image_evidence or {}).get("image_result", [])
                for reaction in item.get("reactions", [])
                for product in str(reaction.get("products") or "").split(";")
                if valid_image_smiles(product)
            }
            edge = _named_image_edge(step, coref, image_evidence or {})
            if edge:
                for product in (coref.get('ner') or {}).get('REACTION_PRODUCT') or []:
                    if not isinstance(product, dict) or not product.get('identifier_only'):
                        continue
                    name = product.get('name', '').strip()
                    named_target = re.escape(name).replace(r'\ ', r'\s+')
                    if not name or not re.search(r'(?:preparation|synthesis)\s+of\s+' + named_target +
                                                r'(?:\s*[:：]|\b)', step.get('local_text', ''), re.I):
                        continue
                    if valid_image_smiles(product.get('smiles')) and canonical_image_smiles(product['smiles']) != canonical_image_smiles(edge['products']):
                        product['structure_source'] = 'adjacent_image_conflict'
                        product['image_product_candidates'] = [edge['products']]
                    else:
                        product['smiles'] = edge['products']
                        product['structure_source'] = 'adjacent_image'
                        product['image_source'] = {'images': (image_evidence or {})['image_files'],
                                                   'source_reaction_edge': edge}
            standalone = [item for item in (image_evidence or {}).get('per_image_detections', [])
                          if len(item.get('molecule_boxes') or []) == 1
                          and not item.get('label_mol_links')]
            # MinerU places an unnumbered molecule caption in image evidence,
            # even when the Example heading supplies no chemical title.
            captions = {text.strip() for text in [step.get('title', ''), *[
                part.get('text', '') for part in step.get('source_parts', [])
                if part.get('source_role') == 'image_annotation']]
                if re.fullmatch(r'\s*\[?Chemical formula\s+\d+\]?\s*', text, re.I)}
            caption = next(iter(captions)) if len(captions) == 1 else None
            standalone_products = {item['molecule_boxes'][0].get('smiles') for item in standalone
                                   if valid_image_smiles(item['molecule_boxes'][0].get('smiles'))}
            # Only a single explicitly captioned structure can be associated
            # without an identifier. Never choose a molecule from a scheme.
            if (caption and len(steps) == 1
                    and len((image_evidence or {}).get('image_files', [])) == 1
                    and len(standalone_products) == 1):
                for product in (coref.get('ner') or {}).get('REACTION_PRODUCT') or []:
                    if isinstance(product, dict) and product.get('name', '').strip('[] ') == caption.strip('[] '):
                        image_smiles = next(iter(standalone_products))
                        if valid_image_smiles(product.get('smiles')):
                            from rdkit import Chem
                            if Chem.MolToSmiles(Chem.MolFromSmiles(product['smiles'])) != Chem.MolToSmiles(Chem.MolFromSmiles(image_smiles)):
                                product['structure_source'] = 'adjacent_image_conflict'
                                product['image_product_candidates'] = [image_smiles]
                                continue
                        product['smiles'] = image_smiles
                        product['structure_source'] = 'adjacent_image'
                        product['image_source'] = {'images': (image_evidence or {})['image_files'],
                                                   'caption': caption}
            matches = [
                product for product in (coref.get("ner") or {}).get("REACTION_PRODUCT") or []
                if isinstance(product, dict) and re.search(
                    r"目标产物|target product|title compound", product.get("name", ""), re.I)]
            # Chunk-adjacent images cannot be assigned to one of several steps.
            if len(image_products) == 1 and len(steps) == 1 and matches:
                product_smiles = next(iter(image_products))
                for product in matches:
                    product["smiles"] = product_smiles
                    product["structure_source"] = "adjacent_image"
            elif len(image_products) > 1:
                # Ambiguous evidence: mark the conflict instead of guessing.
                for product in matches:
                    product.setdefault("structure_source", "adjacent_image_conflict")
                    product.setdefault("image_product_candidates", sorted(image_products))
            if len(steps) == 1 and len(image_products) == 1:
                from rdkit import Chem
                image_mol = Chem.MolFromSmiles(next(iter(image_products)))
                for product in (coref.get("ner") or {}).get("REACTION_PRODUCT") or []:
                    if not isinstance(product, dict) or not product.get("smiles"):
                        continue
                    mol = Chem.MolFromSmiles(product["smiles"])
                    if mol and image_mol and Chem.MolToSmiles(mol, isomericSmiles=False) != Chem.MolToSmiles(image_mol, isomericSmiles=False):
                        product["structure_source"] = "adjacent_image_conflict"
                        product["image_product_candidates"] = sorted(image_products)
            product_entries = (coref.get('ner') or {}).get('REACTION_PRODUCT') or []
            if len(product_entries) == 1 and isinstance(product_entries[0], dict):
                from pipeline.validation.normalize import repair_elemental_form
                repair_elemental_form(product_entries[0], step['local_text'])
            validation = pipe._get_validation().validate_record(coref)
            rxn = validation.get("rxn_validation", {})
            rxn_status = rxn.get("status")
            status = ("non_reaction" if rxn_status == "non_reaction" and step_kind != "formulation"
                      else "needs_review" if (step_kind == "formulation" or invalid_action or
                      rxn_status not in ("passed", "corrected")) else "ok")
            if len(parsed) > 1 and not variant:
                status = "needs_review"
            quality_record = dict(coref, **{k: v for k, v in validation.items() if k not in ("ner", "action")})
            quality = evaluate_quality(quality_record, status)
            if (item_extraction.get("action_parse_details") or {}).get("loop_truncated"):
                # The sequence was shortened after a degenerate loop. Keep the
                # record visible for review instead of admitting the truncated
                # action list silently.
                quality["blocking_issues"].append("action_sequence_truncated")
                quality["admission_eligible"] = False
            if len(parsed) > 1 and not variant:
                quality["blocking_issues"].append("multiple_records_without_step_alignment")
                quality["admission_eligible"] = False
            # Mapping status remains distinct, while the step route enforces quality.
            if status == "ok" and "quality" in validation and not quality["admission_eligible"]:
                status = "needs_review"
            validation["quality"] = quality
            reactions.append({
                "step_id": step["step_id"], "step_source": deepcopy(step),
                "quality": quality,
                "index": len(reactions) + 1, "segment_index": segment_index,
                "record_index": record_index, "text": text,
                "status": status,
                "reason": (rxn.get("diagnosis") if status == "non_reaction" else quality["blocking_issues"][0] if quality["blocking_issues"] else rxn.get("diagnosis")) if status != "ok" else None,
                "extraction": item_extraction, "coref": coref, "validation": validation,
            })
    base["reactions"] = reactions
    # Intermediates are valid step products, but an example must also include
    # its titled terminal product. No name similarity is treated as proof.
    # A bare paragraph marker ("[0231]") is not a product title: the check
    # cannot be asked there and is skipped, not failed.
    chunk_title = re.sub(r"^\[\d{1,4}\]$", "", str(chunk.get("title") or "").strip()).strip()
    if steps and chunk_title and chunk.get("kind", "example") in ("example", "compound"):
        terminal = steps[-1]
        titled = product_title_norm(terminal.get("title")) == product_title_norm(chunk_title)
        last_items = [r for r in reactions if r.get("step_id") == terminal["step_id"]]
        covered = titled and bool(last_items) and all(
            any(product_title_matches(chunk_title, p.get("name") if isinstance(p, dict) else p)
                for p in ((r.get("coref") or {}).get("ner") or {}).get("REACTION_PRODUCT", []))
            for r in last_items)
        base["terminal_product_coverage"] = "supported" if covered else "unknown"
        if not covered:
            for r in last_items:
                q = r.get("quality")
                if q:
                    q["blocking_issues"].append("terminal_product_coverage_unknown")
                    q["admission_eligible"] = False
                    q["record_completeness"] = "incomplete"
                    if r["status"] == "ok" and "quality" in (r.get("validation") or {}):
                        r["status"] = "needs_review"
                        r["reason"] = "terminal_product_coverage_unknown"
    first_needs_review = next(
        (r for r in reactions if r["status"] == "needs_review"), None)
    first_non_reaction = next(
        (r for r in reactions if r["status"] == "non_reaction"), None)
    if any(r["status"] == "needs_review" for r in reactions):
        base["status"] = "needs_review"
    elif any(r["status"] == "ok" for r in reactions):
        base["status"] = "ok"
    else:
        # Every reaction was rejected by validation: the chunk itself is a
        # non_reaction and its top-level reason names the diagnosis (CHE-1
        # closure, v7 review P-15). These reasons are disjoint from the
        # short-circuit kinds {assay, analysis, purification}, so
        # select_reclassified keeps treating only classification artifacts
        # as re-extractable.
        base["status"] = "non_reaction" if reactions else "needs_review"
    if base["status"] == "needs_review" and first_needs_review is not None:
        base["reason"] = first_needs_review.get("reason")
    elif base["status"] == "non_reaction":
        base["reason"] = (first_non_reaction or {}).get("reason") or "non_reaction"
    if len(reactions) == 1:
        base.update({key: reactions[0][key] for key in
                     ("extraction", "coref", "validation")})
    else:
        base.update(extraction=None, coref=None, validation=None)
    base["context_version"] = chunk.get("context_version")
    return base
