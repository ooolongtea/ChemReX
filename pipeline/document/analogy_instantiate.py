"""Analogical-example repair.

Two failure shapes keep analogical examples ("依据实施例16C的方法，将实施例16B
替换为实施例43B") out of the route graph:

1. A step is described by reference to another procedure. Expand its cited
   source steps, apply local substitutions, and keep intermediate identities
   separate from the template and other analogical instances.
2. The extraction worked but the step was misclassified as a general
   procedure template (a numbered "1. 化合物19的合成" heading pattern) and
   the template gate blocked it. ``reclassify_analogical_instances`` flips
   such records back to concrete instances when the text carries its own
   characterization data.

Both passes are deterministic and model-free, so the offline repair path can
apply them to archived results.
"""

import re
from copy import deepcopy

from pipeline.validation.quality import evaluate_quality, source_metadata
from pipeline.extraction.yield_values import reported_yields
from .step_context import (_ANALYTICAL, is_analogical_text,
                           procedure_citations, printed_step_number, step_product_title)

# "将实施例16B替换为实施例43B" — the old material is replaced by the new one.
_SUBSTITUTE_REPLACE = re.compile(
    r"将\s*(?P<old>[^，。；;]{1,120}?)\s*替换为\s*(?P<new>[^，。；;]{1,120})")
# "原料019-1取代了实施例8第一步中的原料008-1" — the new material displaces
# the old one.
_SUBSTITUTE_DISPLACE = re.compile(
    r"(?P<new>[^，。；;\n]{1,140}?)\s*取代了?\s*"
    r"(?P<old>[^，。；;\n]{1,100})")
_SPECIFIC_INSTANCE = re.compile(
    r"化合物\s*\d+|中间体\s*\d+|compound\s*\d+|intermediate\s*\d+|\b\d{2,3}-\d\b", re.I)
# "与实施例8中第一步至第三步的化学反应基本相同": the same-manner cue follows
# the reference across a descriptive tail.
_ANALOGICAL_INSTANCE = re.compile(
    r"与.{0,40}?(?:基本相同|大致相同|相同|同样|相似)"
    r"|in\s+the\s+same\s+manner\s+as|analogous\s+to", re.I)
_NONREACTION_KINDS = {'assay', 'analysis', 'purification', 'formulation'}


def _nonreaction_chunk(record, chunk):
    return ((chunk.get('process_kind') or (record.get('source') or {}).get('process_kind'))
            in _NONREACTION_KINDS or
            (record.get('status') == 'non_reaction' and record.get('reason') in _NONREACTION_KINDS))


def _is_analogical_text(text):
    return is_analogical_text(text) or bool(_ANALOGICAL_INSTANCE.search(text))


def _substitutions(text):
    """OLD→NEW material substitutions stated by an analogical example."""
    # Only a comma introducing another replacement separates clauses. Locants
    # such as N,N'-dimethyl retain their chemical punctuation.
    text = re.sub(r',\s*(?=(?:用|将|这里用|不同的是)[^，。；;\n]{0,140}?(?:取代|替换))', '，', text)
    subs = []
    for match in _SUBSTITUTE_REPLACE.finditer(text):
        subs.append((_clean_material(match['old']), _clean_material(match['new'])))
    for match in _SUBSTITUTE_DISPLACE.finditer(text):
        subs.append((_clean_material(match['old']), _clean_material(match['new'])))
    return list(dict.fromkeys((old, new) for old, new in subs if old and new))


def _clean_material(value):
    value = value.strip()
    value = re.sub(r'^(?:实施例|实例)\s*[A-Za-z0-9-]+(?:中|第[^，。]{1,8}步中)(?:的)?', '', value)
    value = re.split(r'(?:唯一)?不同(?:之处|处)?(?:的)?是|这里|而且', value)[-1]
    # These are grammatical introductions, not part of the substance name.
    previous = None
    while previous != value:
        previous = value
        value = re.sub(r'^(?:唯一不同(?:之处|处)?(?:的)?是|不同(?:之处|处)?(?:的)?是|'
                       r'这里|在第[一二三四五六七八九十\d]+步中|'
                       r'第[一二三四五六七八九十\d]+步中的|'
                       r'(?:的)?反应起始(?:物|原料)|而且|用|将)\s*', '', value)
    return value.strip()


def _material_key(value):
    from pipeline.validation.quality import _norm
    value = re.sub(r'^(?:原料|中间体|化合物|产物|compound\b|intermediate\b)\s*',
                   '', str(value or '').strip(), flags=re.I)
    return _norm(value).replace('–', '-').replace('—', '-')


def _renamed(compound, name, source='analogical_substitution'):
    """A renamed molecule has no evidence for the template's molecular graph."""
    result = deepcopy(compound) if isinstance(compound, dict) else {}
    for key in ('smiles', 'canonical_smiles', 'cxsmiles', 'standard_name', 'naming_raw',
                'dict_key', 'stereochemistry', 'image_source', 'resolution_trace',
                'image_scope_unverified', 'image_product_candidates', 'atom_mapping'):
        result.pop(key, None)
    result.update(name=name, structure_source=source, resolution_status='identifier_only',
                  identifier_only=True, status='structure_unavailable', ignored=False,
                  reason='unresolved_structure')
    return result


def _apply_substitution(compounds, old, new):
    """Rename a substituted material; its stale structure never survives."""
    changed = False
    for index, comp in enumerate(compounds if isinstance(compounds, list) else []):
        name = comp.get('name') if isinstance(comp, dict) else comp
        if not name or _material_key(name) != _material_key(old):
            continue
        if isinstance(comp, dict):
            replacement = _renamed(comp, new)
            comp.clear()
            comp.update(replacement)
        else:
            compounds[index] = new
        changed = True
    return changed


def _reaction_compounds(reaction):
    from .routes import compound_lists
    return compound_lists(reaction)


def _method_citations(step, chunk):
    text = step.get('local_text') or ''
    if not _is_analogical_text(text):
        return []
    # A trailing table introduction describes other instances, not this synthesis.
    if re.search(r'\b(?:following\s+(?:Example\s+)?compounds?|Examples?\s+\S+\s+to\s+)'
                 r'|下(?:列|述)(?:实施例|化合物)', text, re.I):
        return []
    return procedure_citations(text, chunk.get('parent_example_id'))


def _template_record(citation, records, chunks, step):
    key = citation['id']
    refs = [r for r in step.get('procedure_references') or [] if r.get('id') == key]
    if any(r.get('status') in ('cycle', 'ambiguous') or r.get('relation') == 'precursor_preparation'
           for r in refs):
        return None
    chunk_ids = {r.get('chunk_id') for r in refs if r.get('status') == 'resolved' and r.get('chunk_id')}
    candidates = []
    for record in records:
        chunk = chunks.get(record.get('id')) or record.get('source') or {}
        if _nonreaction_chunk(record, chunk):
            continue
        if chunk_ids:
            matches = record.get('id') in chunk_ids
        else:
            matches = (chunk.get('reference_id') == key or record.get('id') == key)
        if matches:
            candidates.append(record)
    # Duplicate printed identifiers are alternatives, never first-match wins.
    return candidates[0] if len(candidates) == 1 else None


def _selected_reactions(template, citation, chunks):
    chunk = chunks.get(template['id']) or template.get('source') or {}
    steps = chunk.get('steps') or []
    selected = citation['steps']
    if selected == ():
        return None
    by_step = {s['step_id']: s for s in steps}
    out, found = [], set()
    for reaction in template.get('reactions') or []:
        source = (by_step.get(reaction.get('step_id')) or
                  (steps[0] if len(steps) == 1 else None) or reaction.get('step_source') or {})
        number = printed_step_number(source)
        if selected is not None and number not in selected:
            continue
        if not ((reaction.get('coref') or {}).get('ner') or {}).get('REACTION_PRODUCT'):
            return None
        if reaction.get('reason') == 'invalid_or_empty_ner':
            return None
        found.add(number)
        out.append((reaction, source))
    if selected is not None and not set(selected).issubset(found):
        return None
    return out or None


def _solvent_override(text):
    match = re.search(r'这里(?:用的是|的溶剂是|用的溶剂是)\s*([^，。；,;\n]+?)(?:作溶剂|[，。；,;\n]|$)', text)
    return match[1].strip() if match else None


def _local_products(existing, title):
    from pipeline.validation.quality import product_title_matches
    products = [p for r in existing for p in _reaction_compounds(r).get('products') or []]
    return [deepcopy(p) for p in products if isinstance(p, dict)
            and (product_title_matches(title, p.get('name'))
                 or _material_key(title) == _material_key(p.get('name')))]


def _local_target(step, chunk):
    title = step_product_title(step.get('title') or chunk.get('title'))
    if title:
        return title
    # A body can name its target when MinerU retained only the example number
    # in the heading. Require an explicit local synthesis/characterization cue.
    names = set(re.findall(
        r'化合物\s*([0-9]+[A-Za-z]?)(?![A-Za-z0-9-])'
        r'(?=(?:及其[^。；;\n]{0,60}?)?(?:合成|的合成|的分析数据))',
        step.get('local_text') or ''))
    return '化合物' + names.pop() if len(names) == 1 else ''


def _expand_step(step, chunk, sources, subs, existing, precursors=()):
    """Instantiate a complete selection atomically, keeping local identities."""
    text = step.get('local_text') or ''
    title = _local_target(step, chunk)
    if not title:
        return None
    if len(re.findall(r'替换为|取代了?', text)) > len(subs):
        return None
    solvent = _solvent_override(text)
    if solvent and len(sources) != 1:
        return None  # The affected operation has not been identified unambiguously.
    if re.search(r'溶剂|温度.{0,8}(?:改|不同)|时间.{0,8}(?:改|不同)', text) and not solvent:
        # An unhandled local condition must not be replaced by a template default.
        if re.search(r'不同|改为|改成|这里', text):
            return None
    all_inputs = [c for _template, reaction, _source, _citation in sources
                  for role in ('reactants', 'reagents', 'solvents')
                  for c in _reaction_compounds(reaction).get(role) or []]
    input_keys = {_material_key(c.get('name') if isinstance(c, dict) else c) for c in all_inputs}
    aliases = {_material_key(old): {_material_key(old)} for old, _new in subs}
    alias_evidence = []
    for old, _new in subs:
        key = _material_key(old)
        if key in input_keys:
            continue
        # A template can name a reagent in prose and label it in the drawing.
        # Its own unambiguous image dictionary supplies that identity bridge.
        from .routes import _canonical
        matches = set()
        for template, _reaction, _source, _citation in sources:
            image = template.get('image') or {}
            if key in (image.get('conflicting_identifiers') or []) or key in (image.get('unusable_identifiers') or []):
                continue
            structure = (image.get('dict_result') or {}).get(key)
            if not isinstance(structure, str) or '*' in structure:
                continue
            canonical = _canonical(structure)
            if not canonical:
                continue
            matches.update(_material_key(c.get('name')) for c in all_inputs if isinstance(c, dict)
                           and (c.get('canonical_smiles') or c.get('smiles'))
                           and 'conflict' not in str(c.get('structure_source', ''))
                           and _canonical(c.get('canonical_smiles') or c['smiles']) == canonical)
        if len(matches) != 1:
            return None
        aliases[key].update(matches)
        alias_evidence.append({'identifier': old, 'material_names': sorted(matches),
                               'basis': 'template_image_dictionary_structure'})
    # A local list of substrates is authoritative even without a "replace A by B".
    from pipeline.validation.quality import _literal_in_source, _condition_norm, operation_conditions
    replacements = {_material_key(n) for _, n in subs}
    for reaction in existing:
        for role in ('reactants', 'reagents', 'solvents'):
            for c in _reaction_compounds(reaction).get(role) or []:
                name = c.get('name') if isinstance(c, dict) else c
                if (name and _literal_in_source(name, text) and _material_key(name) not in input_keys | replacements
                        and not (role == 'solvents' and solvent)):
                    # An explicit local substance cannot silently disappear into a clone.
                    return None
    local_conditions = operation_conditions({}, [{'text': text}])
    for condition in local_conditions:
        field = condition['field']
        inherited = [_condition_norm(((r.get('coref') or {}).get('ner') or {}).get(field), field)
                     for _t, r, _s, _c in sources]
        if len(sources) != 1 or not any(condition['value'] and _condition_norm(condition['value'], field) in v
                                        for v in inherited):
            return None
    local_products = _local_products(existing, title)
    mapping = {key: _renamed({}, new) for old, new in subs for key in aliases[_material_key(old)]}
    # Preserve an independently resolved local substrate's structure after renaming.
    local_compounds = [c for r in existing for role in ('reactants', 'reagents', 'solvents')
                       for c in _reaction_compounds(r).get(role) or []]
    for old, new in subs:
        candidates = [c for c in [*local_compounds, *precursors] if isinstance(c, dict)
                      and _material_key(c.get('name')) == _material_key(new)]
        resolved = [c for c in candidates if c.get('canonical_smiles') or c.get('smiles')]
        structures = {c.get('canonical_smiles') or c.get('smiles') for c in resolved}
        if len(structures) == 1:
            for key in aliases[_material_key(old)]:
                mapping[key] = deepcopy(resolved[0])
        elif len(candidates) == 1 and not structures:
            for key in aliases[_material_key(old)]:
                mapping[key] = deepcopy(candidates[0])
    out = []
    for position, (template, original, source, citation) in enumerate(sources):
        terminal = position == len(sources) - 1
        reaction = deepcopy(original)
        compounds = deepcopy(_reaction_compounds(original))
        old_products = compounds.get('products') or []
        if not old_products:
            return None
        # A sequential citation can bridge two differently numbered templates.
        # Infer only a unique coded substrate and a unique preceding product.
        # Within one template, a mismatched printed code remains a source defect.
        if position == 0 or template['id'] != sources[position - 1][0]['id']:
            coded = [c for c in compounds.get('reactants') or [] if isinstance(c, dict)
                     and re.fullmatch(r'\d{2,4}-\d+[a-z]?', _material_key(c.get('name')))]
            preceding = list(precursors) if position == 0 else _reaction_compounds(out[-1]).get('products') or []
            if len(coded) == len(preceding) == 1 and _material_key(coded[0]['name']) not in mapping:
                mapping[_material_key(coded[0]['name'])] = deepcopy(preceding[0])
        for role in ('reactants', 'reagents', 'solvents', 'participants'):
            compounds[role] = [deepcopy(mapping.get(_material_key(c.get('name')), c))
                               if isinstance(c, dict) else deepcopy(mapping.get(_material_key(c), c))
                               for c in compounds.get(role) or []]
        if terminal:
            products = local_products or [_renamed({}, title, 'analogical_title')]
        else:
            products = []
            for product_index, product in enumerate(old_products):
                name = (f"类比中间体[{step['step_id']}/{template['id']}:"
                        f"{original.get('index')}:{product_index + 1}]")
                item = _renamed(product, name, 'analogical_intermediate')
                item['analogical_origin'] = {'chunk_id': template['id'],
                    'reaction_index': original.get('index'), 'product_index': product_index,
                    'name': product.get('name')}
                products.append(item)
        for old_product, product in zip(old_products, products):
            mapping[_material_key(old_product.get('name'))] = deepcopy(product)
        compounds['products'] = deepcopy(products)
        if solvent:
            compounds['solvents'] = [_renamed({}, solvent, 'local_condition_override')]
        for role, items in compounds.items():
            if isinstance(items, list):
                for index, item in enumerate(items):
                    if isinstance(item, dict):
                        item.update(source_field=role, source_index=index)
        scoped = deepcopy(step)
        # Each expanded reaction inherits only its own procedure, not its siblings.
        blocks = deepcopy(source.get('source_parts') or [{'text': original.get('text') or '',
                    'block_idx': None, 'page_idx': None}])
        scoped['procedure_references'] = [{'id': citation['id'], 'status': 'resolved',
            'chunk_id': template['id'], 'blocks': blocks,
            'text': '\n'.join(b['text'] for b in blocks)}]
        scoped['procedure_references'].extend(deepcopy(source.get('procedure_references') or []))
        scoped['context_issues'] = [i for i in step.get('context_issues') or []
            if i != 'telescoped_procedures_without_reported_intermediates']
        scoped['unit_kind'] = 'single_step'
        if not terminal:
            scoped.update(local_text='', source_parts=[], title=products[0]['name'], title_source={})
        else:
            scoped['title'] = title
        reaction.update(step_id=step['step_id'], step_source=scoped, text=text,
                        segment_index=step.get('step_index', 1), record_index=position + 1)
        fields = {'reactants': 'STARTING_MATERIAL', 'products': 'REACTION_PRODUCT',
                  'reagents': 'REAGENT_CATALYST', 'solvents': 'SOLVENT'}
        for layer in ('coref', 'extraction'):
            holder = reaction.setdefault(layer, {})
            ner = holder.setdefault('ner', {})
            for role, field in fields.items():
                ner[field] = deepcopy(compounds.get(role) or [])
            holder['reactant_side_participants'] = deepcopy(compounds.get('participants') or [])
        reaction.setdefault('validation', {}).setdefault('rxn_validation', {})['normalized_compounds'] = compounds
        reaction['analogical_instantiation'] = {
            'template_chunk_id': template['id'],
            'template_reference_id': (template.get('source') or {}).get('reference_id'),
            'template_reaction_index': original.get('index'),
            'template_step_id': original.get('step_id'),
            'from_steps': list(citation['steps']) if citation['steps'] is not None else None,
            'substitutions': [{'from': old, 'to': new} for old, new in subs],
            'title_product': title, 'expanded': True, 'terminal': terminal}
        if alias_evidence:
            reaction['analogical_instantiation']['substitution_aliases'] = deepcopy(alias_evidence)
        _refresh_instance_evidence(reaction, text if terminal else '', scoped)
        out.append(reaction)
    return out


def _refresh_instance_evidence(reaction, text, step):
    """Replace template-local evidence and validation with this instance's data."""
    from pipeline.validation.constraints import check_constraints
    from pipeline.validation.ms_check import evaluate_product_mass

    coref = reaction.get('coref')
    if not isinstance(coref, dict) or not isinstance(coref.get('ner'), dict):
        # Failed template extraction supplies no compounds to instantiate.
        return
    reported = reported_yields(text)
    values = list(dict.fromkeys(r['normalized'] for r in reported))
    coref['ner']['YIELD_PERCENT'] = values[0] if len(values) == 1 else None
    coref.update(text=text, ms_source_text=text, step_source=deepcopy(step),
                 record_kind='reaction_candidate')
    coref.update(source_metadata(coref['ner'], text, step, coref.get('action')))
    extraction = reaction.get('extraction')
    if isinstance(extraction, dict):
        extraction.setdefault('ner', {})['YIELD_PERCENT'] = coref['ner'].get('YIELD_PERCENT')
        extraction.update(source_metadata(extraction['ner'], text, step, coref.get('action')))
        extraction.update(text=text, ms_source_text=text, step_source=deepcopy(step),
                          record_kind='reaction_candidate')
    previous = reaction.get('validation') or {}
    normalized = deepcopy((previous.get('rxn_validation') or {}).get('normalized_compounds'))
    # Old atom mappings, scores, constraints, MS and yield evidence describe
    # the template. Recompute local checks; mapping remains explicitly deferred.
    validation = deepcopy(coref)
    validation.pop('quality', None)
    validation['rxn_validation'] = {'status': 'deferred', 'needs_human_review': True,
                                  'patch_accepted': False}
    if isinstance(normalized, dict):
        validation['rxn_validation']['normalized_compounds'] = normalized
    constraints = check_constraints(coref)
    mass = evaluate_product_mass(text, (normalized or {}).get('products') or [])
    if mass['issue'] is not None:
        constraints['issues'].append(mass['issue'])
        constraints['passed'] = False
        constraints['summary'] = f"{len(constraints['issues'])} constraint issue(s)"
    validation['rxn_validation'].update(field_constraints=constraints, mass_check=mass)
    # Coref can carry the template's validation too; it must not survive there.
    for key in ('rxn_validation', 'field_constraints', 'quality'):
        coref.pop(key, None)
    reaction['validation'] = validation
    quality = evaluate_quality(validation, 'needs_review')
    validation['quality'] = deepcopy(quality)
    reaction.update(quality=quality, status='needs_review',
                    reason=(quality['blocking_issues'][0] if quality['blocking_issues']
                            else 'analogical_instantiation'))


def refresh_analogical_instances(records, chunks):
    """Repair already saved instances in place, preserving their reaction ids."""
    chunks_by_id = {chunk.get('id'): chunk for chunk in chunks}
    refreshed = []
    for record in records:
        chunk = chunks_by_id.get(record.get('id')) or {}
        if _nonreaction_chunk(record, chunk):
            continue
        steps = {step['step_id']: step for step in chunk.get('steps') or []}
        for reaction in record.get('reactions') or []:
            if not reaction.get('analogical_instantiation'):
                continue
            before = deepcopy(reaction)
            instance = reaction['analogical_instantiation']
            step = (reaction.get('step_source') if instance.get('expanded') else
                    steps.get(reaction.get('step_id')) or reaction.get('step_source')) or {}
            text = (step.get('local_text') or '') if instance.get('expanded') else (
                step.get('local_text') or reaction.get('text') or '')
            _refresh_instance_evidence(reaction, text, step)
            if before != reaction:
                refreshed.append(record['id'])
    return list(dict.fromkeys(refreshed))


def instantiate_analogical_reactions(records, chunks):
    """Expand step-local procedure dependencies until no pending step can advance."""
    chunks_by_id = {chunk['id']: chunk for chunk in chunks}
    pending = {}
    for record in records:
        chunk = chunks_by_id.get(record.get('id'))
        if not chunk or not isinstance(record.get('reactions'), list) or _nonreaction_chunk(record, chunk):
            continue
        for step in chunk.get('steps') or []:
            existing = [r for r in record['reactions'] if r.get('step_id') == step['step_id']
                        or len(chunk['steps']) == 1]
            if any(r.get('analogical_instantiation') for r in existing):
                continue
            citations = _method_citations(step, chunk)
            if citations:
                pending[(record['id'], step['step_id'])] = (record, chunk, step, citations, existing)
    instantiated = []
    while pending:
        progressed = False
        for identity, (record, chunk, step, citations, existing) in list(pending.items()):
            sources = []
            for citation in citations:
                if citation['id'].startswith('step_') and citation.get('scope'):
                    number = re.fullmatch(r'step_(\d+)', citation['id'])
                    selection = citation['steps']
                    if selection is None:
                        selection = (int(number[1]),) if number else ()
                    if not selection:
                        break
                    citation = dict(citation, id=citation['scope'], steps=selection)
                template = _template_record(citation, records, chunks_by_id, step)
                if template is None or template['id'] == record['id']:
                    break
                selected = _selected_reactions(template, citation, chunks_by_id)
                if not selected or any((template['id'], source.get('step_id')) in pending
                                       for _r, source in selected):
                    break
                sources.extend((template, r, source, citation) for r, source in selected)
            else:
                if len(sources) == len(existing) == 1:
                    local = (existing[0].get('coref') or {}).get('ner') or {}
                    if (existing[0].get('reason') != 'invalid_or_empty_ner'
                            and local.get('STARTING_MATERIAL') and local.get('REACTION_PRODUCT')):
                        # A one-step NER already represents the cited operation. Retain
                        # its resolved molecules; structural incompleteness alone is
                        # not evidence that replacing it with a template is better.
                        pending.pop(identity)
                        progressed = True
                        continue
                subs = _substitutions(step.get('local_text') or '')
                step_position = next(i for i, s in enumerate(chunk['steps']) if s['step_id'] == step['step_id'])
                prior_id = chunk['steps'][step_position - 1]['step_id'] if step_position else None
                prior = [r for r in record['reactions'] if r.get('step_id') == prior_id]
                precursors = (_reaction_compounds(prior[-1]).get('products') or []) if prior else []
                expanded = _expand_step(step, chunk, sources, subs, existing, precursors)
                if not expanded:
                    continue
                previous = record['reactions']
                # Replace only this source step, preserving explicit siblings and order.
                position = next((i for i, r in enumerate(previous) if r in existing), len(previous))
                record['reactions'] = [r for r in previous[:position] if r not in existing] + expanded + [
                    r for r in previous[position:] if r not in existing]
                for index, reaction in enumerate(record['reactions'], 1):
                    reaction['index'] = index
                for template_id in dict.fromkeys(t['id'] for t, _, _, _ in sources):
                    instantiated.append({'chunk_id': record['id'], 'template': template_id,
                                         'substitutions': [list(s) for s in subs]})
                pending.pop(identity)
                progressed = True
        if not progressed:
            # Missing templates and cycles leave their original NER untouched.
            break
    return instantiated


def reclassify_analogical_instances(records):
    """An analogical instance with its own characterization data is a real
    example, never a general procedure template (CN105085489B pattern)."""
    reclassified = []
    for record in records:
        for reaction in record.get("reactions") or []:
            step = reaction.get("step_source") or {}
            quality = reaction.get("quality") or {}
            if "procedure_template_without_specific_instance" not in (quality.get("blocking_issues") or []):
                continue
            text = step.get("local_text") or reaction.get("text") or ""
            if not _is_analogical_text(text):
                continue
            if not (_ANALYTICAL.search(text) and _SPECIFIC_INSTANCE.search(text)):
                continue
            step["record_kind"] = "reaction_candidate"
            coref = reaction.get("coref") or {}
            for layer in ('extraction', 'coref', 'validation'):
                holder = reaction.get(layer) or {}
                if holder.get('record_kind') == 'procedure_template':
                    holder['record_kind'] = 'reaction_candidate'
                if isinstance(holder.get('step_source'), dict):
                    holder['step_source']['record_kind'] = 'reaction_candidate'
            validation = reaction.get("validation") or {}
            quality_record = dict(coref, **{k: v for k, v in validation.items()
                                            if k not in ("ner", "action")})
            quality_record['step_source'] = reaction.get('step_source')
            quality_record['record_kind'] = 'reaction_candidate'
            new_quality = evaluate_quality(quality_record, reaction.get("status"))
            reaction["quality"] = new_quality
            validation['quality'] = deepcopy(new_quality)
            rv = (validation.get("rxn_validation") or {})
            if (not new_quality["blocking_issues"]
                    and rv.get("status") in ("passed", "corrected")
                    and not rv.get("needs_human_review", True)):
                reaction["status"] = "ok"
                reaction["reason"] = None
            else:
                reaction["status"] = "needs_review"
                reaction["reason"] = (new_quality["blocking_issues"][0]
                                      if new_quality["blocking_issues"] else reaction.get("reason"))
            reaction["analogical_reclassified"] = True
            reclassified.append(record.get("id"))
    return reclassified


def apply_analogy_repair(records, chunks):
    """Document-level analogical repair used by both the batch pipeline and
    the offline route-repair scan."""
    reclassified = reclassify_analogical_instances(records)
    refreshed = refresh_analogical_instances(records, chunks)
    instantiated = instantiate_analogical_reactions(records, chunks)
    if reclassified or instantiated or refreshed:
        changed = set(reclassified) | set(refreshed) | {r['chunk_id'] for r in instantiated}
        for record in records:
            if record.get('id') not in changed:
                continue
            reactions = record.get('reactions') or []
            chunk = next((c for c in chunks if c.get('id') == record.get('id')), {})
            steps = chunk.get('steps') or []
            if steps and reactions:
                from pipeline.validation.quality import product_title_matches
                title = _local_target(steps[-1], chunk)
                if title:
                    covered = (reactions[-1].get('step_id') == steps[-1]['step_id'] and any(
                        product_title_matches(title, p.get('name'))
                        for p in _reaction_compounds(reactions[-1]).get('products') or []))
                    record['terminal_product_coverage'] = 'supported' if covered else 'unknown'
            review = next((r for r in reactions if r.get('status') == 'needs_review'), None)
            if review is not None:
                record.update(status='needs_review', reason=review.get('reason'))
            record.update({key: reactions[0].get(key) if len(reactions) == 1 else None
                           for key in ('extraction', 'coref', 'validation')})
        result = {"reclassified_chunk_ids": reclassified, "instantiated": instantiated}
        if refreshed:
            result['refreshed_chunk_ids'] = refreshed
        return result
    return None
