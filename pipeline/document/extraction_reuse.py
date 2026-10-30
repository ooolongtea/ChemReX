"""Reuse original extraction responses, never old admission decisions.

Document scope and literal step input must match. Current parsers and source
checks decide independently whether NER and actions need a new model call.
"""
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import re

from .results import latest_by_id
from .step_context import extraction_text, procedure_body
from pipeline.extraction.parser import parse_ner_records, parse_action_output, parse_action_details
from pipeline.validation.quality import source_metadata, evaluate_quality

MISSING_FIELDS = {'source_yield_not_extracted', 'source_temperature_not_extracted',
                  'source_time_not_extracted', 'local_temperature_not_preserved',
                  'local_time_not_preserved', 'reference_temperature_not_preserved',
                  'reference_time_not_preserved', 'yield_unattributed', 'time_unattributed',
                  'temperature_unattributed', 'step_product_title_unverified',
                  'starting_material_unattributed', 'reaction_product_unattributed',
                  'starting_material_evidence_missing', 'reaction_product_evidence_missing',
                  'reagent_catalyst_unattributed', 'solvent_unattributed'}


class ExtractionReuse:
    def __init__(self, records, source=None, legacy_configuration=None, allow_unversioned=None):
        self.inputs = defaultdict(list)
        self.normalization_inputs = defaultdict(list)
        self.body_index = defaultdict(dict)
        self.source = str(source) if source else None
        self.legacy_configuration = legacy_configuration
        self.allow_unversioned = (os.getenv('CHEMREX_REUSE_SOURCE_CHECKED_LEGACY') == '1'
                                  if allow_unversioned is None else allow_unversioned)
        for record in records:
            units = {}
            legacy_by_language = {}
            for rx in record.get('reactions') or []:
                ex = rx.get('extraction') or {}
                key = (rx.get('segment_index'), rx.get('text'), ex.get('language', 'en'))
                configuration = ex.get('extraction_configuration') or legacy_configuration
                if configuration is None and os.getenv('CHEMREX_LEGACY_CODE_REVISION'):
                    from scripts.legacy_extraction_provenance import historical_configuration_for_record
                    if key[2] not in legacy_by_language:
                        legacy_by_language[key[2]] = historical_configuration_for_record(
                            record, key[2], os.environ['CHEMREX_LEGACY_CODE_REVISION'])
                    configuration = legacy_by_language[key[2]]
                unit = units.setdefault(key, {'raw': ex.get('raw') or {}, 'reactions': [],
                    'configuration': configuration,
                    'source': self.source,
                    'chunk_id': record['id'], 'chunk_fingerprint': record.get('chunk_fingerprint')})
                unit['reactions'].append(rx)
            for (_, text, language), unit in units.items():
                if text:
                    self.normalization_inputs[(text, language)].append(unit)
                    # A deferred/empty record is not a competing model answer.
                    # Keep its normalization evidence, but do not let it make
                    # an actual saved response appear ambiguous.
                    if any(unit['raw'].get(task) for task in ('ner', 'action')):
                        self.inputs[(text, language)].append(unit)
                        self.body_index[(procedure_body(text), language)].setdefault(text, []).append(unit)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        path = folder / 'chunks.jsonl'
        if not path.is_file():
            return cls([])
        chunks = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        records = latest_by_id(folder / 'results.jsonl', {c['id']: c['chunk_fingerprint'] for c in chunks})
        return cls(records.values(), folder / 'results.jsonl')

    def overlay_recent(self, recent):
        """Prefer the latest saved execution for identical inputs on resume.

        Original backups remain available for other inputs. Current outputs
        still pass assess(), including configuration, ambiguity and source checks.
        """
        for key, units in recent.inputs.items():
            self.inputs[key] = units
            self.body_index[(procedure_body(key[0]), key[1])][key[0]] = units
        # New responses may have worse normalization than an older run. Keep
        # both candidates for the existing name/structure consistency checks.
        for key, units in recent.normalization_inputs.items():
            self.normalization_inputs[key].extend(units)

    def include(self, other):
        """Keep candidates from multiple verified copies of the same PDF.

        Conflicting responses and structures remain ambiguous under the existing
        checks, rather than silently overwriting one source with another.
        """
        for key, units in other.inputs.items():
            self.inputs[key].extend(units)
            self.body_index[(procedure_body(key[0]), key[1])][key[0]] = self.inputs[key]
        for key, units in other.normalization_inputs.items():
            self.normalization_inputs[key].extend(units)

    def assess(self, step, language, configuration=None):
        text = extraction_text(step)
        candidates = self.inputs.get((text, language), [])
        reasons = []
        if not candidates:
            # A corrected reference context hides an otherwise usable response
            # for an unchanged procedure body. Re-attach same-body responses;
            # the configuration gate, ambiguity check and every current source
            # check below still decide independently whether they survive.
            same_body = self.body_index.get((procedure_body(text), language)) or {}
            fallback = [dict(u, reactions=[], chunk_fingerprint=None, body_fallback=True)
                        for other, units in same_body.items() if other != text for u in units]
            if fallback:
                candidates = fallback
                reasons.append('same_body_fallback_after_reference_context_change')
        if configuration is not None and candidates:
            compatible = [c for c in candidates if c.get('configuration') == configuration]
            if not compatible:
                compatible = [c for c in candidates if (c.get('configuration') or {}).get('prompt_provenance_verified')
                              and (c.get('configuration') or {}).get('prompts') == configuration.get('prompts')]
            if not compatible and self.allow_unversioned:
                # Treat old raw output as input data, not proof of a model or
                # prompt version. Exact text, unambiguous responses and all
                # current field/source checks below are still required.
                compatible = [c for c in candidates if c.get('configuration') is None]
                if compatible:
                    reasons.append('legacy_configuration_unknown_source_checked')
            if not compatible:
                reasons.append('extraction_configuration_changed_or_unknown')
            candidates = compatible
        unit = None
        if not candidates:
            reasons.append('model_input_changed_or_no_prior_output')
        elif len({json.dumps(c['raw'], sort_keys=True) for c in candidates}) > 1:
            reasons.append('ambiguous_prior_responses')
        else:
            unit = candidates[0]
        ner_diagnostics = []
        parsed = parse_ner_records(unit['raw'].get('ner', ''), ner_diagnostics) if unit else []
        action_details = parse_action_details(unit['raw'].get('action', '') if unit else '')
        actions = action_details['actions']
        needs_ner = not parsed or not any(any(n.values()) for n in parsed)
        needs_action = action_details['status'] != 'ok'
        if unit:
            if needs_ner:
                reasons.append('invalid_prior_ner')
                action_evidence = source_metadata({}, step['local_text'], step, actions)['field_evidence']['ACTION']
                if any(e.get('status') == 'unattributed' for e in action_evidence):
                    needs_action = True
            else:
                missing = set()
                for ner in parsed:
                    ner = deepcopy(ner)
                    if step.get('title') and step.get('boundary_status') == 'source_grounded':
                        ner['REACTION_PRODUCT'] = [step['title'] if isinstance(p, str) and re.fullmatch(
                            r'标题化合物|标题产物|目标产物|title compound|target product|the title compound',
                            p.strip(), re.I) else p for p in ner.get('REACTION_PRODUCT', [])]
                    metadata = source_metadata(ner, step['local_text'], step, actions)
                    q = evaluate_quality(dict(metadata, ner=ner, action=actions))
                    missing.update(set(q['blocking_issues']) & MISSING_FIELDS)
                    action_issues = {'action_unattributed', 'action_order_unverified'} & set(q['blocking_issues'])
                    if action_issues:
                        needs_action = True
                        reasons.extend(sorted(action_issues))
                if missing:
                    needs_ner = True
                    reasons.extend(sorted(missing))
            if needs_action and not {'action_unattributed', 'action_order_unverified'} & set(reasons):
                reasons.append('action_unattributed' if actions else 'invalid_or_empty_prior_actions')
        return {'unit': unit, 'parsed': parsed, 'actions': actions,
                'ner_parse_diagnostics': ner_diagnostics,
                'action_parse_details': action_details,
                'needs_ner': needs_ner, 'needs_action': needs_action, 'reasons': reasons}

    def extract(self, extractor, step, language):
        configuration = extractor.configuration(language) if callable(getattr(type(extractor), 'configuration', None)) else None
        decision = self.assess(step, language, configuration)
        needs_ner, needs_action = decision['needs_ner'], decision['needs_action']
        text = extraction_text(step)
        if needs_ner and needs_action:
            result = extractor.extract(text, language)
            return self._retry_defective_fresh(extractor, result, step, language, configuration)
        unit = decision['unit']
        result = {'language': language, 'task': 'both', 'ner_records': deepcopy(decision['parsed']),
                  'ner': deepcopy(decision['parsed'][0]) if len(decision['parsed']) == 1 else {},
                  'action': deepcopy(decision['actions']), 'raw': deepcopy(unit['raw']),
                  'parse_status': {'ner': 'ok', 'action': 'ok'}}
        result['action_parse_details'] = deepcopy(decision['action_parse_details'])
        result['ner_parse_diagnostics'] = deepcopy(decision['ner_parse_diagnostics'])
        if configuration:
            result['extraction_configuration'] = deepcopy(unit['configuration'])
        if needs_ner or needs_action:
            task = 'ner' if needs_ner else 'action'
            fresh = extractor.extract(text, language, task=task)
            for key in ('ner', 'ner_records') if needs_ner else ('action',):
                result[key] = deepcopy(fresh[key])
            result['raw'][task] = fresh['raw'][task]
            result['parse_status'][task] = fresh['parse_status'][task]
            if needs_action:
                result['action_parse_details'] = deepcopy(fresh.get('action_parse_details'))
            if needs_ner:
                result['ner_parse_diagnostics'] = deepcopy(fresh.get('ner_parse_diagnostics', []))
            if configuration:
                result['extraction_configuration'] = deepcopy(configuration)
                if unit.get('configuration') is None:
                    result['extraction_configuration'] = None
                    result['extraction_task_configurations'] = {
                        task: deepcopy(configuration), ('action' if needs_ner else 'ner'): None}
                elif (unit.get('configuration') or {}).get('model') != configuration.get('model'):
                    result['extraction_configuration'].update(model=None,preset=None,
                        prompt_provenance_verified={'mixed_outputs': True},
                        task_configurations={task: configuration,
                            ('action' if needs_ner else 'ner'): deepcopy(unit.get('configuration'))})
        result['reuse_provenance'] = {'source': unit.get('source') or self.source, 'chunk_id': unit['chunk_id'],
            'chunk_fingerprint': unit['chunk_fingerprint'], 'same_step_input': True,
            'ner_reused': not needs_ner, 'action_reused': not needs_action}
        result['reuse_provenance']['original_configuration'] = deepcopy(unit.get('configuration'))
        if unit.get('configuration') is None and configuration is not None:
            result['reuse_provenance']['compatibility_basis'] = 'exact_input_and_current_source_checks'
            result['reuse_provenance']['original_prompt_configuration_known'] = False
        if unit.get('body_fallback'):
            result['reuse_provenance']['same_step_input'] = False
            result['reuse_provenance']['compatibility_basis'] = 'same_procedure_body_and_current_source_checks'
        return (self._retry_defective_fresh(extractor, result, step, language, configuration)
                if needs_ner or needs_action else result)

    def _retry_defective_fresh(self, extractor, result, step, language, configuration):
        """One source-checked retry of defective newly generated fields.

        Preserve both raw responses. A failed second response stays under review;
        a partial action list is never promoted to a complete sequence here.
        """
        if configuration is None:
            return result
        record = {'id': step['step_id'], 'reactions': [{'text': extraction_text(step),
                                                     'extraction': result}]}
        check = ExtractionReuse([record], allow_unversioned=self.allow_unversioned).assess(step, language, configuration)
        if not (check['needs_ner'] or check['needs_action']):
            return result
        task = 'both' if check['needs_ner'] and check['needs_action'] else 'ner' if check['needs_ner'] else 'action'
        previous = deepcopy(result.get('raw'))
        fresh = extractor.extract(extraction_text(step), language, task=task)
        for field in ('ner', 'action') if task == 'both' else (task,):
            for key in ('ner', 'ner_records', 'ner_parse_diagnostics') if field == 'ner' else ('action', 'action_parse_details'):
                if key in fresh:
                    result[key] = deepcopy(fresh[key])
            result['raw'][field] = fresh['raw'][field]
            result['parse_status'][field] = fresh['parse_status'][field]
            if result.get('reuse_provenance'):
                result['reuse_provenance'][field + '_reused'] = False
        result['extraction_retry'] = {'attempts': 2, 'task': task, 'reasons': check['reasons'],
                                      'first_raw': previous, 'last_raw': deepcopy(fresh['raw'])}
        return result

    def enrich_known_names(self, ner, step, language, record_index, image_evidence=None):
        """Keep only a name's own, consistent normalization and valid candidate.

        Image identifier mappings are resolved again against the current scope.
        MS, atom sources, mapping and admission are always evaluated again.
        A name carrying an image-side label ("胺(H)", "式一化合物(K)") is owned
        by the current scope's image dictionary: coref's image_identifier path
        must resolve it, never a stale prior result.
        """
        from pipeline.standardize.compound_standardizer import CompoundStandardizer
        from pipeline.standardize.standardizer import Standardizer, COMPOUND_FIELDS
        from rdkit import Chem
        from pipeline.validation.normalize import normalize_compound, elemental_form_evidence
        from pipeline.validation.ms_check import evaluate_product_mass
        from .source_conflicts import compound_evidence_for_source, compound_conflict
        from .image_labels import carries_image_label, image_scope_labels
        image_labels = image_scope_labels(image_evidence)
        units = self.normalization_inputs.get((extraction_text(step), language), [])
        if not units:
            return ner
        compounds = defaultdict(list)
        for unit in units:
            for rx in unit['reactions']:
                if rx.get('record_index', 1) != record_index:
                    continue
                coref_ner = (rx.get('coref') or {}).get('ner') or {}
                normalized_products = [normalize_compound(p, 'products', i).to_public_dict()
                                       for i,p in enumerate(coref_ner.get('REACTION_PRODUCT') or [])]
                product_conflict = evaluate_product_mass(step.get('local_text', ''), normalized_products)['status'] == 'mismatch'
                for field in COMPOUND_FIELDS:
                    if field == 'REACTION_PRODUCT' and product_conflict:
                        continue
                    for c in ((rx.get('coref') or {}).get('ner') or {}).get(field) or []:
                        if isinstance(c, dict):
                            compounds[c.get('name')].append((c, unit))
        out = deepcopy(ner)
        for field in COMPOUND_FIELDS:
            for i, name in enumerate(out.get(field) or []):
                if not isinstance(name, str):
                    continue
                if image_labels and carries_image_label(name, image_labels):
                    continue
                eligible, naming_only = [], []
                for c, unit in compounds.get(name, []):
                    source_evidence = compound_evidence_for_source(unit.get('source') or self.source)
                    if compound_conflict(c, source_evidence):
                        continue
                    if field == 'REACTION_PRODUCT':
                        form = elemental_form_evidence(step.get('local_text', ''), c.get('smiles'))
                        if form and form['status'] != 'consistent':
                            continue
                    parsed = CompoundStandardizer._parse_text_response(c.get('naming_raw') or '')
                    info = parsed.get(name)
                    if not info or info['standard_name'] != c.get('standard_name'):
                        continue
                    if info['standard_name'].strip().upper() in {'', 'ID', 'UNKNOWN', 'NONE', 'NOT FOUND'}:
                        continue
                    if c.get('structure_source') not in (None, 'name_resolution'):
                        continue
                    if not c.get('smiles'):
                        naming_only.append((info, unit))
                    smiles = c.get('smiles')
                    if c.get('structure_source') != 'name_resolution' or not smiles or Chem.MolFromSmiles(smiles) is None:
                        continue
                    if normalize_compound(c, 'products' if field == 'REACTION_PRODUCT' else 'reactants', i).status != 'valid':
                        continue
                    info.update(smiles=smiles, cxsmiles=c.get('cxsmiles'),
                                stereochemistry=deepcopy(c.get('stereochemistry')),
                                resolution_status='reused_candidate',
                                resolution_trace=c.get('resolution_trace') or [dict(service='prior_result',
                                    query=name, status='reused_candidate')])
                    eligible.append((info, unit))
                if eligible and len({(c['standard_name'], c['smiles']) for c, _ in eligible}) == 1:
                    info, unit = eligible[0]
                    out[field][i] = Standardizer._enrich_compound(name, {name: info})
                    out[field][i]['normalization_reuse'] = {'source': unit.get('source') or self.source,
                        'chunk_id': unit['chunk_id'], 'record_index': record_index}
                elif not eligible and naming_only and len({c['standard_name'] for c,_ in naming_only}) == 1:
                    info, unit = naming_only[0]
                    # Re-run current structure lookup from the name's own saved
                    # naming response, including the repaired isomer handling.
                    out[field][i] = dict(name=name, standard_name=info['standard_name'],
                        naming_raw=info['naming_raw'], requires_structure_resolution=True,
                        normalization_reuse=dict(source=unit.get('source') or self.source,
                            chunk_id=unit['chunk_id'],record_index=record_index,scope='naming_response'))
        return out
