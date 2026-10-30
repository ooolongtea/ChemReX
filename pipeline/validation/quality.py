"""Evidence contract for admission, separate from atom-mapping screening."""
import re
from copy import deepcopy
from .atom_sources import check_atom_sources
from .atom_sources import check_heavy_atom_sources
from pipeline.extraction.yield_values import reported_yields, parse_yield_value, yield_equal

QUALITY_VERSION = '2026-10-20-source-quality-repair'
_FIELDS = ('STARTING_MATERIAL', 'REACTION_PRODUCT', 'REAGENT_CATALYST', 'SOLVENT', 'TEMPERATURE', 'TIME', 'YIELD_PERCENT', 'ACTION')


def _norm(text, keep_spaces=False):
    text = re.sub(r'^\s*\[\d+\]\s*(?=\S)', '', str(text or ''))
    greek = {'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
             'epsilon': 'ε', 'zeta': 'ζ', 'eta': 'η', 'theta': 'θ'}
    text = re.sub(r'\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta)\b',
                  lambda m: greek[m[1]], text)
    text = re.sub(r'\\(?:mathrm|text|mathbf)\{([^}]*)\}', r'\1', str(text or ''))
    # Normalize numeric LaTeX subscripts, including separate math fragments
    # such as CH $_2$ Cl $_2$. Plain source identifiers (sample_2) stay distinct.
    text = re.sub(r'\$([^$]*)\$', lambda m: re.sub(
        r'_\s*(?:\{\s*(\d+)\s*\}|(\d+))', lambda n: n[1] or n[2], m[1]), text)
    text = re.sub(r'_\{\s*(\d+)\s*\}', r'\1', text)
    text = text.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
    text = re.sub(r'(?<![A-Za-z])(?:RT|room temperature|room temp|ambient temperature|ambient temp)(?![A-Za-z])', '室温', text, flags=re.I)
    text = re.sub(r'\\+circ', '°', text)
    text = text.translate(str.maketrans({'（':'(', '）':')', '，':',', '−':'-', '％':'%'}))
    text = re.sub(r'(?<=[\u4e00-\u9fff])-(?=[\u4e00-\u9fff])', '', text)
    text = re.sub(r'(?<=[α-ωΑ-Ω])-\s*(?=[\u4e00-\u9fff])', '', text)
    if keep_spaces:
        # Word boundaries only survive normalization when spaces do; the
        # boundary-aware name matcher works on this variant.
        # "$30^{\circ}$ C." keeps a space between the degree sign and C.
        return re.sub(r'°\s*c', '℃', re.sub(r'\\|\$|[{}^]', '', re.sub(r'\s+', ' ', text)).lower())
    return re.sub(r'°\s*c', '℃', re.sub(r'\s|\\|\$|[{}^]', '', text).lower())


def _condition_norm(text, field, keep_spaces=False):
    text = str(text or '')
    if field == 'TIME':
        text = re.sub(r'\\(?:mathrm|text|mathbf)\{([^}]*)\}', r'\1', text)
        # Convert units while word boundaries still exist: removing spaces
        # first makes "2 h to afford" look like "2htoafford". The bare "h"
        # stays lowercase-only: NMR proton counts ("1H NMR", "(d, 1H, J=6.7
        # Hz)") are not reaction times.
        text = re.sub(r'(\d)\s*(?:hours?|hrs?)(?![a-z])', r'\1小时', text, flags=re.I)
        text = re.sub(r'(\d)\s*h(?![a-zA-Z0-9_{}\\-])', r'\1小时', text)
        text = re.sub(r'(\d)\s*(?:minutes?|min)(?![a-z])', r'\1分钟', text, flags=re.I)
    return _norm(text, keep_spaces=keep_spaces).replace('–', '-').replace('—', '-')


def product_title_norm(text):
    text = _norm(text).rstrip(':：').strip()
    text = re.sub(r'^\d{1,3}[.、．](?=(?:中间体|化合物|终产物|产品|目标物))', '', text)
    english = re.match(r'^(?:synthesis|preparation)of', text, re.I)
    chinese = bool(re.search(r'的(?:制备|合成)$', text))
    if english:
        text = text[english.end():]
    text = re.sub(r'的(?:制备|合成)$', '', text)
    if english or chinese:
        text = re.sub(r'\((?:\d+|P[- ]?\d+|E\d+)\)$', '', text, flags=re.I)
    return text


def product_title_matches(title, name):
    title, name = product_title_norm(title), product_title_norm(name)
    if title == name:
        return True
    parent = re.fullmatch(r'化合物(\d+[a-z]?)', title)
    if parent:
        # Combined parent/salt labels refer to this same numbered compound.
        # Preserve their original name and structure rather than renaming a
        # resolved local product to an identifier-only title.
        salt = re.fullmatch(
            re.escape(title) + r'(?:及其|的)(?:甲磺酸盐|磺酸盐|盐酸盐|二盐酸盐|氢溴酸盐|硫酸盐)'
            r'(?:' + re.escape(parent[1]) + r')?'
            r'(?:[.·][a-z0-9_()+.\-]+|\([a-z0-9_()+.\-]+|(?:ms[o]?h|hcl)[_0-9n]*)?', name)
        if salt:
            return True
    # Some patents title the parent and explicitly isolate its salt in the
    # body. Allow that direction only; a free base cannot satisfy a salt title.
    return bool(title and name.startswith(title) and re.fullmatch(
        r'(?:mono|di|tri)?hydro(?:chloride|bromide|iodide)(?:salt)?|sulfate(?:salt)?|'
        r'mesylate(?:salt)?|盐酸盐|二盐酸盐|氢溴酸盐|硫酸盐|甲磺酸盐', name[len(title):]))


_ANALYSIS_START = re.compile(
    r'(?:\$[^$\n]{0,30}\$\s*[HC]?\s*|[¹²³⁹]+\s*[HC]\s*|\b(?:1|13|19)\s*[HCF]\s*|'
    r'(?:\{\s*\}\s*)?\^\s*\{\s*(?:1|13|19)\s*\}\s*(?:\\mathrm\{)?[HCF]\}?\s*(?:\\?\)\s*)?)?\bNMR\b|'
    r'UPLC|(?:ESI|HR|LC|APCI)[- ]?MS|\bMS\s*\(\s*(?:APCI|ESI|EI|CI)\b|'
    r'元素分析|\belemental\s+analysis\b|\bm\s*/\s*[ze]\b', re.I)


def reaction_condition_text(text, workup=False):
    match = None
    for candidate in _ANALYSIS_START.finditer(text or ''):
        token = candidate[0]
        preceding = (text or '')[max(0,candidate.start()-65):candidate.start()]
        preceding = re.split(r'[。;\n]|(?<!\d)\.(?!\d)', preceding)[-1]
        following = (text or '')[candidate.end():candidate.end()+35]
        if (re.fullmatch(r'(?:UPLC|LC[- ]?MS)', token, re.I)
                and (re.search(r'直至|直到|监测|完成|until|complete|monitor|checked|determined', preceding, re.I)
                     or re.match(r'\s*(?:测定|监测|checked)', following, re.I))):
            continue
        match = candidate
        break
    value = (text or '')[:match.start()] if match else (text or '')
    if workup:
        value = re.split(r'洗涤|萃取|纯化|浓缩|washed|extracted|purified|concentrated', value, maxsplit=1, flags=re.I)[0]
    return value


def operation_conditions(ner, sources):
    result, workup = [], False
    pattern = re.compile(r'(?:below\s+|less than\s+|低于\s*)?-?\d+(?:\.\d+)?(?:\s*[-–—]\s*\d+(?:\.\d+)?)?\s*(?:°\s*C|℃|hrs?\b|hours?\b|h\b|minutes?\b|min\b|小时|分钟)|room temperature|ambient temperature|室温|overnight|过夜', re.I)
    for part in sources:
        text = reaction_condition_text(part['text'])
        boundary = re.search(r'washed|extracted|purified|concentrated|洗涤|萃取|纯化|浓缩', text, re.I)
        for match in pattern.finditer(text):
            phase = 'workup' if workup or (boundary and match.start() >= boundary.start()) else 'reaction'
            field = 'TEMPERATURE' if re.search(r'℃|°\s*C|temperature|室温', match.group(0), re.I) else 'TIME'
            value = _condition_norm(match.group(0), field)
            result.append({'field': field, 'value': match.group(0), 'phase': phase,
                           'extracted': value in _condition_norm(ner.get(field), field),
                           'block_idx': part.get('block_idx'), 'page_idx': part.get('page_idx'),
                           'span': list(match.span())})
        workup = workup or bool(boundary)
    return result


_CJK_MODIFIER_LEAD = '无异叔仲邻间对正偏顺反氢氧聚环高二三四五六过甲乙丙丁-'
_CJK_MODIFIER_TRAIL = '钠钾钙镁锌铁铜铝钡锂盐酯酐氯溴碘'


def _boundary_pattern(value):
    """Whole-name match on normalized text. Separator runs stay flexible in
    both directions — a space in the value may be missing in the OCR-glued
    source ("制备型HPLC" vs "制备型 HPLC") — but a name must not sit inside
    a longer word: "ethanol" cannot pass on "methanol", nor "丙醇" on
    "异丙醇"."""
    stripped = value.strip()
    pattern = r'[-\s]*'.join(re.escape(ch) for ch in stripped if not ch.isspace())
    lead = 'A-Za-z0-9' if re.match(r'[A-Za-z0-9]', stripped[0]) else _CJK_MODIFIER_LEAD
    trail = 'A-Za-z0-9' if re.match(r'[A-Za-z0-9]', stripped[-1]) else _CJK_MODIFIER_TRAIL
    return re.compile(r'(?<![' + lead + r'])' + pattern + r'(?![' + trail + r'])')


def _action_forms(tok):
    """Inflected forms of an action lemma: regular suffixes, e-drop
    ("elute" -> "eluting"), doubled consonants ("stir" -> "stirred") and
    y -> i ("dry" -> "dried"). Noun derivations are deliberately absent."""
    forms = {tok, tok + 's', tok + 'es', tok + 'd', tok + 'ed', tok + 'ing',
             tok + tok[-1] + 'ed', tok + tok[-1] + 'ing'}
    if tok.endswith('e'):
        forms.add(tok[:-1] + 'ing')
    if tok.endswith('y'):
        forms |= {tok[:-1] + 'ied', tok[:-1] + 'ies'}
    return sorted(forms, key=len, reverse=True)


def _action_pattern(value):
    """Actions are extracted as lemma verbs ("suspend", "add") while the
    narrative carries inflected forms ("was suspended", "added"), so a
    Latin token matches any of its inflected forms. Noun derivations stay
    out: "add" must not pass on "addition", nor "mix" on "mixture". CJK
    action verbs carry no compound-modifier guard: "加入" must match
    before "溴" even though a compound name would not."""
    stripped = value.strip()
    parts = []
    for tok in stripped.split():
        if re.match(r'[A-Za-z]', tok[0]) and len(tok) > 1:
            parts.append(r'(?:' + '|'.join(re.escape(f) for f in _action_forms(tok)) + r')')
        else:
            parts.append(r'[-\s]*'.join(re.escape(ch) for ch in tok))
    lead = 'A-Za-z0-9' if re.match(r'[A-Za-z0-9]', stripped[0]) else ''
    trail = 'A-Za-z0-9' if re.match(r'[A-Za-z0-9]', stripped[-1]) else ''
    return re.compile((r'(?<![' + lead + r'])' if lead else '') +
                      r'[-\s]*'.join(parts) +
                      (r'(?![' + trail + r'])' if trail else ''))


def _literal_in_source(piece, text, action=False):
    value = _norm(piece, keep_spaces=True)
    if not value.strip(' -'):
        return False
    pattern = _action_pattern(value) if action else _boundary_pattern(value)
    return bool(pattern.search(_norm(text, keep_spaces=True)))


def action_order_violated(actions, local_text):
    """The delivered action order must be traceable to the source narrative.
    Greedy forward scan over literally matchable actions; an action that only
    occurs before its predecessor's matched position is an order violation."""
    if not actions or len(actions) < 2 or not local_text:
        return False
    context = _norm(local_text, keep_spaces=True)
    pos = positioned = 0
    for item in actions:
        name = item.get('name', '') if isinstance(item, dict) else str(item)
        value = _norm(name, keep_spaces=True)
        if not value.strip(' -'):
            continue
        pattern = _action_pattern(value)
        found = pattern.search(context, pos)
        if found:
            pos, positioned = found.end(), positioned + 1
        elif pattern.search(context):
            return True
    return False


def source_metadata(ner, local_text, step, action=None):
    """Attach literal source evidence; absence never becomes proof of completeness."""
    sources = step.get('source_parts') or []
    programs = [r for r in step.get('procedure_references') or [] if r.get('relation') != 'precursor_preparation']
    evidence = {}
    for field in _FIELDS:
        value = action if field == 'ACTION' else ner.get(field)
        evidence[field] = []
        if value in (None, '', []):
            evidence[field].append({'status': 'not_reported_or_not_extracted'})
            continue
        for item in value if isinstance(value, list) else [value]:
            name = item.get('name', '') if isinstance(item, dict) else str(item)
            pieces = [p.strip() for p in re.split(r'[;；]', name) if p.strip()] if field in ('TEMPERATURE', 'TIME') else [name]
            for piece in pieces or [name]:
                def supports(text):
                    if field in ('TEMPERATURE', 'TIME'):
                        text = reaction_condition_text(text)
                    value = _condition_norm(piece, field) if field in ('TEMPERATURE', 'TIME') else _norm(piece)
                    if field == 'TIME':
                        value = re.sub(r'^(?:随后|然后)(?:额外|另外)?(?=\d)', '', value)
                    context = _condition_norm(text, field) if field in ('TEMPERATURE', 'TIME') else _norm(text)
                    # A name is evidence only as a whole word: "ethanol" must
                    # not pass because "methanol" contains it.
                    literal_piece = re.sub(r'^(?:随后|然后)(?:额外|另外)?\s*(?=\d)', '', str(piece)) if field == 'TIME' else piece
                    literal = _literal_in_source(literal_piece, text, action=(field == 'ACTION'))
                    if field in ('TEMPERATURE', 'TIME') and re.match(r'-?\d', value):
                        suffix = r'(?!\d)' if value[-1].isdigit() else ''
                        literal = bool(re.search(r'(?<![\d.\-])' + re.escape(value) + suffix, context))
                    if field == 'ACTION' and piece == '回温':
                        return literal or bool(re.search(r'回到室温|恢复至室温|warmed to room temperature', text, re.I))
                    if field == 'ACTION':
                        from pipeline.extraction.parser import action_supported
                        return literal or action_supported(piece, text)
                    return literal
                matched = [p for p in sources if supports(p['text'])]
                if field == 'REACTION_PRODUCT' and product_title_norm(name) == product_title_norm(step.get('title')) and step.get('title_source', {}).get('block_idx') is not None:
                    matched = matched or [step['title_source']]
                inherited = [dict(b, procedure_id=r['id']) for r in programs if r.get('status') == 'resolved'
                             for b in r['blocks'] if supports(b['text'])]
                if not matched and not inherited and field == 'REACTION_PRODUCT' and re.search(r'标题化合物|目标产物|title compound|target product', name, re.I):
                    matched = [{'text': step.get('title'), 'block_idx': None, 'page_idx': None}]
                evidence[field].append({'value': piece, 'extracted_value': name,
                                        'status': 'local' if matched else 'inherited' if inherited else 'unattributed',
                                        'sources': deepcopy(matched or inherited)})
    reported = reported_yields(local_text)
    extracted = ner.get('YIELD_PERCENT')
    values = [r['normalized'] for r in reported]
    explicit = extracted not in (None, '') and any(yield_equal(v, extracted) for v in values)
    yield_info = {'kind': 'explicit' if explicit else 'unattributed' if extracted is not None else 'not_extracted' if reported else 'not_reported',
                  'value': extracted, 'reported_values': values,
                  'source_text': [r['source_text'] for r in reported], 'basis': None}
    parsed_yield = parse_yield_value(extracted)
    if parsed_yield:
        supported = next((r for r in reported if yield_equal(r['normalized'], extracted)), None)
        yield_info.update(range=parsed_yield['range'], approximate=parsed_yield['approximate'] or bool(supported and supported['approximate']),
                          raw=(supported['raw'] if supported else str(extracted)))
    # Deterministic, conservative derivation: a single substrate with a literal
    # mmol amount, and a single isolated product amount in this very step.
    starting = ner.get('STARTING_MATERIAL') or []
    if not reported and len(starting) == 1:
        name = starting[0].get('name', '') if isinstance(starting[0], dict) else starting[0]
        pos = local_text.find(name)
        substrate = re.search(r'[（(][^）)]{0,100}?([\d.]+)\s*mmol[^）)]*[）)]', local_text[pos + len(name):pos + len(name) + 180], re.I) if pos >= 0 else None
        isolated = list(re.finditer(r'(?:得到|获得|afforded|yielded|obtained)[^。\n]{0,80}?[（(][^）)]{0,100}?([\d.]+)\s*mmol[^）)]*[）)]', local_text, re.I))
        if substrate and len(isolated) == 1:
            a, b = float(substrate.group(1)), float(isolated[0].group(1))
            if a > 0 and 0 <= b <= a:
                derived = round(100 * b / a, 2)
                if extracted is None or (parsed_yield and parsed_yield['value'] is not None and abs(parsed_yield['value'] - derived) <= 0.02):
                    yield_info = {'kind': 'derived', 'value': str(derived), 'reported_values': [],
                                  'source_text': [substrate.group(0), isolated[0].group(0)],
                                  'basis': {'substrate_mmol': substrate.group(1), 'product_mmol': isolated[0].group(1),
                                            'formula': '100 * product_mmol / substrate_mmol', 'decimals': 2}}
                    ner['YIELD_PERCENT'] = str(derived)
    stereochemistry = {'kind': 'unspecified', 'descriptors': [], 'structure_coarsened': False}
    title = step.get('title') or ''
    # Source semantics belong to the product: avoid taking a precursor's
    # relative configuration as the final product's specification.
    product_region = title + ' ' + '\n'.join(p['text'] for p in sources if not re.search(r'加入|搅拌|added|stirred', p['text'], re.I))
    relative = re.findall(r'\([^)]*\d\s*[SR]\s*\\?\*[^)]*\)|\brel-[^\s,;]+', product_region, re.I)
    mixture_text = re.sub(r'\s+', ' ', title + ' ' + local_text)
    mixtures = re.findall(r'[^。\n]{0,40}(?:非对映异构体(?:的)?\s*混合物|顺(?:式)?(?:[／/]|和|及)反(?:式)?[^。\n]{0,20}混合物|顺反混合物|diastereomeric mixture|cis[-/]trans mixture)[^。\n]{0,40}', mixture_text, re.I)
    # Example339A explicitly isolates the title compound "as a mixture of
    # two isomers". This describes the product, whereas a starting-material
    # isomer mixture alone must not determine the product's stereochemistry.
    mixtures += re.findall(
        r'\b(?:title compound|target (?:compound|product)|product)\b[^.。\n]{0,80}?'
        r'\bas\s+(?:a\s+)?mixture\s+of\s+(?:(?:two|three|\d+)\s+)?(?:stereo)?isomers\b',
        mixture_text, re.I)
    # Standalone "(S)"/"(R)" also declare absolute configuration; the
    # lookbehind keeps plural parentheses like "salt(s)" out.
    absolute = re.findall(r'(?<![A-Za-z])\(\s*[+\-−]?\s*(?:[SR]\s*[,，]?\s*)+[+\-−]?\s*\)|\([^)]*\d[SR][^)]*\)',
                          title, re.I)
    from pipeline.chemical_quality import racemic_structure_name, split_isomer_label
    product_names = [p.get('name', '') if isinstance(p, dict) else str(p)
                     for p in ner.get('REACTION_PRODUCT') or []]
    racemic = [p for p in [title, *product_names] if racemic_structure_name(p)
               and _norm(p) in _norm(title + ' ' + local_text)]
    if racemic:
        stereochemistry.update(kind='mixture', descriptors=list(dict.fromkeys(racemic)), structure_coarsened=True)
    elif mixtures:
        stereochemistry.update(kind='mixture', descriptors=mixtures, structure_coarsened=True)
    elif relative:
        stereochemistry.update(kind='relative', descriptors=list(dict.fromkeys(relative)), structure_coarsened=True)
    elif absolute:
        stereochemistry.update(kind='absolute', descriptors=absolute)
    labels = list(dict.fromkeys(label for name in [title, *product_names]
                                if (label := split_isomer_label(name)[1])))
    if labels and not racemic and not mixtures and not relative:
        stereochemistry.update(kind='unassigned_isomer', descriptors=absolute + labels,
                               structure_coarsened=True)
    return {'field_evidence': evidence, 'yield_info': yield_info, 'stereochemistry': stereochemistry,
            'operation_conditions': operation_conditions(ner, sources),
            'step_source': deepcopy(step), 'quality_version': QUALITY_VERSION}


def evaluate_quality(record, reaction_status=None, chunk_status=None):
    rv = record.get('rxn_validation') or {}
    step = record.get('step_source') or {}
    atom = deepcopy(rv.get('atom_source_check') or check_atom_sources(rv.get('effective_expression') or rv.get('base_expression') or ''))
    heavy = check_heavy_atom_sources(rv.get('effective_expression') or rv.get('base_expression') or '', record)
    carbon_credit = sum(s['count'] for s in heavy.get('reagent_sources', []) if s['element'] == 'C')
    if (atom['status'] == 'incomplete' and carbon_credit and not heavy['element_deficit'].get('C')
            and len(atom.get('missing_carbon_maps') or []) <= carbon_credit):
        atom.update(status='complete', element_deficit=heavy['element_deficit'],
                    reagent_sources=heavy['reagent_sources'])
    issues = list(step.get('context_issues') or [])
    if any(c.get('field') in ('STARTING_MATERIAL','REACTION_PRODUCT')
           for c in record.get('source_conflicts') or []):
        issues.append('source_structure_conflict')
    if record.get('record_kind') == 'procedure_template' or step.get('record_kind') == 'procedure_template':
        if record.get('scheme_template'):
            # A multi-step scheme-template extraction is a reference-route
            # step (one record per transformation), not a missing instance.
            pass
        else:
            issues.append('procedure_template_without_specific_instance')
    if step.get('boundary_status') != 'source_grounded':
        issues.append('step_boundary_unknown')
    if rv.get('status') not in ('passed', 'corrected') or rv.get('needs_human_review', True):
        issues.append('mapping_or_structure_requires_review')
    if atom['status'] != 'complete':
        issues.append('carbon_source_' + atom['status'])
    if heavy['status'] == 'incomplete':
        issues.append('heavy_atom_source_incomplete')
    if any(d.get('reason') == 'unbalanced_brackets' for d in record.get('ner_parse_diagnostics') or []):
        issues.append('unbalanced_brackets')
    for issue in (rv.get('field_constraints') or {}).get('issues') or []:
        issues.append('field_' + issue.get('field', '') + '_' + issue.get('type', 'unknown'))
    action_expected = record.get('record_kind', step.get('record_kind')) not in ('summary', 'title', 'non_procedure')
    if not record.get('action') and action_expected:
        issues.append('action_missing')
    if action_expected and (record.get('parse_status') or {}).get('action') not in (None, 'ok'):
        issues.append('action_parse_failed')
    if not action_expected:
        issues = [issue for issue in issues if issue != 'field_ACTION_missing']
    if (record.get('yield_info') or {}).get('kind') == 'unattributed':
        issues.append('yield_unattributed')
    if (record.get('yield_info') or {}).get('kind') == 'not_extracted':
        issues.append('source_yield_not_extracted')
    mass_status = (rv.get('mass_check') or {}).get('status')
    if mass_status == 'mismatch':
        issues.append('ms_mass_mismatch')
    if mass_status in ('unknown', 'not_reported_or_uninterpretable') and re.search(r'm\s*/\s*[ze]|(?:ESI|HR|LC|APCI)[- ]?MS', record.get('ms_source_text') or '', re.I):
        issues.append('ms_interpretation_unknown')
    stereo = record.get('stereochemistry') or {}
    if stereo.get('structure_coarsened'):
        issues.append('stereochemistry_coarsened')
    products = (rv.get('normalized_compounds') or {}).get('products') or []
    structure = 'supported' if products and all(p.get('status') == 'valid' for p in products) else 'unknown'
    if not products or any(p.get('status') != 'valid' for p in products):
        issues.append('product_structure_unavailable')
    for product in (record.get('ner') or {}).get('REACTION_PRODUCT') or []:
        if isinstance(product, dict):
            if product.get('structure_source') == 'adjacent_image_conflict' or product.get('image_product_candidates'):
                issues.append('image_structure_conflict')
                structure = 'conflict'
            if product.get('image_scope_unverified'):
                issues.append('image_identifier_scope_unverified')
            if stereo.get('kind') == 'absolute':
                smiles = str(product.get('smiles') or '')
                centers = sum(len(re.findall(r'[SR]', d, re.I)) for d in stereo.get('descriptors') or [])
                if len(re.findall(r'@{1,2}', smiles)) < max(1, centers):
                    issues.append('absolute_stereochemistry_unrepresented')
    evidence = record.get('field_evidence') or {}
    if not evidence:
        issues.append('field_evidence_missing')
    for field in ('STARTING_MATERIAL', 'REACTION_PRODUCT'):
        if not evidence.get(field) or any(e.get('status') == 'not_reported_or_not_extracted' for e in evidence.get(field, [])):
            issues.append(field.lower() + '_evidence_missing')
    contexts = [step.get('local_text', '')] + [r.get('text', '') for r in step.get('procedure_references', [])
                                            if r.get('relation') != 'precursor_preparation']
    reaction_context = ' '.join(reaction_condition_text(text) for text in contexts)
    ner = record.get('ner') or {}
    if re.search(r'-?\d+(?:\.\d+)?\s*(?:℃|°C)|室温|\broom temperature\b', _norm(reaction_context), re.I) and not ner.get('TEMPERATURE'):
        issues.append('source_temperature_not_extracted')
    if (re.search(r'\d+(?:\.\d+)?\s*(?:(?:hours?|hrs?)\b|分钟|小时)|过夜|\bovernight\b', reaction_context, re.I)
            or re.search(r'\d+(?:\.\d+)?\s*h\b', reaction_context)) and not ner.get('TIME'):
        issues.append('source_time_not_extracted')
    # Explicit local conditions must survive inherited program defaults.
    # Restrict to reaction text, leaving workup and analytical temperatures out.
    local_reaction = reaction_condition_text(step.get('local_text', ''), workup=True)
    def condition_keys(text, field):
        text = _condition_norm(text, field)
        if field == 'TEMPERATURE':
            return set(re.findall(r'(?:-?\d+(?:\.\d+)?-)?-?\d+(?:\.\d+)?℃|室温', text))
        return set(re.findall(r'(?:\d+(?:\.\d+)?-)?\d+(?:\.\d+)?(?:小时|分钟)|过夜|overnight', text))
    for field in ('TEMPERATURE', 'TIME'):
        expected = condition_keys(local_reaction, field)
        missing = sorted(expected - condition_keys(ner.get(field), field))
        if missing:
            issues.append('local_' + field.lower() + '_not_preserved')
        override_pattern = (r'温度(?:改为|改成|调整至)|temperature.{0,25}(?:changed|modified)'
                            if field == 'TEMPERATURE' else
                            r'时间(?:改为|改成|调整为)|time.{0,25}(?:changed|modified)')
        if re.search(override_pattern, local_reaction, re.I) and any(
                e.get('status') == 'inherited' for e in evidence.get(field, [])):
            issues.append('overridden_program_' + field.lower() + '_retained')
    # Inherited program conditions are part of this step's effective input.
    # When the local text is silent for a field, the referenced program's
    # reaction-phase conditions must survive in the response as well.
    for field in ('TEMPERATURE', 'TIME'):
        if condition_keys(local_reaction, field):
            continue
        expected = set()
        for ref in step.get('procedure_references') or []:
            if ref.get('relation') == 'precursor_preparation' or ref.get('status') != 'resolved':
                continue
            for block in ref.get('blocks') or []:
                expected |= condition_keys(reaction_condition_text(block.get('text', ''), workup=True), field)
        if expected - condition_keys(ner.get(field), field):
            issues.append('reference_' + field.lower() + '_not_preserved')
    # Product and reactant evidence are mandatory. Numeric conditions require
    # source attribution; null means not reported, never an invented value.
    for field in ('STARTING_MATERIAL', 'REACTION_PRODUCT', 'REAGENT_CATALYST', 'SOLVENT', 'TEMPERATURE', 'TIME', 'ACTION'):
        if any(e.get('status') == 'unattributed' for e in evidence.get(field, [])):
            issues.append(field.lower() + '_unattributed')
    if action_order_violated(record.get('action'), step.get('local_text', '')):
        issues.append('action_order_unverified')
    if rv.get('mass_check', {}).get('status') == 'mismatch':
        structure = 'conflict'
    if step.get('title'):
        names = [p.get('name') if isinstance(p, dict) else p
                 for p in (record.get('ner') or {}).get('REACTION_PRODUCT', [])]
        if not any(product_title_matches(step['title'], name) for name in names):
            issues.append('step_product_title_unverified')
    if structure == 'supported' and any(i in issues for i in ('reaction_product_unattributed', 'step_product_title_unverified', 'reaction_product_evidence_missing')):
        structure = 'unknown'
    if reaction_status is not None and reaction_status != 'ok':
        issues.append('reaction_record_requires_review')
    if chunk_status is not None and chunk_status != 'ok':
        issues.append('chunk_requires_review')
    issues = list(dict.fromkeys(issues))
    if re.search(r'tosylate|toluenesulfonate|甲苯磺酸', step.get('local_text', ''), re.I) and re.search(
            r'anhydrous|dehydrat|Dean[- ]Stark|脱水|无水', step.get('title', '') + step.get('local_text', ''), re.I):
        if not any(re.search(r'tosylate|toluenesulfonate|甲苯磺酸', str(p.get('name') if isinstance(p, dict) else p), re.I)
                   for p in ner.get('REACTION_PRODUCT') or []):
            issues.append('salt_form_unverified')
    issues = list(dict.fromkeys(issues))
    return {'version': QUALITY_VERSION, 'structure_status': structure, 'atom_source_check': atom,
            'heavy_atom_source_check': heavy,
            'record_completeness': 'incomplete' if issues else 'complete',
            'template_record': bool(record.get('scheme_template')),
            'blocking_issues': issues, 'admission_eligible': not issues}
