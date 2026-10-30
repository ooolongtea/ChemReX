"""Document-local procedure context and source-grounded chemical step units."""
import hashlib
import json
import re
from copy import deepcopy

CONTEXT_VERSION = '2026-10-16-isomer-latex-reference-repair'
_LABEL_ID = r'(?:\d+[A-Z]?|[A-Z]\d*)(?![A-Za-z0-9])'
_PROGRAM_ID = (r'(?:\d+[A-Z]?(?!\s*,\s*\d+\s*[-–—])|[A-Z]\d*)(?![A-Za-z0-9_—–-])'
               r"(?!['’′]\s*[-–—])"
               r'(?!\s*(?:mg|kg|g|mmol|mol|mL|L)\b|\s*%|\s*(?:天|小时|分钟|当量|克|毫升|毫摩尔))')
# MinerU can append a heading to the preceding introductory paragraph.
_GENERIC = re.compile(r'^(?:.*?[.!?。]\s+)?(?:\[\d+\]\s*)?(?:通用程序|General Procedures?)\s*('
                      + _LABEL_ID + r')(?:(?:\s*[:：—–-]\s*|\s+(?=using\s+Example\b))(.*))?\s*$', re.I | re.S)
_REFERENCE = re.compile(r'(?<![A-Za-z])(?:通用(?:程序|方案)|(?:General\s+)?Procedures?)\s*('
                        + _PROGRAM_ID + r'(?:\s*(?:和|及|、|,|and|then|&)\s*'
                        r'(?:通用(?:程序|方案)|(?:General\s+)?Procedures?)?\s*' + _PROGRAM_ID + r')*)', re.I)
_OPERATION = re.compile(r'加入|滴加|搅拌|加热|冷却|溶解|过滤|洗涤|萃取|浓缩|干燥|使.+得到|通用程序|\b(?:add(?:ed|ing)?|stirr?(?:ed|ing)?|heated|dissolved|treated|afforded|washed|filtered|extracted|concentrated|purified|dried|combined|obtained|diluted|cooled|general procedure)\b', re.I)
_ANALYTICAL = re.compile(r'NMR|UPLC|LC[- ]?MS|m\s*/\s*[ez]|\bMS\b', re.I)
_PARAGRAPH = re.compile(r'^\s*\[\d+\]\s*')
_NAMED_IDENTIFIER = (r'(?:\d+(?:\.\d+)*(?:\s*[-–—]\s*\d{1,3})*[A-Za-z]?'
                     r'|[A-Z]{1,4}(?:\s*[-–—]\s*)?\d+|[A-Z]{1,4}|[零〇一二两三四五六七八九十百]+)')
_NAMED_REFERENCE = re.compile(
    r'(?P<kind>(?:(?:(?:Reference|Production|Synthetic|Preparative)\s+)?Examples?|Intermediate|Compound|Boronate\s+Ester|Steps?|Preparation)(?![A-Za-z])|实施例|实例|制备实施例|制备例|中间体|中间物|步骤)\s*'
    r'(?P<identifier>'+_NAMED_IDENTIFIER+r')(?P<prime>[\'’′]?)(?![A-Za-z0-9])', re.I)
_REFERENCE_TYPE_WORD = re.compile(
    r'(?:(?:Reference|Production|Synthetic|Preparative)\s+)?'
    r'(?:Examples?|Intermediates?|Compounds?|Boronate\s+Esters?|Steps?|Preparations?'
    r'|实施例|实例|制备实施例|制备例|中间体|中间物|步骤)', re.I)
_ANALOGY = re.compile(r'in\s+analogy\s+to|analogous(?:ly)?|according\s+to\s+(?:the\s+)?procedure|'
                      r'according\s+to\s+(?:the\s+)?(?=(?:(?:Reference|Production|Synthetic|Preparative)\s+)?Examples?)|'
                      r'same\s+manner\s+as|'
                      r'(?:procedure|method)\s+similar\s+to|'
                      r'(?:procedure|method)\s+(?:described|reported|used|outlined)|as\s+described\s+in|as\s+(?:in|for)\b|'
                      r'following\s+(?:the\s+)?(?:general\s+)?procedure|参照|按照|根据|依据|类似于|'
                      r'采用(?:在)?(?=实施例|实例|步骤|中间体|制备例)|使用(?=实施例|实例|制备例)|'
                      r'按(?=实施例|实例|步骤|中间体|制备例)|使用针对(?=实例)|针对(?=实例)|'
                      r'(?:完全)?重复了(?=(?:上述)?(?:实施例|实例|步骤|中间体|制备例))|'
                      r'与(?:上述)?方法\s*[A-Z]\s*的步骤\s*\d+\s*相同', re.I)
# "与制备例 7 同样/相同" names the reference before its cue, so _ANALOGY's
# post-cue region cannot see it. Parse the whole fragment explicitly. The
# cue may follow the identifier across a descriptive tail ("与实施例8中第一
# 步至第三步的化学反应基本相同").
_SAME_MANNER = re.compile(
    r'与(?:上述|前述)?(?P<kind>实施例|实例|制备实施例|制备例|中间体|中间物)\s*'
    r'(?P<identifier>' + _NAMED_IDENTIFIER + r')[^。；;]{0,160}?(?:同样|相同|相似|一样|一致)', re.I)
_EXTERNAL_DOCUMENT = re.compile(r'\b(?:of|in)\s+(?:WO|U\.?S\.?|CN)\s*(?:Patent\s*(?:No\.?\s*)?)?\d', re.I)
_METHOD_HEADING = re.compile(r'^\s*(?:\[\d+\]\s*)?(?:方法|Method)\s*([A-Z])\s*[:：]\s*$', re.I)
_METHOD_STEP_REFERENCE = re.compile(r'与(?:上述)?方法\s*([A-Z])\s*的步骤\s*(\d+)\s*相同', re.I)
_GENERAL_SECTION = re.compile(r'^\s*(?:General Procedures?|通用程序|通用方法)\s*[:：]?\s*$', re.I)
_BARE_GENERAL_HEADING = re.compile(r'^\s*(?:\[\d+\]\s*)?(?:\d{1,3}\s+)?([A-Z]|\d+)\s*[:：.]\s*(.{4,180})\s*$')
_NUMERIC_TITLE = re.compile(r'[\d\s.\-–—。]+')
_NUMBERED_STEP_HEADING = re.compile(
    r'^(?P<number>\d{1,3})\s*[.、．]\s*'
    r'(?P<product>(?:中间体|化合物|终产物|产品|目标物)\s*'
    r'[A-Za-z0-9][A-Za-z0-9.\-–—]{0,23})(?:的合成|的制备)\s*[:：]?$')


def is_analogical_text(text):
    return bool(_ANALOGY.search(text or '') or _SAME_MANNER.search(text or ''))


def numbered_step_heading(text):
    return _NUMBERED_STEP_HEADING.fullmatch(_PARAGRAPH.sub('', text or '').strip())


def step_product_title(title):
    """Separate a printed step heading from its product identity."""
    value = _PARAGRAPH.sub('', title or '').strip()
    match = numbered_step_heading(value)
    return match['product'].strip() if match else value


def printed_step_number(step):
    match = numbered_step_heading(step.get('title'))
    if match:
        return int(match['number'])
    match = re.match(r'^(?:Step|步骤)\s*(\d+)\b', step.get('title') or '', re.I)
    return int(match[1]) if match else None


def referenced_step_numbers(text):
    """None means the whole procedure; () means an unsupported explicit selector."""
    from .segmenter import _number
    number = r'[零〇一二两三四五六七八九十百\d]+'
    chinese = re.search(r'第\s*(' + number + r')\s*(?=步|至|到|[~～\-—–、,，和及])', text)
    english = re.search(r'\bSteps?\s+(\d+)', text, re.I)
    if not chinese and not english:
        return () if re.search(r'第[^。；;]{0,12}步|\bSteps?\b', text, re.I) else None
    tail = text[(chinese or english).start():]
    # Stop at the end of the selector, not at subsequent modifications/amounts.
    if chinese:
        match = re.match(r'第\s*' + number + r'(?:\s*(?:步)?\s*(?:至|到|[~～\-—–、,，]|以及|和|及)\s*第?\s*' + number + r')*\s*步', tail)
        if not match:
            return ()
        values = [int(_number(n)) for n in re.findall(number, match[0])]
    else:
        match = re.match(r'Steps?\s+\d+(?:\s*(?:to|through|and|[\-–—,])\s*(?:Steps?\s*)?\d+)*', tail, re.I)
        values = [int(n) for n in re.findall(r'\d+', match[0])]
    tokens = list(re.finditer(number if chinese else r'\d+', match[0]))
    selected = values[:1]
    for i, value in enumerate(values[1:], 1):
        connector = match[0][tokens[i-1].end():tokens[i].start()]
        if re.search(r'至|到|[~～\-—–]|\b(?:to|through)\b', connector, re.I):
            if values[i-1] > value:
                return ()
            selected.extend(range(values[i-1] + 1, value + 1))
        else:
            selected.append(value)
    return tuple(dict.fromkeys(selected)) if all(n > 0 for n in selected) else ()


def procedure_citations(text, scope=None):
    """Use the procedure resolver's relation/scope rules for route expansion too."""
    keys, scopes = _reference_targets(text, scope)
    citations = []
    matches = list(_NAMED_REFERENCE.finditer(text))
    for key in keys:
        if scopes.get(key) in ('precursor_preparation', 'external_document'):
            continue
        selectors = []
        for mention in (m for m in matches if reference_key(m) == key):
            start = max((text.rfind(char, 0, mention.start()) for char in '。；;'), default=-1) + 1
            end_of_sentence = re.search(r'[。；;]', text[mention.end():])
            sentence_end = mention.end() + end_of_sentence.start() if end_of_sentence else len(text)
            local_keys, local_scopes = _reference_targets(text[start:sentence_end], scope)
            if key not in local_keys or local_scopes.get(key) in ('precursor_preparation', 'external_document'):
                continue
            # A template named inside a replacement clause identifies the old
            # material, not a second request for its entire procedure.
            clause_start = max(text.rfind(char, start, mention.start())
                               for char in '，,。；;\n') + 1
            if re.search(r'取代了?|替换(?:为|了)?', text[clause_start:mention.start()]):
                continue
            end = next((m.start() for m in matches if m.start() >= mention.end()), len(text))
            fragment = re.split(r'[。；;]', text[mention.end():end], maxsplit=1)[0]
            selector = referenced_step_numbers(mention[0] if key.startswith('step_') else fragment)
            selectors.append(selector)
        selector = (() if any(s == () for s in selectors) else None if None in selectors else
                    tuple(dict.fromkeys(n for s in selectors for n in s))) if selectors else None
        citations.append({'id': key, 'steps': selector, 'scope': scopes.get(key)})
    return citations


def scope_numbered_program(entry, numbers):
    """Restrict inherited evidence to printed steps; never use list positions."""
    if numbers is None:
        return entry
    selected, found, current = [], set(), None
    for block in entry.get('blocks') or []:
        for text in re.split(r'\n|(?=\[\d+\]\s*\d{1,3}\s*[.、．]\s*'
                             r'(?:中间体|化合物|终产物|产品|目标物))', block.get('text') or ''):
            heading = printed_step_number({'title': _PARAGRAPH.sub('', text).strip()})
            if heading is not None:
                current = heading
                found.add(heading)
            if current in numbers:
                selected.append(dict(block, text=text))
    if not numbers or not set(numbers).issubset(found):
        return dict(entry, status='missing', blocks=[], text='', requested_steps=list(numbers))
    return dict(entry, blocks=selected, text='\n'.join(b['text'] for b in selected),
                requested_steps=list(numbers))


def _heading_key(value):
    """Punctuation/spacing-insensitive comparison of a heading against a title."""
    return re.sub(r"[\s()（）〔〕\[\]【】.,;:、，。；：·'\"‘’“”]+", "", str(value or "")).casefold()


# A numbered heading naming a specific compound ("1. 化合物19的合成") is an
# instance heading, not a general procedure: the bare-heading heuristic must
# not read its "合成/制备" keyword as a procedure definition.
_SPECIFIC_INSTANCE_HEADING = re.compile(
    r'化合物\s*\d+|中间体\s*\d+|compound\s*\d+|intermediate\s*\d+', re.I)


def _bare_general_heading(text):
    match = _BARE_GENERAL_HEADING.fullmatch(text)
    return match if (match
                     and re.search(r'Procedure|Coupling|Reduction|Formation|Saponification|Displacement|'
        r'Oxidation|Synthesis|Bromination|Alkylation|Thiolysis|Conversion|还原|合成|偶联|水解|制备', match[2], re.I)
                     and not _SPECIFIC_INSTANCE_HEADING.search(match[2])) else None


def reference_key(match):
    from .segmenter import _number
    identifier = match['identifier']
    if identifier.lower() in ('of', 'the', 'and', 'for', 'in', 'to', 'on', 'but', 'as',
                               'was', 'were', 'is', 'with', 'from', 'using', 'then', 'also',
                               'add', 'stir', 'use', 'into', 'it', 'at', 'has', 'not', 'any', 'its'):
        return None
    kind = match['kind'].lower().replace(' ', '_')
    if kind.endswith('examples'):
        kind = kind[:-1]
    if kind == 'steps':
        kind = 'step'
    if kind == 'preparative_example':
        # Printed "Preparative Example" headings are parsed as plain Examples
        # (segmenter.py:673-676), so the citation must address that unit id.
        kind = 'example'
    kind = {'实施例': 'example', '实例': 'example', '制备例': 'example', '制备实施例': 'example',
            '中间体': 'intermediate', '中间物': 'intermediate', '步骤': 'step'}.get(kind, kind)
    numeric = re.fullmatch(r'(\d+(?:\.\d+)*)((?:\s*[-–—]\s*\d{1,3})*)([A-Za-z]?)', identifier)
    if numeric:
        number = numeric[1] if '.' in numeric[1] else str(int(numeric[1]))
        composite = ''.join('_' + str(int(part)) for part in re.findall(r'\d+', numeric[2]))
        ident = number + composite + numeric[3].lower()
    else:
        ident = str(_number(identifier))
    return kind + '_' + ident + ('_prime' if match['prime'] else '')


def block_parts(block):
    if block.get('type') == 'list':
        return [str(x).strip() for x in block.get('list_items', []) if str(x).strip()]
    from .segmenter import _captions, image_annotations
    if block.get('type') in ('image', 'chart'):
        return list(dict.fromkeys(_captions(block) + image_annotations(block)))
    text = str(block.get('text') or ' '.join(_captions(block))).strip()
    return [text] if text else []


def _procedure_ordered_indices(blocks):
    from .segmenter import _ordered_indices, heading_match
    ordered = _ordered_indices(blocks)
    section, scoped = False, set()
    for idx in ordered:
        for part in block_parts(blocks[idx]):
            if _GENERAL_SECTION.fullmatch(part):
                section = True
            elif heading_match(part) or re.match(r'^Examples?\s*$', part, re.I):
                section = False
        if section:
            scoped.add(idx)
    pages = {}
    for idx in ordered:
        pages.setdefault(blocks[idx].get('page_idx', blocks[idx].get('page_number', 1)-1), []).append(idx)
    result = []
    for indices in pages.values():
        bare = [i for i in indices if i in scoped and any(_bare_general_heading(p) for p in block_parts(blocks[i]))]
        boxes = [blocks[i].get('bbox') for i in indices]
        boxes = [b for b in boxes if isinstance(b, list) and len(b) == 4]
        if not bare or len(boxes) < 4:
            result.extend(indices)
            continue
        midpoint = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2
        def key(i):
            box = blocks[i].get('bbox')
            return (int((box[0]+box[2])/2 >= midpoint), box[1], box[0]) if isinstance(box,list) and len(box)==4 else (2,i,0)
        result.extend(sorted(indices, key=key))
    return result


def procedure_index(blocks, source):
    from .segmenter import heading_match, segment_blocks
    index = {}
    current = None
    general_section = False
    ordered = _procedure_ordered_indices(blocks)
    for offset, idx in enumerate(ordered):
        block = blocks[idx]
        for part in block_parts(block):
            if _GENERAL_SECTION.fullmatch(part):
                general_section = True
                current = None
                continue
            if heading_match(part) or re.match(r'^Examples?\s*$', part, re.I):
                general_section = False
            bare = _bare_general_heading(part) if general_section else None
            if bare:
                current = {'id': bare[1], 'source': source, 'parts': [], 'blocks': [
                    dict(block_idx=idx, page_idx=block.get('page_idx'), bbox=block.get('bbox'), text=part)]}
                index.setdefault(bare[1], []).append(current)
                continue
            match = _GENERIC.fullmatch(part)
            if match:
                key = match.group(1).upper()
                heading = list(re.finditer(r'通用程序|General Procedures?', part[:match.start(1)], re.I))[-1]
                prefix = part[:heading.start()].strip()
                if current and prefix:
                    # The previous method's last paragraph can share a block
                    # with the next heading. Do not move it into the next method.
                    current['parts'].append(prefix)
                    current['blocks'].append({'block_idx': idx, 'page_idx': block.get('page_idx'),
                                              'bbox': block.get('bbox'), 'text': prefix})
                current = {'id': key, 'source': source, 'parts': [], 'blocks': []}
                index.setdefault(key, []).append(current)
                if match.group(2):
                    current['parts'].append(match.group(2))
                current['blocks'].append({'block_idx': idx, 'page_idx': block.get('page_idx'), 'bbox': block.get('bbox'),
                                          'text': part[heading.start():]})
            elif current:
                if block.get('type') in ('header', 'footer', 'page_number'):
                    continue
                following = next((p for i in ordered[offset + 1:offset + 3] for p in block_parts(blocks[i])), '')
                if (heading_match(part) or (not general_section and _product_heading(part)
                        and (_OPERATION.search(following) or _ANALOGY.search(following))) or re.search(r'权利要求|具体实施方式|^Examples?\b|'
                        r'^(?:Chemical\s+Analysis|Analytical\s+Procedures|Characterization|化学分析|分析方法|表征)',
                        _PARAGRAPH.sub('', part), re.I)):
                    current = None
                else:
                    current['parts'].append(part)
                    current['blocks'].append({'block_idx': idx, 'page_idx': block.get('page_idx'), 'bbox': block.get('bbox'), 'text': part})
    for entries in index.values():
        for entry in entries:
            entry['text'] = '\n'.join(entry.pop('parts'))
    # Index named procedures separately from general procedures. Duplicate
    # local Step numbers retain their parent rather than overwriting each other.
    parent = None
    method = None
    method_at_block = {}
    for i, block in enumerate(blocks):
        for part in block_parts(block):
            heading = heading_match(part)
            if heading and (heading['kind'].lower().endswith('example') or heading['kind'].lower() in
                            ('compound', 'boronate ester', 'intermediate', '中间物', '中间体', '实施例', '实例')):
                method = None
            marker = _METHOD_HEADING.fullmatch(part)
            if marker:
                method = marker[1].lower()
        method_at_block[i] = method
    for chunk in segment_blocks(blocks, source):
        if chunk['kind'] in ('example', 'compound', 'boronate_ester', 'intermediate'):
            parent = chunk['reference_id']
        # A/B are alternatives inside a parent, rather than document-wide
        # Step numbers. Use the marker preceding the step's actual heading.
        heading_idx = (chunk.get('title_source') or {}).get('block_idx', chunk['block_start'])
        method = method_at_block.get(heading_idx)
        key = chunk['reference_id']
        if not (_OPERATION.search(chunk['text']) or _ANALOGY.search(chunk['text'])):
            # Scheme labels and chemical title fragments are evidence for a
            # later procedure, not alternative definitions of that procedure.
            continue
        evidence = [{'block_idx': i, 'page_idx': blocks[i].get('page_idx'),
                     'bbox': blocks[i].get('bbox'), 'text': '\n'.join(block_parts(blocks[i]))} for i in chunk['block_indices']]
        for part in evidence:
            original = part['text']
            if re.search(r'\S[\s\S]+?\bStep\s+\d+[A-Za-z]?\s*[:：]', original, re.I):
                slices = [p for p in chunk.get('text_parts', []) if p in original]
                heading = chunk.get('title_source') or {}
                if heading.get('block_idx') == part['block_idx'] and heading.get('text'):
                    slices.insert(0, heading['text'])
                part['text'] = '\n'.join(dict.fromkeys(slices))
        index.setdefault(key, []).append({'id': key, 'kind': chunk['kind'],
            'chunk_id': chunk['id'], 'heading_type': chunk.get('heading_type'),
            'parent_id': parent if chunk['kind'] == 'step' else None,
            'method_variant': method if chunk['kind'] == 'step' else None,
            'source': source, 'text': chunk['text'], 'blocks': evidence})
    # An Example heading may only introduce child Steps. Preserve those
    # methods as alternatives under the parent citation; concatenating them
    # would silently turn a route into a single reaction.
    parents = {}
    for entries in list(index.values()):
        for entry in entries:
            if entry.get('parent_id'):
                parents.setdefault(entry['parent_id'], []).append(entry)
    for key, children in parents.items():
        if key not in index:
            index[key] = [dict(child, id=key, referenced_step_id=child['id'])
                          for child in children]
    return index


def _precursor_reference(text, cue, region):
    prefix = text[max(0, cue.start()-240):cue.start()]
    # Parenthetical material provenance, e.g. substrate (... prepared as in X).
    if (re.search(r'\([^()]{0,120}\bprepared\s*$', prefix, re.I)
            and re.search(r'\b(?:Example|Intermediate)\s+\d+[^()]*\)', region, re.I)):
        return True
    # Standalone preparation of an input material is provenance, whereas
    # "the product was prepared as in X" remains the current synthesis method.
    sentence = re.split(r'(?<!\d)\.(?:\s|$)|[;。\n]', prefix)[-1]
    prepared = r'\b(?:was|were|had\s+been)\s+(?:(?:previously|already)\s+)?prepared\s*$'
    material = r'\b(?:starting\s+materials?|substrates?|reactants?|precursors?|reagents?|catalysts?|solvents?)\b'
    subject = r'^\s*(?:(?:the|a|an|this|these)\s+)?'
    if re.search(subject + material + r'[^.;。]{0,100}' + prepared, sentence, re.I):
        return True
    from .segmenter import ASSAY_MARKER
    return bool(ASSAY_MARKER.search(text) and re.search(
        subject + r'\b(?:association\s+complexes|siRNA|vesicles|nanoparticles)\b[^.;。]{0,100}'
        + prepared, sentence, re.I))


def _reference_targets(text, scope=None, subs=None):
    keys = []
    scopes = {}
    for match in _METHOD_STEP_REFERENCE.finditer(text):
        key = 'step_' + str(int(match[2]))
        if key not in keys:
            keys.append(key)
        scopes[key] = (scope or '') + '::method_' + match[1].lower()
    for match in _REFERENCE.finditer(text):
        if not re.match(r'通用(?:程序|方案)|General\s+', match[0], re.I) and re.fullmatch(r'[a-z]', match[1]):
            # "local procedure a" may use the indefinite article, not label A.
            continue
        for key in re.findall(r'(?<![A-Za-z0-9])' + _PROGRAM_ID, match.group(1), re.I):
            key = key.upper()
            if key not in keys:
                keys.append(key)
            if _EXTERNAL_DOCUMENT.match(text[match.end():].lstrip()):
                scopes[key] = 'external_document'
    for cue in _ANALOGY.finditer(text):
        region = text[cue.end():cue.end() + 240]
        # A full stop ends the citation, but not Step 1.3's numeric identifier.
        region = re.split(r'(?<!\d)\.(?:\s|$)|[;。\n]', region, maxsplit=1)[0]
        named = list(_NAMED_REFERENCE.finditer(region))
        citations = named[:1]
        for left, right in zip(named, named[1:]):
            between = region[left.end():right.start()]
            if re.fullmatch(r'\s*(?:[,，:/&、]|and|和|及|的|中(?:的)?|of|in|for)?\s*', between, re.I):
                citations.append(right)
            else:
                break
        named = citations
        explicit_examples = {reference_key(m) for m in named if 'example' in m['kind'].lower() or m['kind'].lower() in ('compound', 'boronate ester', 'intermediate') or m['kind'] in ('实施例', '实例', '中间物', '中间体')}
        explicit_examples.discard(None)
        external = bool(_EXTERNAL_DOCUMENT.search(region))
        citation_scope = 'external_document' if external else next(iter(explicit_examples)) if len(explicit_examples) == 1 else scope
        precursor = _precursor_reference(text, cue, region)
        if precursor:
            citation_scope = 'precursor_preparation'
        containers = set()
        # "Example 4, Step 3" cites that step, not the entire Example route.
        for left, right in zip(named, named[1:]):
            between = region[left.end():right.start()]
            if ((reference_key(left) or '').startswith(('example_', 'compound_', 'boronate_ester_', 'intermediate_')) and
                    (reference_key(right) or '').startswith('step_') and
                    re.fullmatch(r'\s*(?:的|中(?:的)?|[,，:]|\s)?\s*', between)):
                containers.add(reference_key(left))
            if ((reference_key(left) or '').startswith('step_') and
                    (reference_key(right) or '').startswith(('example_', 'compound_', 'boronate_ester_', 'intermediate_')) and
                    re.fullmatch(r'\s*(?:of|in|for)\s*', between, re.I)):
                containers.add(reference_key(right))
        for match in named:
            key = reference_key(match)
            if key and key.startswith(('example_', 'compound_', 'boronate_ester_', 'intermediate_')) and not external and not precursor:
                citation_scope = key
                if subs is not None:
                    # match offsets are relative to the analogy region.
                    _capture_sub_designators(region, match.end(), key, subs)
            if key and key not in keys and key not in containers:
                keys.append(key)
                scopes[key] = citation_scope
            elif key in keys and scopes.get(key) == 'precursor_preparation' and not precursor:
                # The same example may also be explicitly used as the current
                # method. A prior material citation must not hide that use.
                scopes[key] = citation_scope
            if key and external:
                scopes[key] = 'external_document'
            tail = region[match.end():]
            while key:
                extra = re.match(r'\s*(?:and|[,/]|和|及|、|&)\s*('+_NAMED_IDENTIFIER+r')(?![A-Za-z0-9—–-])'
                    r'(?!\s*(?:mg|kg|g|mmol|mol|mL|L|hours?|hrs?|h|minutes?|min|days?)\b|\s*[％%℃°]|\s*(?:天|小时|分钟|克|毫升))', tail, re.I)
                if not extra:
                    break
                if _REFERENCE_TYPE_WORD.match(extra[1].strip()):
                    # "Example 2, Step 1": a full reference type follows the
                    # separator — the main parser owns it, not appended numbers.
                    break
                extra_key = reference_key({'kind': match['kind'], 'identifier': extra[1], 'prime': ''})
                if extra_key and extra_key not in keys:
                    keys.append(extra_key)
                    scopes[extra_key] = citation_scope
                tail = tail[extra.end():]
    for match in _SAME_MANNER.finditer(text):
        # "分别与实施例8第二步以及实施例22第四步相同" cites both methods.
        for named in _NAMED_REFERENCE.finditer(match[0]):
            key = reference_key(named)
            if key and key not in keys:
                keys.append(key)
                scopes[key] = scope
    return keys, scopes


_SUB_LETTER = r'(?:iii|ii|iv|viii|vii|vi|ix|[a-z])'
_SUB_DESIGNATOR = re.compile(r'\s*[\(（]?\s*(' + _SUB_LETTER + r')\s*[\)）](?![A-Za-z0-9])', re.I)
_SUB_HEADING = re.compile(r'^\s*[\(（]?\s*(' + _SUB_LETTER + r')\s*[\)）.．、]\s*\S', re.I)


def _capture_sub_designators(text, pos, key, subs):
    """Record the sub-step letters of a citation like "Example 34(d)" or
    "Example 7 (a) and (b)": the referenced program's other sub-steps are
    not this step's effective input."""
    found, connector = [], re.compile(r'\s*(?:and|和|及|、|,|，)\s*')
    while True:
        match = _SUB_DESIGNATOR.match(text, pos)
        if not match:
            break
        found.append(match.group(1).lower())
        pos = match.end()
        bridge = connector.match(text, pos)
        if not bridge:
            break
        pos = bridge.end()
    if found:
        subs.setdefault(key, set()).update(found)


def _scope_sub_example(entry, letters):
    """Restrict a resolved program to the cited sub-step(s). Without this,
    a step citing "Example 34(d)" inherits the conditions of the uncited
    siblings 34(a)-(c) as well."""
    if not letters or not entry.get('blocks'):
        return entry
    blocks = entry['blocks']
    marks = [(i, m.group(1).lower()) for i, block in enumerate(blocks)
             if (m := _SUB_HEADING.match(str(block.get('text') or '')))]
    cited = [mark for mark in marks if mark[1] in letters]
    if not cited:
        return entry
    start = cited[0][0]
    following = marks.index(cited[-1]) + 1
    end = marks[following][0] if following < len(marks) else len(blocks)
    scoped = blocks[start:end]
    return dict(entry, blocks=deepcopy(scoped),
                text='\n'.join(str(block.get('text') or '') for block in scoped))


_COMPOSITE_REFERENCE = re.compile(
    r'(example|intermediate|compound|boronate_ester)_\d+(?:_\d+)+$')


def _composite_parent_keys(key):
    """Map "141-1"/"488-1" citations to the units their printed headings parsed as.

    Heading parsing keeps only the first number (segmenter.py:14-19), so the
    exact composite id is absent from the procedure index; the parent unit is
    the only source-grounded target that exists for the printed heading.
    """
    keys = []
    current = key or ''
    while _COMPOSITE_REFERENCE.fullmatch(current):
        current = current.rsplit('_', 1)[0]
        keys.append(current)
    return keys


def _composite_heading_evidence(entry, key):
    """True when the unit's source heading carries the cited composite number.

    "制备例 141-1" and "实施例 141" can both have parsed to example_141; only
    the printed heading block "制备例 141 - 1" identifies the cited sub-number.
    """
    match = _COMPOSITE_REFERENCE.fullmatch(key or '')
    if not match:
        return False
    tail = key[len(match[1]) + 1:]
    printed = r'\s*[-–—]\s*'.join(re.findall(r'\d+', tail))
    if not printed:
        return False
    pattern = re.compile(r'(?<!\d)' + printed + r'(?!\d)')
    texts = [str(block.get('text') or '') for block in entry.get('blocks') or []
             if isinstance(block, dict)]
    texts.append(str(entry.get('text') or ''))
    return any(pattern.search(text) for text in texts)


def resolve_programs(text, index, scope=None):
    subs = {}
    keys, scopes = _reference_targets(text, scope, subs)
    selections = {c['id']: c['steps'] for c in procedure_citations(text, scope)}
    refs, issues, visited = [], [], set()
    def resolve(key, trail, local_scope, targets, subs, selections):
        key_scope = targets.get(key, local_scope)
        identity = (key, key_scope)
        if identity in trail:
            refs.append({'id': key, 'status': 'cycle', 'source': None, 'blocks': []})
            issues.append('procedure_reference_cycle')
            return
        if identity in visited:
            return
        visited.add(identity)
        candidates = [] if key_scope == 'external_document' else index.get(key, [])
        if key_scope != 'external_document' and not candidates:
            for parent_key in _composite_parent_keys(key):
                candidates = index.get(parent_key, [])
                if not candidates:
                    continue
                if len({entry['text'] for entry in candidates if entry['text'].strip()}) > 1:
                    scoped = [entry for entry in candidates
                              if _composite_heading_evidence(entry, key)]
                    if scoped:
                        candidates = scoped
                break
        if key_scope and '::method_' in key_scope:
            parent_scope, variant = key_scope.rsplit('::method_', 1)
            candidates = [c for c in candidates if c.get('parent_id') == parent_scope
                          and c.get('method_variant') == variant]
            key_scope = parent_scope
        if key_scope and (key.startswith('step_') or any(c.get('parent_id') == key_scope for c in candidates)):
            candidates = [c for c in candidates if c.get('parent_id') == key_scope]
        distinct = {entry['text'] for entry in candidates if entry['text'].strip()}
        if key_scope == 'precursor_preparation':
            refs.append(dict(candidates[0], status='resolved', relation='precursor_preparation') if len(distinct) == 1
                        else dict(id=key, status='missing' if not distinct else 'ambiguous',
                                  relation='precursor_preparation', source=None, blocks=[]))
            return
        alternative_methods = len(distinct) == 1 and len(set(re.findall(
            r'(?:方法|\bMethod)\s*([A-Z])\s*[:：]', candidates[0]['text'], re.I))) > 1
        if len(distinct) == 1 and re.search(r'\bAlternatively\b|另一种方法', candidates[0]['text'], re.I):
            from .segmenter import _de_latex
            options = re.split(r'\bAlternatively\b|另一种方法', _de_latex(candidates[0]['text']), flags=re.I)
            # Two explicitly alternative reaction conditions are not a
            # sequential temperature/time program. Workup choices alone do
            # not trigger this diagnostic.
            condition = r'\d+(?:\.\d+)?\s*(?:℃|°\s*C|hours?\b|hrs?\b|minutes?\b|min\b|小时|分钟)'
            alternative_methods = alternative_methods or sum(bool(re.search(condition, p, re.I)) for p in options) > 1
        if len(distinct) == 1 and not alternative_methods:
            entry = _scope_sub_example(dict(candidates[0], status='resolved'), subs.get(key))
            if not key.startswith('step_'):
                entry = scope_numbered_program(entry, selections.get(key))
            refs.append(entry)
            if entry['status'] != 'resolved':
                issues.append('procedure_reference_missing')
                return
            child_scope = entry.get('parent_id') or (key if 'example_' in key else key_scope)
            child_subs = {}
            nested, child_targets = _reference_targets(entry['text'], child_scope, child_subs)
            child_selections = {c['id']: c['steps'] for c in procedure_citations(entry['text'], child_scope)}
            for child in nested:
                resolve(child, trail + [identity], child_scope, child_targets, child_subs, child_selections)
        else:
            status = 'missing' if not distinct else 'ambiguous'
            refs.append({'id': key, 'status': status, 'source': None, 'blocks': [],
                         'candidates': deepcopy(candidates)})
            issues.append('procedure_reference_' + status)
    for key in keys:
        resolve(key, [], scope, scopes, subs, selections)
    return refs, issues


def resolve_program_keys(text):
    # Discovery without an index reuses the exact reference grammar.
    keys, _ = _reference_targets(text)
    return keys, []


def _product_heading(part):
    value = _PARAGRAPH.sub('', part).strip()
    if numbered_step_heading(value):
        return True
    if re.fullmatch(r'化合物\s*[A-Za-z0-9]+(?:[-.]\d+)*(?:的)?制备', value):
        return True
    if not value or len(value) < 12 or len(value) > 500:
        return False
    if re.match(r'To\s+', value, re.I):
        # Cross-page reagent sentences can end before their mmol unit. They
        # still belong to the procedure, not to a new product heading.
        return False
    if (_OPERATION.search(value) or _ANALYTICAL.search(value)
            or is_analogical_text(value)):
        return False
    if re.search(r'\b(?:solution|mixture|residue|filtrate|organic layer|aqueous|in vacuo|over Na|the resulting)\b|[.!?](?:\s|$)', value, re.I):
        return False
    if re.search(r'\d\s*(?:mg|mmol|mL|℃|hours?|分钟|小时)\b|[。；;]', value, re.I):
        return False
    return bool(re.search(r'苯|甲|乙|氨|吡|酰|酸|腈|环|基|酯|fluoro|chloro|bromo|amino|amide|nitrile|methyl|phenyl|pyr|\bStep\b|\bPreparation of\b|\bSalt\b|\bForm\s+[A-Z0-9]+|步骤', value, re.I))


_ACQUISITION = re.compile(
    r'\b(?:afford(?:ed|s|ing)?|yield(?:ed|s|ing)?|obtain(?:ed|s|ing)?|provid(?:e|ed|es|ing)|isolat(?:e|ed|es|ing))\b'
    r'|\bto\s+(?:give|yield)\b|得到|制得|获得|生成', re.I)


def _caption_product_evidence(parts, index, title):
    """A scheme caption may head a step only with source-grounded evidence.

    MinerU emits scheme product labels and starting-material labels alike as
    image annotations. A caption becomes a step title only when the following
    procedure in this chunk reports obtaining that same compound (US8598181
    ex.18); reactant labels and repeated panel captions stay body text.
    """
    key = _heading_key(title)
    if len(key) < 8:
        return False
    following, size = [], 0
    for part in parts[index + 1:]:
        if part.get('source_role') == 'image_annotation':
            break
        text = str(part.get('text') or '')
        following.append(text)
        size += len(text)
        if size > 1600:
            break
    region = '\n'.join(following)
    if not region:
        return False
    for sentence in re.split(r'(?<=[.!?。；;])\s*', region):
        if key not in _heading_key(sentence):
            continue
        cues = list(_ACQUISITION.finditer(sentence))
        if not cues:
            continue
        # "to afford <name>"/"得到 <name>" put the compound after the verb.
        if key in _heading_key(sentence[cues[-1].start():]):
            return True
        # "<name> was obtained" names the product before the verb; only an
        # exact leading name (not a reagent sentence such as "a solution of X")
        # counts as the transformation's product.
        if _heading_key(sentence[:cues[0].start()]) == key:
            return True
    return False


def incomplete_title(title):
    value = re.sub(r'\$|\\(?:mathrm|text|mathbf)', '', title or '')
    # Scientific names may contain balanced brackets, but an open trailing
    # bond or an unclosed bracket cannot be a complete product identity.
    return bool(re.search(r'[-–—({\[]\s*$', value) or any(value.count(a) > value.count(b)
                for a, b in [('(', ')'), ('[', ']'), ('{', '}')]))


def _join_title(left, right):
    left, right = left.strip(), _PARAGRAPH.sub('', right).strip()
    return left + ('' if left.endswith(('-', '[', '(', '{')) else ' ') + right


def specific_target_title(title):
    match = re.fullmatch(r'Procedure\s+for\s+(?:the\s+)?(?:Preparation|Synthesis)\s+of\s+'
        r'((?:Example|Compound)\s+(?:\d+[A-Za-z]?|[A-Z]{1,4}-?\d+))\s*[:：]?\s*', title or '', re.I)
    return match[1].strip() if match else title


def is_template_definition(kind, title, parts):
    definition_heading = any(_GENERIC.fullmatch(p['text'].strip()) or _bare_general_heading(p['text'].strip()) for p in parts)
    if kind == 'scheme':
        # A collected scheme narrative is a template by construction.
        return True
    if title or (kind != 'fallback' and not definition_heading):
        return False
    text = '\n'.join(p['text'] for p in parts)
    if definition_heading:
        # A "General Procedure" can also report a finite worked synthesis.
        # A measured, locanted chemical name is evidence of such an instance;
        # equivalents of an unspecified acid/amine/derivative are not.
        measured_name = re.search(r'\b\d+[A-Za-z]?(?:[-,][^\n;。]{3,180}?)'
            r'\(\s*\d+(?:\.\d+)?\s*(?:mg|g|mmol)\b', text, re.I)
        reported_product = re.search(r'\b(?:yield(?:ed)?|afford(?:ed)?|to give|to leave|isolated)\b|得到|收率', text, re.I)
        return not bool(measured_name and reported_product)
    generic = re.search(r'本发明的化合物|compounds?\s+of\s+(?:the\s+|this\s+|present\s+)?invention|'
                        r'一般方法|通用方法|\bgeneral method\b|\bsome embodiments\b', text, re.I)
    actual_amount = re.search(r'\d+(?:\.\d+)?\s*(?:mg|g|kg|mmol|mol|mL|毫克|毫升)\b', text, re.I)
    return bool(generic and not actual_amount)


def split_steps(chunk):
    from .segmenter import classify_process
    parts = chunk.get('source_parts') or [dict(text=p, block_idx=None, page_idx=None)
        for p in chunk.get('text_parts', [])]
    if not parts:
        parts = [dict(text=chunk.get('procedure') or chunk['text'], block_idx=None, page_idx=None)]
    groups, pending = [], []
    title = chunk.get('title', '')
    title_source = deepcopy(chunk.get('title_source')) or {'text': title, 'block_idx': chunk.get('block_start'),
                    'page_idx': chunk.get('page_start', 1) - 1, 'source': chunk.get('source')}
    title_source.setdefault('blocks', [dict(title_source)] if title else [])
    separated = []
    for part in parts:
        match = _GENERIC.fullmatch(part['text'].strip())
        if match:
            heading = list(re.finditer(r'通用程序|General Procedures?', part['text'][:match.start(1)], re.I))[-1]
            prefix = part['text'][:heading.start()].strip()
            if prefix:
                separated.append(dict(part, text=prefix))
            separated.append(dict(part, text=part['text'][heading.start():].strip(), source_role='procedure_definition'))
        else:
            separated.append(part)
    parts = separated
    for i, part in enumerate(parts):
        text = part['text']
        if part.get('source_role') == 'procedure_definition':
            prior_text = '\n'.join(p['text'] for p in pending)
            prior_definition = any(p.get('source_role') == 'procedure_definition' for p in pending)
            prior_experiment = bool(_OPERATION.search(prior_text) and re.search(
                r'\d+(?:\.\d+)?\s*(?:mg|g|mmol|毫克|毫升)\b', prior_text, re.I))
            if pending and (title or prior_definition or prior_experiment):
                groups.append((title, title_source, pending))
                pending = []
            title = ''
            title_source = deepcopy(part)
            title_source['blocks'] = [deepcopy(part)]
            pending.append(part)
            continue
        remaining = next((p['text'] for p in parts[i + 1:]
                          if _PARAGRAPH.sub('', p['text']).strip()), '')
        operation_seen = any(_OPERATION.search(p['text']) or is_analogical_text(p['text'])
                             for p in pending)
        heading = _product_heading(text)
        if part.get('source_role') == 'image_annotation':
            # A scheme annotation can carry the printed product-name heading
            # of the transformation that follows it (US8598181 ex.18). An
            # identical repeat of the current title stays body text, and the
            # following procedure must report obtaining the same compound;
            # reactant labels and stray panel captions never start a step.
            is_title = (heading and _heading_key(text) != _heading_key(title)
                        and _caption_product_evidence(parts, i, text))
            numeric_takeover = False
        else:
            is_title = heading
            # A purely numeric step label ("1", "488-1") carries no chemical
            # identity: a specific compound-name part becomes the step title
            # even when the following sentence has no operation verb.
            numeric_takeover = heading and bool(_NUMERIC_TITLE.fullmatch(title.strip()))
        continuation = (title and part.get('source_role') != 'image_annotation' and not operation_seen and not _OPERATION.search(text)
                        and not _ANALYTICAL.search(text) and len(text) < 500
                        and not re.match(r'\s*(?:Step\s+[A-Za-z0-9]|步骤|Preparation\b|Synthesis\b)', text, re.I)
                        and (incomplete_title(title) or re.fullmatch(
                            r'benzyl|methyl|ethyl|tert[- ]butyl', title, re.I)))
        if continuation:
            # Consecutive chemical title fragments precede the first operation;
            # replacing the title here used to delete Benzyl/Methyl and rings.
            if _PARAGRAPH.sub('', text).strip() != title:
                title = _join_title(title, text)
                title_source['blocks'].append(deepcopy(part))
                title_source['text'] = title
        elif is_title and (numbered_step_heading(text) or numeric_takeover
                           or _OPERATION.search(remaining) or is_analogical_text(remaining)):
            if operation_seen or (numbered_step_heading(text) and numbered_step_heading(title)):
                groups.append((title, title_source, pending))
                pending = []
            title = _PARAGRAPH.sub('', text).strip()
            title_source = deepcopy(part)
            title_source['blocks'] = [deepcopy(part)]
        pending.append(part)
    if pending:
        groups.append((title, title_source, pending))
    if not groups:
        groups = [(title, title_source, parts)]
    units = []
    for number, (step_title, title_source, source_parts) in enumerate(groups, 1):
        step_title = specific_target_title(step_title)
        local = '\n'.join(p['text'] for p in source_parts)
        template = is_template_definition(chunk.get('kind'), step_title, source_parts)
        reference_text = '\n'.join(p['text'] for p in source_parts
                                   if not _GENERIC.fullmatch(p['text'].strip()))
        refs, issues = resolve_programs(reference_text, chunk.get('procedure_index', {}), scope=chunk.get('parent_example_id'))
        if incomplete_title(step_title):
            issues.append('product_title_incomplete')
        methods = [r for r in refs if r.get('relation') != 'precursor_preparation']
        telescoped = len(methods) > 1
        if telescoped:
            issues.append('telescoped_procedures_without_reported_intermediates')
        isolation_count = len(re.findall(r'得到[^。\n]{0,30}(?:标题化合物|产物)|'
            r'(?:afforded|yielded|obtained)\s+(?:the\s+)?(?:title|desired)|'
            r'\bto\s+(?:give|yield)\s+(?!(?:a\s+)?(?:solution|mixture|suspension)\b)', local, re.I))
        ambiguous = not template and len(groups) == 1 and isolation_count > 1
        process_kind = classify_process(step_title, (step_title + '\n' + local).strip())
        if (not template and process_kind == 'unverified' and methods
                and all(r['status'] == 'resolved' for r in methods)
                and (_ANALOGY.search(local) or _REFERENCE.search(local))):
            inherited_kinds = {classify_process('', r['text']) for r in methods}
            if 'synthesis' in inherited_kinds and not inherited_kinds.intersection(
                    {'formulation', 'assay', 'analysis', 'purification'}):
                process_kind = 'synthesis'
        units.append({'step_id': f"{chunk['id']}:step:{number}", 'step_index': number,
                      'record_kind': 'procedure_template' if template else 'reaction_candidate',
                      'process_kind': process_kind,
                      'title': step_title, 'text': (step_title + '\n' + local).strip(),
                      'title_source': title_source,
                      'local_text': local, 'source_parts': deepcopy(source_parts),
                      'procedure_references': refs, 'context_issues': issues,
                      'unit_kind': 'telescoped' if telescoped else 'single_step',
                      'needs_step_planning': ambiguous,
                      'boundary_status': 'unknown' if ambiguous else 'source_grounded' if chunk.get('source_parts') else 'unknown'})
    return units


REFERENCE_MARKER = '\n\n引用程序'


def procedure_body(text):
    """The step-local text: everything before the appended reference section."""
    return text.split(REFERENCE_MARKER, 1)[0]


def extraction_text(step):
    refs = [r for r in step['procedure_references'] if r['status'] == 'resolved' and r.get('relation') != 'precursor_preparation']
    if not refs:
        return step['text']
    return step['text'] + '\n\n引用程序（仅用于本步骤，局部明确修改优先；通用量不替代本步实际量）：\n' + '\n'.join(
        f"程序 {r['id']} [blocks {','.join(str(b['block_idx']) for b in r['blocks'])}]:\n{r['text']}" for r in refs)


def attach_context(chunks, blocks, source):
    from .segmenter import chunk_fingerprint
    index = procedure_index(blocks, source)
    # Prompt contents participate in cache identity, including version selection.
    from pathlib import Path
    prompts = Path(__file__).parents[1] / 'extraction' / 'prompts'
    digest = hashlib.sha256()
    for path in sorted(prompts.rglob('*')):
        if path.suffix in ('.md', '.json'):
            digest.update(path.relative_to(prompts).as_posix().encode())
            digest.update(path.read_bytes())
    for path in sorted(Path(__file__).parents[1].rglob('*.py')):
        digest.update(path.relative_to(Path(__file__).parents[1]).as_posix().encode())
        digest.update(path.read_bytes())
    title_context = None
    from .source_conflicts import compound_evidence_for_source
    compound_evidence = compound_evidence_for_source(source)
    for chunk in chunks:
        if compound_evidence:
            chunk['source_compound_evidence'] = compound_evidence
        source_parts = deepcopy(chunk.get('source_parts') or []) if chunk.get('kind') == 'scheme' else []
        for text in ([] if source_parts else chunk.get('text_parts', [])):
            matches = [(idx, blocks[idx]) for idx in chunk.get('block_indices', [])
                       if 0 <= idx < len(blocks) and any(text == p or text in p for p in block_parts(blocks[idx]))]
            idx, block = matches[0] if matches else (None, {})
            source_parts.append({'text': text, 'block_idx': idx, 'page_idx': block.get('page_idx'),
                                 'bbox': block.get('bbox'), 'text_level': block.get('text_level'),
                                 'source_role': 'image_annotation' if block.get('type') in ('image', 'chart') else 'procedure'})
        chunk['source_parts'] = source_parts
        if chunk.get('kind') in ('example', 'compound'):
            key = chunk.get('reference_id')
            if title_context and title_context['reference_id'] != key:
                title_context = None
            if (title_context and not chunk.get('title') and source_parts
                    and _OPERATION.search(source_parts[0]['text'])):
                chunk['title'] = title_context['title']
                chunk['title_source'] = deepcopy(title_context['title_source'])
                chunk['block_indices'] = sorted(set(chunk['block_indices']) | {
                    p['block_idx'] for p in chunk['title_source'].get('blocks', [])
                    if p.get('block_idx') is not None})
                chunk['text'] = chunk['title'] + '\n' + chunk['procedure']
        # Only referenced definitions belong to this chunk's input identity.
        refs, _ = resolve_programs(chunk['text'], index)
        keys = {r['id'] for r in refs}
        chunk['procedure_index'] = {k: index[k] for k in keys if k in index}
        chunk['steps'] = split_steps(chunk)
        # A preparation and its characterization may share one example.
        if any(s['process_kind'] == 'synthesis' for s in chunk['steps']):
            chunk['process_kind'] = 'synthesis'
        if chunk['steps'] and chunk.get('title') and (
                chunk['steps'][0]['title'].startswith(chunk['title']) or
                chunk['steps'][0]['title'] == specific_target_title(chunk['title'])):
            # Keep the chunk's terminal title check aligned with source repair.
            chunk['title'] = chunk['steps'][0]['title']
            chunk['title_source'] = deepcopy(chunk['steps'][0]['title_source'])
        elif (chunk['steps'] and _NUMERIC_TITLE.fullmatch((chunk.get('title') or '').strip())):
            # A pure numeric chunk label ("1", "488-1"; printed headings such as
            # "实施例 488-1" lose the sub-number in segmenter.py:14-19) must not
            # hide the compound identity used by terminal-product coverage and
            # route targets. Fill only from a concrete source-grounded heading.
            step_title = (chunk['steps'][0].get('title') or '').strip()
            if (step_title and step_title != chunk.get('title')
                    and not _NUMERIC_TITLE.fullmatch(step_title)
                    and _product_heading(step_title)):
                chunk['title'] = step_title
                chunk['title_source'] = deepcopy(chunk['steps'][0].get('title_source')
                                                 or chunk.get('title_source'))
        if (chunk.get('kind') in ('example', 'compound') and _product_heading(chunk.get('title', ''))
                and not chunk.get('tables') and all(_product_heading(p['text']) for p in source_parts)):
            chunk['has_procedure'] = False
            title_context = chunk
        chunk['context_version'] = CONTEXT_VERSION
        payload = [chunk_fingerprint(chunk), CONTEXT_VERSION, digest.hexdigest(), chunk['steps'],
                   chunk.get('source_compound_evidence') or []]
        chunk['chunk_fingerprint'] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return chunks


def refine_steps(chunk, steps, extractor):
    """Accept model boundaries only if every range is an exact source slice."""
    if not hasattr(extractor, 'plan_steps'):
        return steps
    refined = []
    for unit in steps:
        if not unit.get('needs_step_planning'):
            refined.append(unit)
            continue
        from pipeline.extraction.inference import ContextLengthError
        try:
            plan = extractor.plan_steps(unit['local_text'], unit['title'])
        except ContextLengthError:
            # Keep the original unit; extraction will retain an explicit
            # review record if it also exceeds the provider's context window.
            refined.append(dict(unit, needs_step_planning=False,
                                context_issues=unit['context_issues'] + ['step_plan_context_too_long']))
            continue
        ranges = plan.get('ranges') or []
        cursor, checked = 0, []
        for item in ranges:
            start, end = item.get('start'), item.get('end')
            title = item.get('title', '')
            if (type(start) is not int or type(end) is not int or start != cursor
                    or end <= start or end > len(unit['local_text'])
                    or (title and title not in unit['local_text'][start:end] and title != unit['title'])):
                checked = []
                break
            checked.append(item)
            cursor = end
        if not checked or cursor != len(unit['local_text']):
            rejected = dict(unit)
            rejected['context_issues'] = unit['context_issues'] + ['step_plan_unverified']
            rejected['step_plan_raw'] = plan.get('raw')
            refined.append(rejected)
            continue
        for item in checked:
            text = unit['local_text'][item['start']:item['end']]
            refs, issues = resolve_programs(text, chunk.get('procedure_index', {}))
            method_count = sum(r.get('relation') != 'precursor_preparation' for r in refs)
            subset, offset = [], 0
            for part in unit['source_parts']:
                lo, hi = max(item['start'], offset), min(item['end'], offset + len(part['text']))
                if hi > lo:
                    subset.append(dict(part, text=part['text'][lo-offset:hi-offset], span=[lo-offset, hi-offset]))
                offset += len(part['text']) + 1
            refined.append(dict(unit, title=item['title'], text=(item['title']+'\n'+text).strip(),
                                local_text=text, source_parts=subset, procedure_references=refs,
                                context_issues=issues + (['telescoped_procedures_without_reported_intermediates'] if method_count>1 else []),
                                unit_kind='telescoped' if method_count>1 else 'single_step',
                                boundary_status='source_grounded', needs_step_planning=False,
                                step_plan_raw=plan.get('raw'), source_span=[item['start'],item['end']]))
    for i, unit in enumerate(refined, 1):
        unit['step_id'] = f"{chunk['id']}:step:{i}"
        unit['step_index'] = i
    return refined
