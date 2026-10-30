"""Lossless yield expressions shared by extraction, evidence and delivery."""
import re

_NUM = r'\d+(?:\.\d+)?'
_APPROX = r'(?:about|approximately|approx\.?|ca\.?|~|约)'
_RANGE = re.compile(rf'(?P<approx>{_APPROX})?\s*(?P<lower>{_NUM})\s*%?\s*(?:to|[-–—−]|至|到)\s*(?P<upper>{_NUM})\s*%?', re.I)
_SINGLE = re.compile(rf'(?P<approx>{_APPROX})?\s*(?P<value>{_NUM})\s*%?', re.I)


def parse_yield_value(value):
    text = str(value if value is not None else '').strip().replace('％', '%')
    match = _RANGE.fullmatch(text)
    if match:
        low, high = match.group('lower', 'upper')
        return {'value': None, 'range': {'lower': float(low), 'upper': float(high)},
                'approximate': bool(match.group('approx')), 'raw': text,
                'normalized': ('about ' if match.group('approx') else '') + low + '-' + high}
    match = _SINGLE.fullmatch(text)
    if match:
        return {'value': float(match.group('value')), 'range': None,
                'approximate': bool(match.group('approx')), 'raw': text,
                'normalized': ('about ' if match.group('approx') else '') + match.group('value')}
    return None


def yield_equal(left, right):
    a, b = parse_yield_value(left), parse_yield_value(right)
    if not a or not b or bool(a['range']) != bool(b['range']):
        return False
    if a['range']:
        return all(abs(a['range'][k] - b['range'][k]) < .005 for k in ('lower', 'upper'))
    return abs(a['value'] - b['value']) < .005


_EXPR = rf'(?:{_APPROX}\s*)?{_NUM}\s*%(?:\s*(?:to|[-–—−]|至|到)\s*{_NUM}\s*%?)?|(?:{_APPROX}\s*)?{_NUM}\s*(?:to|[-–—−]|至|到)\s*{_NUM}\s*%'
_EXPLICIT = re.compile(rf'(?P<before>{_EXPR})\s*(?:theoretical\s+)?(?:收率|产率|yield)\b|(?:yield|收率|产率)\s*(?:of|was|为|[:：])?\s*(?P<after>{_EXPR})', re.I)
ISOLATION_VERB = r'(?:得到|获得|制得|afford(?:ed|ing)?|yield(?:ed|ing)?|obtain(?:ed|ing)?|give|gave|given|giving|isolat(?:e|ed|ing)|provid(?:e|ed|ing)|furnish(?:ed|ing)?)'
_ISOLATED = re.compile(rf'{ISOLATION_VERB}[^。\n;]{{0,300}}?[（(][^）)]*?(?P<value>{_EXPR})[^）)]*[）)]', re.I)


def reported_yields(text):
    """Percentages need yield wording or an isolated-product predicate."""
    original = text or ''
    # Preserve character offsets so quoted evidence still points into the OCR.
    text = original.replace('\\%', ' %').replace('$', ' ').replace('％', '%')
    refs = []
    for pattern in (_EXPLICIT, _ISOLATED):
        for match in pattern.finditer(text or ''):
            raw = next(v for v in match.groupdict().values() if v is not None)
            # Do not promote purity, reagent concentration or formulation content.
            if re.search(r'purity|ee\b|conversion|纯度|浓度|concentration|w/w|w/v', match.group(0), re.I):
                continue
            parsed = parse_yield_value(raw)
            if parsed and not any(match.start() >= r['span'][0] and match.end() <= r['span'][1] for r in refs):
                refs.append(dict(parsed, span=list(match.span()),
                                 source_text=original[match.start():match.end()]))
    return sorted(refs, key=lambda r: r['span'][0])
