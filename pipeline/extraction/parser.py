"""Robust parsing of NER / Action LLM outputs (local impl, does NOT import eval/).

Mirrors eval/eval.py parsing semantics:
- fenced json / markdown blocks, raw json, outermost {...}/[...] extraction
- NER field-name normalization + multi-value split + yield normalization + null handling
- Action verb-sequence parsing (comma / semicolon / json array)
"""

import json
import re
from typing import Any, Dict, List, Optional

NER_FIELDS = [
    "REACTION_PRODUCT",
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
    "TEMPERATURE",
    "TIME",
    "YIELD_PERCENT",
]
FIELD_NAME_MAPPING = {field.lower(): field for field in NER_FIELDS}
MULTIVALUE_FIELDS = {"REACTION_PRODUCT", "STARTING_MATERIAL", "REAGENT_CATALYST", "SOLVENT"}
SCALAR_FIELDS = {"TEMPERATURE", "TIME", "YIELD_PERCENT"}

_WS_RE = re.compile(r"\s+")
_YIELD_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$")
_NULL_LIKE = {"", "null", "none", "n/a", "notspecified", "not specified", "未指定", "未知", "无", "—", "-", "unknown"}


# ── JSON extraction ──

def _try_load_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_fenced_block(text: str, start_markers: List[str], end_marker: str) -> Optional[str]:
    low = text.lower()
    start_idx = -1
    for m in start_markers:
        idx = low.find(m.lower())
        if idx != -1:
            start_idx = idx + len(m)
            break
    if start_idx == -1:
        return None
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        return None
    return text[start_idx:end_idx].strip()


def _extract_fenced_block_allow_eof(text: str, start_markers: List[str], end_marker: str) -> Optional[str]:
    block = _extract_fenced_block(text, start_markers, end_marker)
    if block is not None:
        return block
    low = text.lower()
    for m in start_markers:
        idx = low.find(m.lower())
        if idx != -1:
            return text[idx + len(m):].strip()
    return None


def parse_json_output(output_str: str) -> Any:
    """Extract a JSON object/array from LLM output (fenced or raw)."""
    if not output_str or not isinstance(output_str, str):
        return None
    s = output_str.strip()

    for start, end in (
        ("```json", "```"), ("```JSON", "```"),
        ("'''json", "'''"), ("'''JSON", "'''"),
        ("```markdown", "```"), ("```MARKDOWN", "```"), ("```md", "```"), ("```MD", "```"),
        ("'''markdown", "'''"), ("'''MARKDOWN", "'''"), ("'''md", "'''"), ("'''MD", "'''"),
    ):
        block = _extract_fenced_block(s, [start], end)
        if block:
            parsed = _try_load_json(block)
            if parsed is not None:
                return parsed

    block = _extract_fenced_block_allow_eof(s, ["```"], "```")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed
    block = _extract_fenced_block_allow_eof(s, ["'''"], "'''")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed

    parsed = _try_load_json(s)
    if parsed is not None:
        return parsed

    first_obj = s.find("{")
    first_arr = s.find("[")
    if first_obj == -1 and first_arr == -1:
        return None
    start = first_obj if (first_obj != -1 and (first_arr == -1 or first_obj < first_arr)) else first_arr
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return _try_load_json(s[start:i + 1].strip())
    return None


# ── normalization helpers ──

def strip_all_whitespace(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return _WS_RE.sub("", s)


def normalize_yield_value(v: Any) -> str:
    from .yield_values import parse_yield_value
    s = strip_all_whitespace(v)
    if not s or _is_null_like(s):
        return ""
    parsed = parse_yield_value(s)
    if parsed:
        return parsed['normalized']
    # Preserve ranges embedded in yield phrases before the scalar fallback.
    from .yield_values import _RANGE
    ranges = list(_RANGE.finditer(s))
    if len(ranges) == 1 and '%' in s:
        return parse_yield_value(ranges[0].group(0))['normalized']
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%?", s)
    if m:
        return m.group(1)
    m2 = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)%", s)
    if m2 and s.count('%') == 1:
        return m2.group(1)
    return ""


def _is_null_like(s: str) -> bool:
    return s.strip().lower() in _NULL_LIKE


def normalize_field_name(field_name: str) -> str:
    if not field_name:
        return ""
    mapped = FIELD_NAME_MAPPING.get(field_name.strip().lower())
    return mapped if mapped else field_name.strip().upper()


def _is_locant_comma(text: str, index: int) -> bool:
    """True for commas that join locants, e.g. ``N, N-`` or ``1, 2-``."""
    left = index - 1
    while left >= 0 and text[left].isspace():
        left -= 1
    start = left
    while start >= 0 and (text[start].isalnum() or text[start] in "'′"):
        start -= 1
    previous = text[start + 1:left + 1]
    right = index + 1
    while right < len(text) and text[right].isspace():
        right += 1
    end = right
    while end < len(text) and (text[end].isalnum() or text[end] in "'′"):
        end += 1
    following = text[right:end]
    if previous.isdigit() and following.isdigit():
        return True
    # N, N-dimethylformamide / N, O-dimethyl... / N, N,N-...
    if (len(previous) == 1 and previous.upper() in {"N", "O", "S", "P"}
            and following[:1].upper() in {"N", "O", "S", "P"}
            and end < len(text) and text[end] in ",-"):
        return True
    return False


def _split_multi_value(text: str, field: Optional[str] = None, diagnostics=None) -> List[str]:
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    s0 = text.strip()
    if _is_null_like(s0):
        return []
    if field == "YIELD_PERCENT":
        y = normalize_yield_value(text)
        return [] if not y else [y]
    s = text.replace("；", ";")
    parts: List[str] = []
    start = 0
    brackets: List[str] = []
    mismatched = False
    opening = {"(": ")", "[": "]", "{": "}", "（": "）", "【": "】"}
    for i, ch in enumerate(s):
        if ch in opening:
            brackets.append(opening[ch])
        elif ch in opening.values():
            if brackets and ch == brackets[-1]:
                brackets.pop()
            else:
                mismatched = True
        elif not brackets and (ch in (";", "、", "，") or (
            ch == "," and i + 1 < len(s) and s[i + 1].isspace()
            and not _is_locant_comma(s, i)
        )):
            # A crystalline-form suffix belongs to the preceding compound;
            # splitting it loses the form on the resolved molecular structure.
            if ch in {",", "，"} and re.match(
                    r"\s*(?:(?:crystalline\s+)?(?:form|polymorph)\s+|晶型\s*)"
                    r"[A-Z0-9αβγδ]+(?=\s*(?:$|[,;；、，]))", s[i + 1:], re.I):
                continue
            cleaned = s[start:i].strip()
            if cleaned:
                parts.append(cleaned)
            start = i + 1
    cleaned = s[start:].strip()
    if cleaned:
        parts.append(cleaned)
    if (brackets or mismatched) and diagnostics is not None:
        diagnostics.append(dict(field=field, value=text, reason='unbalanced_brackets'))
    return parts


# ── NER parsing ──

def parse_ner_records(output_str: str, diagnostics=None,
                      prompt_variant: Optional[str] = None) -> List[Dict[str, Any]]:
    """Preserve every JSON object returned for a multi-reaction chunk."""
    parsed = parse_json_output(output_str)
    records = parsed if isinstance(parsed, list) else [parsed]
    if not records or any(not isinstance(item, dict) for item in records):
        return []
    result = []
    for item in records:
        extra = {str(k).strip().lower(): v for k, v in item.items()}
        raw = {}
        for key, value in item.items():
            field = normalize_field_name(str(key))
            if field in NER_FIELDS:
                raw[field] = value
        if not raw:
            return []
        structured = {}
        for field in NER_FIELDS:
            value = raw.get(field)
            if field == "SOLVENT" and isinstance(value, dict):
                # Some models expose stage roles despite the flat NER schema.
                # Keep reaction solvents; workup evidence remains in raw JSON.
                value = value.get("reaction", value.get("reaction_solvents"))
            if isinstance(value, list):
                parts = []
                for item in value:
                    if item is None or _is_null_like(str(item)):
                        continue
                    text = str(item).strip()
                    if field in MULTIVALUE_FIELDS:
                        parts.extend(_split_multi_value(text, field=field, diagnostics=diagnostics))
                    elif text:
                        parts.append(text)
                if field in MULTIVALUE_FIELDS:
                    # v2 prompts often return ["A;B", "C"]; split and dedupe so
                    # downstream standardization sees one compound per item.
                    parts = list(dict.fromkeys(part for part in parts if part))
            elif field in MULTIVALUE_FIELDS:
                parts = list(dict.fromkeys(
                    part for part in _split_multi_value(value, field=field, diagnostics=diagnostics) if part))
            else:
                parts = []
            if field in MULTIVALUE_FIELDS:
                structured[field] = parts
            elif field == "YIELD_PERCENT":
                numeric = normalize_yield_value(value)
                structured[field] = numeric or None
            else:
                if isinstance(value, list):
                    # The model sometimes returns a JSON list for a scalar
                    # field; str() would persist a Python list literal like
                    # "['室温', '50°C']" (v7 example_64). Join instead.
                    scalar = "; ".join(parts) if parts else ""
                else:
                    scalar = str(value or "").strip()
                if field in ('TIME', 'TEMPERATURE') and not _is_null_like(scalar):
                    kept = []
                    pieces = re.split(r'[;；]', scalar)
                    rejected = False
                    for piece in pieces:
                        operation = re.search(r'过滤|浓缩|研磨|洗涤|纯化|收集|洗脱|蒸发|加入|添加|冷却|'
                                              r'\b(?:filter\w*|concentrat\w*|triturat\w*|wash\w*|purif\w*|collect\w*|elut\w*|evaporat\w*|add\w*|cool\w*)\b', piece, re.I)
                        temporal = re.search(r'\d+(?:\.\d+)?\s*(?:h\b|hrs?\b|hours?\b|min\b|minutes?\b|s\b|seconds?\b|days?\b|weeks?\b|天|小时|分钟|秒)|'
                                             r'过夜|overnight|until|直至|直到|同时|simultaneously|while', piece, re.I)
                        temperature = re.search(r'-?\d+(?:\.\d+)?\s*(?:℃|°\s*C)|室温|环境温度|'
                                                r'\b(?:RT|room temperature|ambient temperature|reflux)\b', piece, re.I)
                        supported_condition = temporal if field == 'TIME' else temperature
                        if operation and not supported_condition:
                            rejected = True
                            if diagnostics is not None:
                                diagnostics.append(dict(record_index=len(result)+1, field=field,
                                                        value=piece.strip(), reason='operation_without_time' if field == 'TIME'
                                                        else 'operation_without_temperature'))
                        elif piece.strip():
                            kept.append(piece.strip())
                    if rejected:
                        scalar = '；'.join(kept)
                structured[field] = None if _is_null_like(scalar) else scalar
        if prompt_variant == 'scheme_template':
            action = extra.get('action')
            if isinstance(action, str):
                action = None if _is_null_like(action) else action.strip()
            elif isinstance(action, list):
                action = [v.strip() for v in action
                          if isinstance(v, str) and not _is_null_like(v)]
            else:
                action = None
            structured['action'] = action
            order = extra.get('step_order')
            structured['step_order'] = order if type(order) is int and order > 0 else None
        result.append(structured)
    return result


def parse_ner_output(output_str: str) -> Dict[str, str]:
    """Legacy single-record parser; reject arrays instead of dropping records."""
    parsed = parse_json_output(output_str)
    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise ValueError("multiple NER records; use parse_ner_records")
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return {}
    return {normalize_field_name(str(k)): "" if v is None else str(v)
            for k, v in parsed.items() if normalize_field_name(str(k)) in NER_FIELDS}


def parse_ner_structured(output_str: str) -> Dict[str, Any]:
    records = parse_ner_records(output_str)
    if len(records) > 1:
        raise ValueError("multiple NER records; use parse_ner_records")
    return records[0] if records else {}


# ── Action parsing ──

_ACTION_VERB = re.compile(
    r"^(?:add|adding|added|mix|mixing|stir|stirring|heat|heating|cool|cooling|"
    r"reflux|refluxing|filter|filtering|wash|washing|dry|drying|extract|"
    r"extracting|purify|purifying|isolate|isolating|concentrate|"
    r"concentrating|evaporate|evaporating|dissolve|dissolving|charge|"
    r"charging|quench|quenching|insulate|insulating|crystallize|"
    r"crystallizing|加入|滴加|混合|搅拌|加热|升温|降温|冷却|回流|过滤|洗涤|"
    r"干燥|萃取|纯化|分离|浓缩|蒸发|溶解|猝灭|保温|结晶|离心|静置|"
    r"研磨|称取|转移|收集|稀释|煅烧|氢化|还原|氧化|偶联|偶合|取代|酰化|消除|"
    r"环化|缩合|酯化|胺化|反应|"
    r"加|保持|通入|监测|调节|润湿|回滤|输送|灌装|抽真空|充氮|"
    r"控制|压塞|轧盖|灭菌|灯检|分层|合并|脱气|鼓入|冲洗|处理|"
    r"淬灭|除去|碱化|倒入|中和|净化|回到室温|鼓入氮气|再次脱气|"
    r"加热回流|回流加热|减压浓缩|冷却至室温|冷却到室温|"
    r"除去溶剂|除去挥发物|逐滴加入|缓慢加入|分批加入|"
    r"separate|separating|combine|combining|neutralize|neutralizing|"
    r"degass|degassing|remove|removing|precipitate|precipitating|"
    r"沉淀|浆化|空气干燥|洗脱|捣碎|减压除去溶剂|温热|真空浓缩|减压除去|"
    r"悬浮|通过|置于|离心分离|水洗|注入|蒸干|加热助溶|称量|包衣|"
    r"保温搅拌|喷雾干燥|精密称量|溶清|加至|精滤|压制|去除|开环|"
    r"转化|回收|weigh|centrifuge|charged|filtered|washed|adjusted|filled|"
    r"cooled|dissolved|combined|heated|stirred|diluted|treated|concentrated|"
    r"discharged|recirculate|fill|prepare|prepared|place|placed|suspended|"
    r"wetting|neutralized|separated|removed|maintained|transferred|collect|"
    r"inject|injecting|purge|flush|elute|eluting|diluting|treat|maintain|"
    r"centrifuging|measuring|adjust|stirred|stand|holding|increase|derivatize|"
    r"monitor|allow|continue|freeze-dry|measure|blend|compress|granulate|"
    r"coat|milled|mill|lubricate|package|return|set|sample|store|"
    r"溶于|水解|回温|重新溶解|重结晶|酸化|挥发|取|磁力搅拌|"
    r"密封|放置|平衡|敞口|封住|吹扫|塞住|测试|冷冻干燥|活化|添加|引入|吸收|"
    r"形成浆液|通过分相器|减压蒸馏|柱色谱|放冷|减压干燥|进一步纯化|脱水|等渗|杀菌|"
    r"冻干|再结晶|降低|分配|分散|调配)$", re.I)

_ACTION_NON_OPERATION = {"得到", "获得", "生成", "用于", "examine", "examined", "examining",
                         "investigate", "investigated", "investigating", "obtain", "obtained", "obtaining"}

# Explicit chemical operations and their inflections; never arbitrary prose.
_CHEMICAL_ACTION = re.compile(
    r'^(?:(?:allow(?:ed|ing)?\s+to\s+)?(?:sit|sat|sitting)|dilut(?:e|ed|ing)|suspend(?:ed|ing)?|dry|dried|drying|'
    r'extract(?:ed|ing)?|purif(?:y|ied|ying)|quench(?:ed|ing)?|'
    r'evaporat(?:e|ed|ing)|mix(?:ed|ing)?|warm(?:ed|ing)?|'
    r'rins(?:e|ed|ing)|transfer(?:red|ring)?|bubbl(?:e|ed|ing)|'
    r'seal(?:ed|ing)?|clos(?:e|ed|ing)|open(?:ed|ing)?|'
    r'degas(?:s|sed|sing)?|seed(?:ed|ing)?|agitat(?:e|ed|ing)|'
    r'pour(?:ed|ing)?|triturat(?:e|ed|ing)|recrystalliz(?:e|ed|ing)|'
    r'lyophiliz(?:e|ed|ing)|chromatograph(?:ed|ing)?|acidif(?:y|ied|ying)|'
    r'spray(?:ed|ing)?|dispers(?:e|ed|ing)|pulveriz(?:e|ed|ing)|'
    r'siev(?:e|ed|ing)|elut(?:e|ed|ing)|granulat(?:e|ed|ing)|'
    r'tak(?:e|en|ing)(?:\s+up)?|pull(?:ed|ing)?|'
    r'partition(?:ed|ing)?|coupl(?:e|ed|ing)|premix(?:ed|ing)?|'
    r'irradiat(?:e|ed|ing)|pass(?:ed|ing)?\s+through)$', re.I)


def action_supported(action, text):
    """Literal English inflections without translating or inventing actions."""
    if re.fullmatch(r'(?:allow(?:ed|ing)?\s+to\s+)?(?:sit|sat|sitting)', str(action), re.I):
        return bool(re.search(r'\b(?:allow(?:ed|ing)?\s+to\s+)?(?:sit|sat|sitting)\b', text, re.I))
    if re.fullmatch(r'distill(?:ed|ing)?(?: off)?', str(action), re.I):
        suffix = r'\s+off' if str(action).lower().endswith(' off') else ''
        return bool(re.search(r'\bdistill(?:ed|ing)?' + suffix + r'\b', text, re.I))
    if re.fullmatch(r'tak(?:e|en|ing)(?:\s+up)?', str(action), re.I):
        suffix = r'\s+up' if re.search(r'\s+up$', str(action), re.I) else ''
        return bool(re.search(r'\btak(?:e|en|ing)' + suffix + r'\b', text, re.I))
    if re.fullmatch(r'pass(?:ed|ing)?\s+through', str(action), re.I):
        return bool(re.search(r'\bpass(?:ed|ing)?\s+through\b', text, re.I))
    roots = ('add', 'mix', 'stir', 'heat', 'cool', 'filter', 'wash', 'dry', 'extract',
             'purify', 'isolate', 'concentrate', 'evaporate', 'dissolve', 'charge', 'quench',
             'dilute', 'suspend', 'warm', 'rinse', 'transfer', 'bubble', 'seal', 'degas',
             'seed', 'agitate', 'pour', 'triturate', 'recrystallize', 'lyophilize',
             'chromatograph', 'acidify', 'spray', 'disperse', 'pulverize', 'sieve', 'elute', 'pull',
             'partition', 'couple', 'premix', 'irradiate')
    for root in roots:
        if root.endswith('e'):
            pattern = re.escape(root[:-1]) + r'(?:e|ed|ing)'
        elif root.endswith('y'):
            pattern = re.escape(root[:-1]) + r'(?:y|ied|ying)'
        elif root in ('stir', 'transfer'):
            pattern = re.escape(root) + r'(?:red|ring)?'
        elif root == 'degas':
            pattern = r'degas(?:s|sed|sing)?'
        else:
            pattern = re.escape(root) + r'(?:ed|ing)?'
        if re.fullmatch(pattern, str(action), re.I):
            return bool(re.search(r'\b' + pattern + r'\b', text, re.I))
    return False


def _truncate_repeated_actions(actions: List[str], min_repeats: int = 8,
                               max_period: int = 8,
                               max_actions: int = 80) -> tuple:
    """Collapse degenerate verb loops emitted by small local models.

    A real procedure rarely repeats the exact same 1-8 action cycle eight
    times; observed model degeneration repeats it 20-30 times. Truncate such a
    tail to at most two cycles and cap pathological sequences. Returns
    ``(actions, diagnostics)``; diagnostics is None when the sequence is clean.
    """
    if not actions:
        return actions, None
    collapsed = [action for index, action in enumerate(actions)
                 if index == 0 or action != actions[index - 1]]
    duplicates_removed = len(actions) - len(collapsed)
    n = len(collapsed)
    best = None
    for period in range(1, min(max_period, n // min_repeats) + 1):
        repeats = 1
        start = n - period
        while (start - period >= 0 and
               collapsed[start - period:start] == collapsed[start:start + period]):
            repeats += 1
            start -= period
        if repeats >= min_repeats:
            keep = min(repeats, 2)
            length = start + keep * period
            if best is None or length < best[0]:
                best = (length, period, repeats)
    if best is not None:
        length, period, repeats = best
        diagnostics = {"period": period, "repeats": repeats,
                       "original_length": len(actions),
                       "truncated_length": length}
        if duplicates_removed:
            diagnostics["duplicates_removed"] = duplicates_removed
        return collapsed[:length], diagnostics
    if n > max_actions:
        diagnostics = {"period": None, "repeats": None,
                       "original_length": len(actions),
                       "truncated_length": max_actions}
        if duplicates_removed:
            diagnostics["duplicates_removed"] = duplicates_removed
        return collapsed[:max_actions], diagnostics
    if duplicates_removed:
        return collapsed, {"duplicates_removed": duplicates_removed,
                           "original_length": len(actions),
                           "truncated_length": n}
    return collapsed, None


def parse_action_details(output_str: str) -> Dict[str, Any]:
    """Preserve supported operations without claiming a partial list is complete."""
    def result(actions, rejected, status):
        return {"actions": actions, "rejected": rejected, "status": status}
    if not output_str or not isinstance(output_str, str):
        return result([], [], "empty_action_sequence")
    s = output_str.strip()
    parsed = parse_json_output(s)
    if isinstance(parsed, dict):
        parsed = next((parsed[k] for k in ("actions", "action_sequence", "steps", "operations")
                       if isinstance(parsed.get(k), list)), None)
    if isinstance(parsed, list):
        parts = [v.strip() if isinstance(v, str) else v for v in parsed]
    elif parsed is not None:
        return result([], [], "invalid_action_format")
    else:
        if ("\n" in s or any(mark in s for mark in ("。", "我们", "需要", "首先", "分析文本"))
                or re.search(r'\b(?:we need|first we|let us)\b', s, re.I)):
            return result([], [], "invalid_action_format")
        parts = [part.strip() for part in re.split(r"[,;；、，]", s)]
    actions, rejected = [], []
    for index, part in enumerate(parts):
        if isinstance(part, str) and part.lower() in _ACTION_NON_OPERATION:
            rejected.append({"index": index, "value": part, "reason": "non_operation"})
            continue
        if isinstance(part, str) and (_ACTION_VERB.fullmatch(part) or _CHEMICAL_ACTION.fullmatch(part)
                                      or re.fullmatch(r'distill(?:ed|ing)?(?: off)?', part, re.I)):
            actions.append(part)
        else:
            reason = ('characterization' if isinstance(part, str) and re.search(
                r'\b(?:HPLC|NMR|MS|analysis|analy[sz]ed)\b|表征|分析', part, re.I)
                else 'unsupported_operation' if isinstance(part, str) else 'invalid_item_type')
            rejected.append({"index": index, "value": part, "reason": reason})
    actions, loop_info = _truncate_repeated_actions(actions)
    incomplete = any(r['reason'] != 'non_operation' for r in rejected)
    details = result(actions, rejected,
                     "partial_action_sequence" if incomplete and actions else
                     "invalid_action_sequence" if incomplete else
                     "ok" if actions else "empty_action_sequence")
    if loop_info:
        details["loop_truncated"] = loop_info
    return details


def parse_action_output(output_str: str) -> List[str]:
    return parse_action_details(output_str)["actions"]
