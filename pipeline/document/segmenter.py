"""Turn ordered PDF/MinerU blocks into title + procedure units.

ReactionSeek's extractor consumes a reaction title and its synthesis procedure.
This module supplies those units while retaining source pages and block indices.
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List


HEADING = re.compile(
    r"^\s*(?:\[\d{1,4}\]\s*)?(?:(?:\d{1,3})\s+)?(?P<kind>(?:(?:(?:Reference|Production|Synthetic|Preparative)\s+)?Example|Intermediate|Step|"
    r"Preparation|Synthesis|Compound|Boronate Ester)(?![A-Za-z])|实施例|实例|制备例|合成例|中间物|中间体|步骤|制备步骤)\s*"
    r"(?P<number>\d{1,4}\s+to\s+\d{1,4}|\d{1,4}(?:\.\d{1,3})*|[零〇一二两三四五六七八九十百]+|[A-Za-z]{1,3}(?:\s*[-–—]\s*)?\d{1,3}|[IVXLCDM]+|[A-Za-z]{1,3})(?P<letter>[A-Za-z]?)(?P<prime>['’′]?)"
    r"(?:\s*[:：.．、\-—–～]\s*|\s+|$|(?=[\u4e00-\u9fff]))(?P<title>.*)$", re.I | re.S,
)
UNNUMBERED_HEADING = re.compile(
    r"^(?P<kind>Preparation|Synthesis)(?:\s+of\s+|\s*[:：]\s*)(?P<title>.+)$|"
    r"^(?P<kind_zh>制备|合成)(?:方法|过程)?[：:]\s*(?P<title_zh>.+)$|"
    r"^(?P<section_kind>Biological (?:Assays|Data|Activity|Evaluation)|Pharmacological Activity)\s*[:：]?$", re.I)
SEGMENTATION_VERSION = "2026-10-27-scheme-image-window"
LINE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
ANALYTICAL = re.compile(
    r"(?:\$[^$]{0,24}\$\s*|¹H\s*|\b1\s*H\s*)NMR\b|\bm/z\s*[:=]", re.I,
)
SECTION_END = re.compile(
    r"^\s*(?:\[\d{1,4}\]\s*)?(?:Useful crystalline polymorphic forms|What is claimed is\s*:?|"
    r"(?:I|We)\s+claim\s*:?|The invention (?:claimed is|is claimed)\s*:?|"
    r"Claims\s*:?(?:\s|$)|Biological Activity\b|Biological Evaluation\b|"
    r"Pharmacological Activity\b|References\s*$|Bibliography\s*$|"
    r"Sequence Listing\b|SEQ\s+ID\s+NO\s*[:：]|序列表|效果实施例|"
    r"权利要求(?:书)?\s*$|生物(?:学)?活性\b|参考文献\s*$)", re.I)
REF = re.compile(r"\b(Intermediate|Example)\s+(\d{1,3}(?:\.\d{1,3})*)([A-Za-z]?)\b|"
                 r"(?:实施例|中间体)\s*([零〇一二两三四五六七八九十百]+|\d+(?:\.\d+)?)", re.I)

# Used for untitled documents and synthesis procedures before the first title.
# Both operation and experimental context are required to avoid clinical prose.
FALLBACK_OPERATION = re.compile(
    r"\b(?:stirred|heated|refluxed|dissolved|added|filtered|evaporated|"
    r"purified|isolated|synthesized|reacted|extracted|"
    r"washed|cooled|precipitated|chromatographed|"
    r"afforded|yielded)\b|"
    r"(?:搅拌|回流|加热|溶解|滴加|加入|过滤|浓缩|纯化|萃取|干燥|制得|合成|还原而制成)", re.I)
FALLBACK_CONTEXT = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?\s*(?:mg|g|kg|mmol|mol|mL|L|h|hours?|min|minutes?|°C|℃|小时|分钟|天)(?![A-Za-z])|"
    r"\b(?:yield|solvent|solution)\b|"
    r"(?:毫克|千克|毫摩尔|摩尔|毫升|摄氏度|收率|溶液|反应液)", re.I)
PROCEDURE_OPERATION = re.compile(
    r'加入|滴加|搅拌|加热|冷却|回流|溶解|过滤|洗涤|萃取|浓缩|干燥|通入|(?:反应|保温)\s*\d|'
    r'\b(?:added|stirred|heated|cooled|refluxed|dissolved|filtered|washed|extracted|'
    r'concentrated|dried|treated|charged|hydrogenated)\b', re.I)


def has_procedure_operations(text):
    return bool(PROCEDURE_OPERATION.search(_de_latex(text)))
NON_REACTION = re.compile(
    r"(?:CCK[- ]?8|IC\s*50|96\s*(?:well|孔)|培养基|细胞|抑制率|"
    r"USP\s+Apparatus|dissolution|溶出|流动相|mobile phase|"
    r"clinical trial|patients?\s+were|患者|给药剂量)", re.I)

# Body-wide markers for the dual gate: a strong assay/analysis signal anywhere
# in the text, combined with the *absence* of synthesis stoichiometry evidence,
# identifies a pure assay/characterization/crystallization-screening chunk even
# when its title and first 500 characters carry no such word (v7 over-recovery:
# BTK/EGFR TR-FRET binding paragraphs, XRPD/TGA/DSC parameter tables, slurry
# competition tests were extracted as reactions).
ASSAY_MARKER = re.compile(
    r"(?:\b(?:assays?|T\s+cells?|cells?|apoptosis|incubat\w*|western\s+blot|"
    r"flow\s+cytometry|mice|mouse|rats?|xenograft|oral\s+gavage|dosed?|dosing|"
    r"cytokines?|ELISA)\b|\d+[- ]well|TR-?FRET|LanthaScreen|CCK[- ]?8|IC\s*50|EC\s*50|MTT|resazurin|"
    r"96\s*(?:well|孔)|温育|培养基|细胞|抑制率|"
    r"(?:comparison|evaluation|measurement|determination)\s+of\s+(?:the\s+)?half[- ]life|"
    r"in\s+vitro\s+in\s+(?:human\s+)?serum|"
    r"结合亲和力|结合活性|亲和力|binding\s+affinity|anti-?proliferative|"
    r"cell\s+viability|抗增殖)", re.I)
ANALYSIS_MARKER = re.compile(
    r"(?:X-?RPD|X-?ray\s+powder|TGA|DSC|DVS|衍射|差示扫描|热重|动态水分吸附|"
    r"溶出|流动相|mobile\s+phase|USP\s+Apparatus|dissolution|"
    r"混悬[^。]{0,12}(?:试验|实验)|晶型筛选|结晶形式之间的稳定性|"
    r"crystal\s+structure|diffraction|stability|tap(?:ped)?\s+density|viscosity|恒温箱|取样检查|characterization|polymorph(?:ic)?\s+screening|crystalline\s+form)", re.I)
STOICH_EVIDENCE = re.compile(
    r"(?:mmol|μmol|umol|equivalents?\b|equiv\b|yield|收率|当量|摩尔)", re.I)
FORMULATION_MARKER = re.compile(
    r'\b(?:formulat(?:ions?|ed|ing)|tablets?|granul\w*|excipients?|fluidized\s+bed|dosage\s+form|'
    r'powder\s+blend|coating|film[- ]coat\w*|compression|manufacturing\s+process)\b|处方|(?<!抑)制剂|片剂|包衣|压片|药物组合物|药液|注射液', re.I)
PREPARATION_OPERATION = re.compile(
    r'\b(?:dissolved|suspended|added|heated|stirred|refluxed|filtered|dried|precipitated|neutralized)\b|溶解(?!氧)|加入|搅拌|过滤|干燥|中和', re.I)

_LATEX_TOKEN = re.compile(r"\\(?:mathrm|text|mathbf)\{([^}]*)\}|\$\$?|\^|\_|[{}\\]")


def _de_latex(text: str) -> str:
    """Strip LaTeX artifacts so IC$_{50}$ / \\mathrm{BTK}^{WT} match plain rules."""
    return _LATEX_TOKEN.sub(lambda m: m.group(1) or "", text or "")


def _number(value: str):
    if re.fullmatch(r'[A-Za-z]+\s*[-–—]\s*\d+', value):
        return re.sub(r'\s*[-–—]\s*', '_', value.lower())
    if re.fullmatch(r'\d+\s+to\s+\d+', value, re.I):
        return re.sub(r'\s+', '_', value.lower())
    if value.isdecimal():
        return int(value)
    if re.fullmatch(r'[A-Za-z]+\d*|\d+(?:\.\d+)+', value):
        return value.lower()
    decimal_letter = re.fullmatch(r'(\d+(?:\.\d+)+)([A-Za-z]?)', value)
    if decimal_letter:
        return decimal_letter[1].lower() + decimal_letter[2].lower()
    # OCR/PDF text may append a CJK character to a reference number
    # ("实施例1一致" -> "1一"); retain the leading numeric token.
    numeric_prefix = re.match(r'\d+(?:\.\d+)*', value)
    if numeric_prefix:
        token = numeric_prefix[0]
        return int(token) if token.isdecimal() else token.lower()
    digits = {ch: i for i, ch in enumerate("零一二三四五六七八九")}
    digits["〇"] = 0
    digits["两"] = 2
    total = current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char == "十":
            total += (current or 1) * 10
            current = 0
        elif char == "百":
            total += (current or 1) * 100
            current = 0
        else:
            raise ValueError(f"unsupported heading number: {value}")
    return total + current


def heading_match(value: str):
    """Match a MinerU title without losing its paragraph-number prefix."""
    collective = re.fullmatch(r'\s*Examples\s+([A-Za-z]{1,3}\d*)\s+and\s+([A-Za-z]{1,3}\d*)\s*', value, re.I)
    if collective:
        # A shared scheme precedes the first procedure; subsequent explicit
        # headings still create their own units. Do not turn "and AL1" into a name.
        value = 'Example ' + collective[1]
    value = re.sub(r'^\s*Examples\s+(\d+\s+to\s+\d+)\s*$', r'Example \1', value, flags=re.I)
    roman = re.match(r'^\s*([ivx]{1,5})\)\s+(\(?\d.*[-,].+)$', value, re.I | re.S)
    if roman and re.search(r'bromo|chloro|fluoro|methyl|phenyl|benz|pyr', roman[2], re.I):
        value = f'Step {roman[1]}: {roman[2]}'
    match = HEADING.match(value.strip())
    if match and match.group('kind').lower() == 'compound' and not re.match(
            r'^\s*(?:\[\d+\]\s*)?Compound\s+\d+[A-Za-z]?\s*[:：]', value, re.I):
        return None
    if match and match.group('number').lower() in ('of', 'and', 'the', 'for', 'to', 'in', 'on', 'as', 'is', 'are', 'was', 'by', 'an'):
        return None
    if match and re.match(r"(?:采用|参照|按照)实施例", match.group("title")):
        return None
    if match and match.group('kind').lower() in ('intermediate', '中间物', '中间体') and re.match(
            r'\s*(?:\(\s*\d+(?:\.\d+)?\s*(?:mg|g|mmol|mol)\b|was\b|were\b|and\b)',
            match.group('title'), re.I):
        return None
    if match and match.group('kind') in ('中间物', '中间体') and re.match(
            r'\s*(?:\([\d.]+\s*(?:mg|g|mmol|mol)\b|溶于|的溶液|与|用|作为|加入)',
            match.group('title'), re.I):
        return None
    return match


def _captions(block: Dict[str, Any]) -> List[str]:
    values = []
    for key in ('table_caption', 'figure_caption', 'image_caption'):
        value = block.get(key) or []
        values.extend(str(v).strip() for v in (value if isinstance(value, list) else [value]))
    return list(dict.fromkeys(values))


def image_annotations(block):
    """Retain printed captions without treating a scheme's reagent as its product.

    Footnotes on a multi-compound scheme remain in the cached PDF source;
    they cannot be assigned to a particular step's characterization here.
    """
    values = _captions(block)
    return [v for v in dict.fromkeys(values) if v and not LINE_NUMBER.fullmatch(v)
            and not heading_match(v) and not re.match(r'^\s*(?:Fig(?:ure)?\.?|图)\s*\d+', v, re.I)]


def _figure_legend(block, match, figures):
    """Standalone Example labels inside a numbered figure are plot legends."""
    if not match or match.group('title').strip() or block.get('text_level'):
        return False
    box = block.get('bbox')
    if not isinstance(box, list) or len(box) != 4:
        return False
    for figure in figures.get(block.get('page_idx'), []):
        outer = figure.get('bbox')
        if (isinstance(outer, list) and len(outer) == 4
                and outer[0] <= box[0] <= box[2] <= outer[2]
                and outer[1] <= box[1] <= box[3] <= outer[3]):
            return True
    return False


def _references(value: str) -> List[str]:
    # Repair a common two-column OCR layout: "Example 40 6" has margin
    # line number 40 followed by the cited Example 6.
    value = re.sub(r"\bExample\s+\d{1,3}\s+(\d{1,3})\b",
                   r"Example \1", value, flags=re.I)
    refs = []
    for group in re.finditer(r"\bExamples\s+(\d{1,3}(?:\s*[,/]\s*\d{1,3})*"
                             r"\s*(?:and\s+\d{1,3})?)", value, re.I):
        for number in re.findall(r"\d{1,3}", group.group(1)):
            target = f"example_{int(number)}"
            if target not in refs:
                refs.append(target)
    for match in REF.finditer(value):
        if re.search(r'(?:Reference|Production|Synthetic|Preparative)\s+$', value[:match.start()], re.I):
            continue
        if match.group(4):
            target = "example_" + str(_number(match.group(4)))
        else:
            # MinerU may put a two-column margin number between Example and its true number.
            if re.match(r"\s+\d{1,3}\b", value[match.end():]):
                continue
            if re.match(r"\s+of\s+(?:WO|US|CN)\s*\d", value[match.end():], re.I):
                continue
            target = f"{match.group(1).lower()}_{_number(match.group(2))}{match.group(3).lower()}"
        if target not in refs:
            refs.append(target)
    from .step_context import _NAMED_REFERENCE, reference_key
    for match in _NAMED_REFERENCE.finditer(value):
        if re.match(r'\s+of\s+(?:WO|US|CN)\s*\d', value[match.end():], re.I):
            continue
        target = reference_key(match)
        if target and target not in refs:
            refs.append(target)
    return refs


def is_reaction_candidate(text: str) -> bool:
    """Evidence annotation; never a hard gate for a titled procedure."""
    clean = _de_latex(text)
    return bool(FALLBACK_OPERATION.search(clean) and FALLBACK_CONTEXT.search(clean)
                and not NON_REACTION.search(clean))


def chunk_fingerprint(chunk: Dict[str, Any]) -> str:
    """Identity of the exact input to extraction and its source span."""
    normalized = re.sub(r"\s+", " ", chunk["text"]).strip()
    payload = (SEGMENTATION_VERSION, normalized, chunk["block_start"],
               chunk["block_end"], chunk["page_start"], chunk["page_end"],
               chunk.get("block_indices", []), chunk.get("images", []),
               chunk.get("tables", []))
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def classify_process(title: str, text: str) -> str:
    """Classify the operation before interpreting any mapped structure score."""
    heading = _de_latex(re.sub(r"\s+", " ", title)).lower()
    body = _de_latex(text)
    lead = (heading + " " + body[:500]).lower()
    preparation = bool(PREPARATION_OPERATION.search(body) and re.search(
        r'\d+(?:\.\d+)?\s*(?:mg|g|kg|m[lL]|mmol|毫克|毫升)\b', body, re.I))
    # CN101503409B, Example 11: dissolving an unspecified active ingredient,
    # making it isotonic and filling ampoules is a dosage-form preparation.
    if re.search(r'(?:活性物质|活性成分|active (?:substance|ingredient)).{0,60}(?:安瓿|ampoules?)', heading, re.I):
        return 'formulation'
    # A bio/formulation body cannot become synthesis just because it was stirred.
    if ASSAY_MARKER.search(heading) or (ASSAY_MARKER.search(body[:500]) and not STOICH_EVIDENCE.search(body[:500])) or (ASSAY_MARKER.search(body) and not STOICH_EVIDENCE.search(body)):
        return 'assay'
    if re.search(r'\b(?:measurements?|determination|crystal structure|diffraction|stability|analys[ie]s)\b|测定|检测|分析|稳定性', heading, re.I):
        return 'analysis'
    if FORMULATION_MARKER.search(lead) or (FORMULATION_MARKER.search(body) and not re.search(
            r'\b(?:coupling|hydrogenation|bromination|amidation|cyclization)\b|偶联|氢化|溴化', body, re.I)):
        return 'formulation'
    procedure = body.strip()
    if procedure.lower().startswith(heading) and heading:
        procedure = procedure[len(heading):].lstrip()
    procedure = re.sub(r'^\s*(?:\[\d{1,6}\]\s*)*', '', procedure)
    pure_sfc = re.match(r'(?:通过|采用|经)\s*(?:制备型|手性)?\s*SFC[^。\n]{0,50}(?:纯化|分离)|'
                        r'.{0,400}\b(?:was|were)\s+(?:purified|separated|resolved)\s+'
                        r'(?:by|using|via)\s+(?:preparative\s+|chiral\s+)?SFC\b', procedure, re.I)
    conversion = re.search(r'\b(?:added|stirred|heated|refluxed|reacted|treated|synthesized|'
                           r'coupled|hydrogenated|oxidized|reduced)\b|'
                           r'加入|搅拌|加热|回流|反应|水解|还原|氧化|偶联|合成|氢化|脱保护', procedure, re.I)
    # Example163a/b: reported masses, mmol and yields describe physical
    # isolation of isomers, not formation of new covalent structures. A
    # synthesis followed by SFC must retain its synthesis classification.
    if pure_sfc and not conversion:
        return 'purification'
    if re.search(r'form\s+screening|polymorph(?:ic)?\s+screening|slurry\s+competition|晶型筛选|竞争搅拌试验|'
                 r'结晶形式之间的稳定性|相关结果参见以下表', heading + ' ' + body, re.I) and not STOICH_EVIDENCE.search(body):
        return 'analysis'
    if preparation and re.search(r'\b(?:preparation|prepare|salt|form\s+[a-z0-9]+|crystalline|hydrate)\b|制备|盐|晶型|水合', heading, re.I):
        return 'synthesis'
    if re.search(r"\b(?:pharmacokinetic|clinical|cytotoxicity|cell viability|bioassay)\b|"
                 r"(?:细胞|抑制率|半数抑制浓度|药代动力学|药效学|结合亲和力|结合活性|亲和力|"
                 r"抗增殖)", heading, re.I):
        return "assay"
    if re.search(r"\b(?:determination|measurements?|analys[ie]s|assays?|"
                 r"dissolution|stability|solubility|impurities|electrophoresis|"
                 r"content|density|excipient studies|compression studies|"
                 r"characterization|polymorph\w*|crystalline form)\b|"
                 r"(?:测定|检测|分析|稳定性|溶出|色谱|纯度|含量|表征|鉴定)", heading, re.I):
        return "analysis"
    if re.search(r"\b(?:purification|carbon processing|carbon treatment|"
                 r"diafiltration|ultrafiltration|concentration and isolation)\b|"
                 r"(?:纯化|水洗|脱色)", heading, re.I):
        return "purification"
    if re.search(r"\b(?:synthesis|reaction|coupling)\b|(?:合成|偶联)", heading, re.I):
        return "synthesis"
    if re.search(r"通用程序|general procedure", lead, re.I):
        return "synthesis"
    # Dual gate on the whole body: a strong assay/analysis marker with no
    # stoichiometry evidence anywhere cannot be a synthesis procedure, no
    # matter how operation-like its verbs read (slurry stirring, incubation).
    if ASSAY_MARKER.search(body) and not STOICH_EVIDENCE.search(body):
        return "assay"
    if ANALYSIS_MARKER.search(body) and not STOICH_EVIDENCE.search(body) and not preparation:
        return "analysis"
    if NON_REACTION.search(lead):
        if re.search(r"CCK[- ]?8|IC\s*50|96\s*(?:well|孔)|培养基|细胞|抑制率|"
                     r"clinical|patients?|患者|给药", lead, re.I):
            return "assay"
        # dissolution / mobile phase / USP apparatus and any other
        # NON_REACTION hit is analysis context; never fall through to
        # synthesis (v7 let a "96孔" hit with no sub-branch slip through).
        return "analysis"
    if re.search(r"suppositor|tablets?|composition|formulation|处方|制剂|片剂|注射液", heading, re.I):
        return "formulation"
    if re.search(r"\bingredient\s+amount\b|\bq\.s\.", lead, re.I):
        return "formulation"
    # Heading-only analysis/purification above plus the body-wide dual gate
    # decide non-synthesis chunks. A synthesis body's characterization tail
    # (1H NMR, purity, HPLC) must not short-circuit it here — such tails
    # carry none of the strong markers, and real syntheses carry
    # stoichiometry evidence, so the dual gate leaves them untouched.
    return "synthesis" if is_reaction_candidate(body) else "unverified"


def _finish_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    chunk["candidate_evidence"] = is_reaction_candidate(chunk["text"])
    chunk["process_kind"] = classify_process(chunk["title"], chunk["text"])
    chunk["segmentation_version"] = SEGMENTATION_VERSION
    chunk["chunk_fingerprint"] = chunk_fingerprint(chunk)
    return chunk


def fallback_procedures(blocks: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Group experimental paragraphs throughout a document without headings.

    No early-document cap or text truncation is applied. Long runs are split
    into bounded windows, retaining all selected text and its block range.
    """
    chunks: List[Dict[str, Any]] = []
    candidates = []
    excluded_section = False
    for index, block in enumerate(blocks):
        value = _block_text(block)
        if heading_match(value) or UNNUMBERED_HEADING.match(value) or re.search(r'通用程序|General Procedures?|^具体实施方式$|^EXAMPLES$', value, re.I):
            excluded_section = False
        boundary = SECTION_END.match(value)
        if boundary and (not value[boundary.end():].strip() or re.search(r'SEQ\s+ID|Sequence Listing|序列表|效果实施例', value, re.I)):
            excluded_section = True
        if excluded_section:
            continue
        if block.get("type", "text") in ("text", "list") or experimental_footer(block):
            value = _block_text(block)
            if SECTION_END.match(value):
                continue
            if (value and not NON_REACTION.search(value)
                    and (FALLBACK_OPERATION.search(value) or FALLBACK_CONTEXT.search(value))):
                page = int(block.get("page_idx", block.get("page_number", 1) - 1)) + 1
                candidates.append((index, page, value))

    def emit(run: List[tuple]) -> None:
        if not run:
            return
        # Include intervening paragraphs: the operation and the reagent/amount
        # may be in separate MinerU blocks, and either block alone can look
        # incomplete. The next paragraph often contains the workup/product.
        expanded = []
        end = min(run[-1][0] + 1, len(blocks) - 1)
        # A dangling sentence can finish at the top of the next page.
        while end + 1 < len(blocks) and end < run[-1][0] + 5:
            tail = _block_text(blocks[end])
            nxt = _block_text(blocks[end + 1])
            if (not tail or re.search(r"[。.!?；;]$", tail)
                    or heading_match(nxt) or SECTION_END.match(nxt)):
                break
            end += 1
        start = run[0][0]
        if start and re.search(r'通用程序|General Procedures?', _block_text(blocks[start - 1]), re.I):
            start -= 1
        for index in range(start, end + 1):
            block = blocks[index]
            if block.get("type", "text") not in ("text", "list") and not experimental_footer(block):
                continue
            value = _block_text(block)
            if block.get('_covered') or SECTION_END.match(value) or heading_match(value):
                break
            if value:
                page = int(block.get("page_idx", block.get("page_number", 1) - 1)) + 1
                expanded.append((index, page, value))
        if not expanded or not is_reaction_candidate(" ".join(item[2] for item in expanded)):
            return
        window: List[tuple] = []
        size = 0

        def write_window() -> None:
            if not window:
                return
            raw = " ".join(item[2] for item in window)
            if not is_reaction_candidate(raw) or not has_procedure_operations(raw):
                return
            chunk_id = f"fallback_{window[0][0]}_{window[-1][0]}_{len(chunks) + 1}"
            # Attach the figures this window's text describes: expand outward
            # across contiguous image/chart/table blocks. The VLM routinely
            # splits one scheme figure into consecutive image blocks, so the
            # former strict ±1 window orphaned the other halves.
            related_idx = []
            for start, step in ((window[0][0] - 1, -1), (window[-1][0] + 1, 1)):
                index = start
                while 0 <= index < len(blocks) and blocks[index].get("type") in ("image", "chart", "table"):
                    related_idx.append(index)
                    index += step
            related = [blocks[i].get("img_path") for i in sorted(related_idx)
                       if blocks[i].get("img_path")]
            chunks.append(_finish_chunk({
                "id": chunk_id, "reference_id": chunk_id,
                "kind": "fallback", "number": len(chunks) + 1,
                "title": "", "source": source,
                "page_start": window[0][1], "page_end": window[-1][1],
                "block_start": window[0][0], "block_end": window[-1][0],
                "images": list(dict.fromkeys(related)), "tables": [],
                "block_indices": sorted({item[0] for item in window}),
                "procedure_raw": raw, "procedure": raw,
                "text_parts": [item[2] for item in window],
                "text": raw, "references": _references(raw), "has_procedure": True,
            }))

        for index, page, value in expanded:
            # Split an unusually large MinerU block without discarding its tail.
            parts = []
            while len(value) > 6000:
                cut = value.rfind(" ", 4000, 6000)
                if cut < 0:
                    cut = 6000
                parts.append(value[:cut].strip())
                value = value[cut:].strip()
            if value:
                parts.append(value)
            for part in parts:
                if window and size + len(part) + 1 > 6000:
                    write_window()
                    window, size = [], 0
                window.append((index, page, part))
                size += len(part) + 1
        write_window()

    run: List[tuple] = []
    for item in candidates:
        if run and (item[0] - run[-1][0] > 2 or any(
                SECTION_END.match(_block_text(b)) or heading_match(_block_text(b))
                or b.get('_covered')
                for b in blocks[run[-1][0] + 1:item[0]])):
            emit(run)
            run = []
        run.append(item)
    emit(run)
    return chunks


def _block_text(block: Dict[str, Any]) -> str:
    value = block.get("text")
    if value is None and block.get("type") == "list":
        value = block.get("list_items", [])
    if isinstance(value, list):
        value = " ".join(str(part) for part in value)
    return re.sub(r"\s+", " ", str(value or "")).strip()


def experimental_footer(block):
    """MinerU can label the last line of a chemical paragraph as a footer."""
    value = _block_text(block)
    return block.get('type') == 'footer' and len(value) > 35 and is_reaction_candidate(value)


def _table_text(block: Dict[str, Any]) -> str:
    value = block.get("table_body") or ""
    value = re.sub(r"</(?:tr|td|th)>", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def _clean_block(value: str) -> str:
    # MinerU margin line numbers occur at the start of individual blocks.
    # Keep numbers in chemical names, quantities, and references.
    return re.sub(r"^\s*\d{1,3}\s+(?=[A-Z][a-z])", "", value).strip()


def _clean_procedure(parts: List[str]) -> str:
    raw = " ".join(_clean_block(part) for part in parts).strip()
    analytical = ANALYTICAL.search(raw)
    if analytical:
        raw = raw[:analytical.start()].rstrip(" ;.,")
    return raw.strip()


def _ordered_indices(blocks: List[Dict[str, Any]]) -> List[int]:
    """Repair pages where MinerU emits heading headers after their body.

    Only such pages are spatially sorted. The original indices remain attached
    to chunks, so result positions still refer to the source content list.
    """
    pages: Dict[int, List[int]] = {}
    for index, block in enumerate(blocks):
        page = int(block.get("page_idx", block.get("page_number", 1) - 1))
        pages.setdefault(page, []).append(index)
    ordered = []
    for page_indices in pages.values():
        late_headings = [i for i in page_indices
                         if (blocks[i].get("type") == "header" or blocks[i].get("text_level"))
                         and heading_match(_block_text(blocks[i]))]
        first_body = next((i for i in page_indices
                           if blocks[i].get("type", "text") == "text"
                           and _block_text(blocks[i])), None)
        late_footers = any(experimental_footer(blocks[i]) for i in page_indices)
        if first_body is None or (not late_footers and (not late_headings or min(late_headings) < first_body)):
            ordered.extend(page_indices)
            continue
        boxes = [blocks[i].get("bbox") for i in page_indices]
        boxes = [box for box in boxes if isinstance(box, list) and len(box) == 4]
        if len(boxes) < 4:
            ordered.extend(page_indices)
            continue
        midpoint = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2

        def sort_key(index: int) -> tuple:
            box = blocks[index].get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                return (2, index, 0)
            center = (box[0] + box[2]) / 2
            return (0 if center < midpoint else 1, box[1], box[0])

        ordered.extend(sorted(page_indices, key=sort_key))
    return ordered


def _source_events(blocks):
    """Split explicit inline step headings without changing source block IDs."""
    for index in _ordered_indices(blocks):
        block = blocks[index]
        text = block.get('text', '')
        letter = re.match(r'^\s*(?:\[\d+\]\s*)?([a-g])\)\s*([^。]+)。\s*(.+)$', text, re.S | re.I)
        from .step_context import _product_heading
        if (letter and block.get('type') in ('text', 'header')
                and (letter[2].startswith('外消旋') or _product_heading(letter[2]))
                and (FALLBACK_OPERATION.search(letter[3]) or re.search(r'按照|类似', letter[3]))):
            yield index, dict(block, text=f'Step {letter[1]}: {letter[2]}')
            yield index, dict(block, text=letter[3])
            continue
        cuts = [m.start() for m in re.finditer(r'(?<![A-Za-z0-9])Step\s+\d+[A-Za-z]?\s*[:：]', text, re.I)]
        cuts = [c for c in cuts if text[:c].strip()]
        if cuts and block.get('type') in ('text', 'header'):
            ends = [0, *cuts, len(text)]
            for start, end in zip(ends, ends[1:]):
                yield index, dict(block, text=text[start:end].strip())
        else:
            yield index, block


def segment_blocks(blocks: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Segment ordered blocks, keeping continuations across PDF page boundaries."""
    chunks: List[Dict[str, Any]] = []
    current = None
    occurrences: Dict[str, int] = {}
    used_ids: set = set()
    parent_example_id = None

    def claim_id(candidate: str) -> str:
        # A repeated-heading suffix can alias another heading's parsed id
        # (e.g. a 5th "Example C" becomes example_c_5, colliding with a real
        # "Example C-5"). First emission keeps the id; later collisions get
        # an _altN tail that no heading parse can produce.
        while candidate in used_ids:
            base, n = candidate, 2
            while f"{base}_alt{n}" in used_ids:
                n += 1
            candidate = f"{base}_alt{n}"
        used_ids.add(candidate)
        return candidate

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        parts = current.pop("_parts")
        raw = " ".join(parts).strip()
        markers = list(re.finditer(r"\bProcedure\s+(\d+)\s*:", raw, re.I))
        if len(markers) > 1:
            records = []
            for i, marker in enumerate(markers):
                end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
                part = raw[marker.end():end].strip()
                record = dict(current)
                record["parent_id"] = current["id"]
                record["subprocedure"] = int(marker.group(1))
                record["id"] = claim_id(f"{current['id']}_p{marker.group(1)}")
                record["procedure_raw"] = part
                record["text_parts"] = [part]
                records.append(record)
        else:
            current["procedure_raw"] = raw
            current["text_parts"] = parts
            current["_parts_clean"] = parts
            records = [current]
        for record in records:
            record["block_indices"] = list(dict.fromkeys(record["block_indices"]))
            record["procedure"] = _clean_procedure(record.pop("_parts_clean", None) or [record["procedure_raw"]])
            if not record["procedure"] and re.search(
                    r"procedure|measured|prepared|制备|测定|加入|反应", record["title"], re.I):
                record["procedure"] = record["title"]
            record["text"] = (record["title"] + "\n" + record["procedure"]).strip()
            record["references"] = _references(record["title"] + " " + record["procedure"])
            record["has_procedure"] = bool(record["procedure"] or record["tables"])
            chunks.append(_finish_chunk(record))
        current = None

    figures = {}
    for block in blocks:
        captions = _captions(block) + list(block.get('chart_caption') or [])
        if block.get('type') in ('image', 'chart') and any(
                re.match(r'^\s*(?:Fig(?:ure)?\.?|图)\s*\d+', str(c), re.I) for c in captions):
            figures.setdefault(block.get('page_idx'), []).append(block)
    events = []
    for index, block in _source_events(blocks):
        if block.get("type") == "list" and isinstance(block.get("list_items"), list):
            events.extend((index, dict(block, type="text", text=str(item)))
                          for item in block["list_items"] if str(item).strip())
        else:
            events.append((index, block))

    for index, block in events:
        block_type = block.get("type", "text")
        if block_type == 'page_number' or (block_type == 'footer' and not experimental_footer(block)):
            continue
        page = int(block.get("page_idx", block.get("page_number", 1) - 1)) + 1
        caption = next((c for c in _captions(block) if heading_match(c)), "")
        if block_type in ("image", "chart", "table") and not caption:
            if current:
                current['page_end'] = page
                current['block_end'] = index
                current['block_indices'].append(index)
                if block.get("img_path"):
                    current["images"].append(block["img_path"])
                if block_type in ('image', 'chart'):
                    current['_parts'].extend(image_annotations(block))
                if block_type == "table":
                    body = _table_text(block)
                    if body:
                        current["tables"].append({"block_idx": index, "text": body})
                        current["_parts"].append(body)
            continue
        value = caption or _block_text(block)
        section_heading = UNNUMBERED_HEADING.match(value)
        keep_section = section_heading and section_heading.group('section_kind')
        if (SECTION_END.match(value) and not keep_section) or re.match(r"^(?:\[\d{1,4}\]\s*)?效果实施例", value):
            finish()
            continue
        if not value or LINE_NUMBER.fullmatch(value):
            continue
        match = heading_match(value)
        if _figure_legend(block, match, figures):
            continue
        unnumbered = None if match else (UNNUMBERED_HEADING.match(value)
                                        if len(value) <= 180 else None)
        if block_type == "header" and not (match or unnumbered):
            continue
        if match or unnumbered:
            finish()
            kind = (match.group("kind") if match else
                    (unnumbered.group("kind") or unnumbered.group("kind_zh") or unnumbered.group('section_kind'))).lower()
            heading_type = kind.replace(' ', '_')
            kind = {"实施例": "example", "实例": "example", "制备例": "example",
                    "合成例": "example", "中间物": "intermediate", "中间体": "intermediate", "步骤": "step",
                    "制备步骤": "step", "制备": "preparation",
                    "合成": "synthesis"}.get(kind, heading_type)
            number = _number(match.group("number")) if match else 0
            letter = match.group("letter").lower() if match else ""
            base_id = f"{heading_type if 'example' in heading_type else kind}_{number}{letter}"
            if match and match.group('prime'):
                base_id += '_prime'
            if 'example' in heading_type or kind in ('example', 'compound', 'boronate_ester', 'intermediate'):
                # Reference/Comparative Example headings own their steps too;
                # leaving the parent scope on the prior example lets "Step 1"
                # resolve against the wrong instance.
                parent_example_id = base_id
            occurrences[base_id] = occurrences.get(base_id, 0) + 1
            suffix = f"_{occurrences[base_id]}" if occurrences[base_id] > 1 else ""
            title = (match.group("title") if match else
                     (unnumbered.group("title") or unnumbered.group("title_zh") or unnumbered.group('section_kind'))).strip()
            title = re.sub(r"\s*\[\d{1,4}\]\s*$", "", title).strip()
            inline_procedure = title if FALLBACK_OPERATION.search(title) else ''
            inherited_heading = re.match(r'^(.+?)。\s*((?:与(?:上述)?方法|按照|采用|参照).+)$', title)
            separate_inheritance = inherited_heading and not FALLBACK_OPERATION.search(inherited_heading[1])
            if separate_inheritance:
                title, inline_procedure = inherited_heading[1], inherited_heading[2]
            if inline_procedure and not separate_inheritance:
                # A heading and its recipe can occupy the same OCR paragraph.
                # Keep the recipe as evidence, rather than a chemical title.
                title = ''
            if block_type in ('image', 'chart') and kind == 'step' and not title:
                from .step_context import _product_heading
                annotations = image_annotations(block)
                if (sum(bool(heading_match(c)) for c in _captions(block)) == 1
                        and len(annotations) == 1 and _product_heading(annotations[0])
                        and not re.search(r'starting material|reagent|catalyst', annotations[0], re.I)):
                    title = annotations[0]
            current = {
                "id": claim_id(base_id + suffix), "reference_id": base_id,
                "kind": kind, "number": number,
                "heading_type": heading_type,
                "parent_example_id": parent_example_id if kind == 'step' else None,
                "title": title, "source": source, "page_start": page,
                "title_source": {"text": title, "block_idx": index, "page_idx": page - 1,
                                 "bbox": block.get('bbox'), "text_level": block.get('text_level')},
                "page_end": page, "block_start": index, "block_end": index,
                "images": [], "tables": [], "block_indices": [index],
                "_parts": [inline_procedure] if inline_procedure else [],
            }
            if block_type in ('image', 'chart'):
                if block.get('img_path'):
                    current['images'].append(block['img_path'])
                current['_parts'].extend(c for c in image_annotations(block) if c != title)
            if block_type == "table":
                body = _table_text(block)
                if body:
                    current["tables"].append({"block_idx": index, "text": body})
                    current["_parts"].append(body)
                if block.get("img_path"):
                    current["images"].append(block["img_path"])
            continue
        if current is None:
            continue
        current["page_end"] = page
        current["block_end"] = index
        current["block_indices"].append(index)
        if (not current["title"] and not current["_parts"]
                and not FALLBACK_OPERATION.search(value)
                and not re.search(r'\b(?:was|were|obtained|determined|measured)\b|[。；;]|[.!?](?:\s|$)', value, re.I)):
            current["title"] = value
            current['title_source'] = {'text': value, 'block_idx': index, 'page_idx': page - 1,
                                       'bbox': block.get('bbox'), 'text_level': block.get('text_level')}
        else:
            current["_parts"].append(value)
    finish()
    return chunks



# A conceptual scheme narrative: letter/number-coded compounds chained by
# transformation verbs, without worked-example amounts.
_SCHEME_CODE = re.compile(
    r'[\(（](?P<bracket>[IVXLCDM]{1,12}(?:[-′\'][A-Za-z0-9]{1,4})?'
    r'|[A-Z]{1,2}(?:[-′\'][A-Za-z0-9]{1,4})?|[A-Z]\d{1,3}[a-z]?'
    r'|\d{1,3}[a-z]?(?:[-′\'][A-Za-z0-9]{1,4})?)[\)）]'
    r'|(?i:\b(?:formulas?|compounds?)(?:\s+of)?(?:\s+the)?\s+(?:formula\s+)?)'
    r'(?P<english>[IVXLCDM]{1,12}(?:-[A-Za-z0-9]{1,4})?|[A-Z]\d{1,3}[a-z]?|\d{1,3}[a-z]?)(?![A-Za-z0-9-])'
    r'|(?:通式|化合物|产物|产品|中间体|式)\s*'
    r'(?P<chinese>[一二三四五六七八九十百]{1,3}|[A-Z]\d{0,3}[a-z]?|\d{1,3}[a-z]?)(?![A-Za-z0-9-])'
    r'|(?<=[\u4e00-\u9fff])(?P<bare>[A-QS-Z]\d{1,3}[a-z]?)(?![A-Za-z0-9-])')
_SCHEME_TRANSFORM = re.compile(
    r'\b(?:may|can|is|are)\s+(?:react|be\s+reacted|be\s+prepared|be\s+synthesi[sz]ed'
    r'|be\s+treated|be\s+condensed|be\s+converted|be\s+deprotected)\b'
    r'|\breact(?:s|ed|ing)?\s+(?:[^.;\n]{1,100}?\s+)?with\b|\bgiving\b'
    r'|\bto\s+(?:prepare|give|afford|obtain|synthesi[sz]e|provide|form)\b|resultant\s+compound'
    r'|\b(?:coupl(?:e|es|ed|ing)|condens(?:e|es|ing|ation)|condensed(?=\s+(?:with|to|using|in|under|and)\b)|deprotect(?:s|ed|ing|ion)'
    r'|reduc(?:e|es|ed|ing|tion)|oxidiz(?:e|es|ed|ing|ation)|alkylat(?:e|es|ed|ing|ion)'
    r'|brominat(?:e|es|ed|ing|ion)|silylat(?:e|es|ed|ing|ion))\b'
    r'|与.{0,60}反应|经.{0,20}反应|得到|制得|生成|偶合|偶联|取代反应|取代(?=[^。；;]{0,60}(?:得到|生成|制得))|酰化|还原|环化|缩合|消除'
    r'|转化为|脱保护|以提供|酯化|关环|成环|水解', re.I)
_SCHEME_AMOUNT = re.compile(r'\d+(?:\.\d+)?\s*(?:mg|g|kg|mmol|m[lL]|毫克|毫升|克)\b', re.I)
# A literature pointer names a compound and defers its preparation to a
# citation ("can be obtained as described in WO 2002/068420"): it carries
# scheme-shaped codes and verbs but describes no transformation.
_SCHEME_LITERATURE_POINTER = re.compile(
    r'\b(?:can|may)\s+be\s+(?:prepared|obtained|synthesi[sz]ed|made)\b'
    r'[^.。;；\n]{0,140}?(?:as\s+described|described\s+in\s+the\s+literature'
    r'|in\s+the\s+publications?|known\s+to\s+(?:the\s+)?skilled)'
    r'|文献(?:中)?(?:已)?(?:报道|记载|公开)(?:的)?(?:方法|制备|合成)', re.I)
_SCHEME_BOUNDARY = re.compile(
    r'^(?:\[\d+\]\s*)?(?:本(?:申请|发明)的第.{1,5}方面|药物组合物|权利要求|'
    r'What is claimed|Claims\b|Biological (?:Activity|Evaluation))', re.I)


def scheme_codes(text):
    """Read complete scheme identities, shared by detection and route edges."""
    clean = _de_latex(text)
    codes = []
    for match in _SCHEME_CODE.finditer(clean):
        prefix = clean[:match.start()]
        if match['bracket']:
            if re.search(r'(?:碳|氢|氧|氮|硫|硼|硅|铝|钯|铂|铁|铜|镍|铑|钌|铱|锰|钴|palladium|platinum|iron|copper|nickel|'
                         r'rhodium|ruthenium|iridium|manganese|cobalt)\s*$', prefix, re.I):
                continue
            if (re.search(r'(?:^|[^A-Za-z])(?:C|N|O|S|B|P|CH|NH|OH)$', prefix)
                    or re.match(r'\s*(?:或\s*[\(（]S[\)）]\s*)?(?:构型|configuration\b)', clean[match.end():])):
                continue
            # List markers and example sub-step references are not molecules.
            if ((match['bracket'].isdigit() and clean[match.end():].strip()
                 and (not prefix.rsplit('\n', 1)[-1].strip()
                      or (re.search(r'[,，;；:]\s*$', prefix)
                          and not re.search(r'[\)）]\s*[,，]\s*$', prefix))))
                    or re.search(r'\b(?:Example|Step)\s*\d+\s*[-—–]?\s*$', prefix, re.I)):
                continue
        code = next(v for v in match.groupdict().values() if v is not None).upper()
        codes.append(code)
        if match['chinese'] or match['english']:
            tail = clean[match.end():]
            while (item := re.match(r'\s*(?:和|及|、|,|\band\b|\bor\b)\s*(\d{1,3}[a-z]?)(?![A-Za-z0-9])', tail, re.I)):
                codes.append(item[1].upper())
                tail = tail[item.end():]
    return list(dict.fromkeys(codes))


def _scheme_parts(block):
    values = block.get('list_items') if block.get('type') == 'list' else [block.get('text') or '']
    for value in values or []:
        for part in re.split(r'\n\s*\n|(?=\[\d{4}\])', str(value)):
            part = part.strip()
            if _SCHEME_BOUNDARY.match(part) or SECTION_END.match(part):
                return
            if part:
                yield part


def is_scheme_description(text: str) -> bool:
    """A general-scheme paragraph describes a synthetic strategy, not a
    worked example: coded compounds chained by transformation verbs and no
    experimental amounts.  A paragraph whose every transform defers to a
    literature citation is a pointer list, not a scheme."""
    clean = _de_latex(text)
    codes = len(scheme_codes(clean))
    transforms = len(_SCHEME_TRANSFORM.findall(clean))
    if transforms and transforms <= len(_SCHEME_LITERATURE_POINTER.findall(clean)):
        return False
    return codes >= 2 and transforms >= 1 and not _SCHEME_AMOUNT.search(clean)


def scheme_chunks(blocks: List[Dict[str, Any]], source: str,
                  covered: set) -> List[Dict[str, Any]]:
    """Collect conceptual scheme narratives the procedure gate always skips.

    The fallback pass requires operation and amount evidence, so a scheme
    paragraph ("may be prepared by the following Processes", 方案A为合成…)
    never becomes a chunk. Collect them explicitly as kind="scheme" so the
    scheme-template extraction can see them."""
    chunks: List[Dict[str, Any]] = []
    used: set = set()
    taken: set = set()
    for index, block in enumerate(blocks):
        recovered = block.get('markdown_scheme_paragraphs') or []
        if (index in covered and not recovered) or index in used or index in taken:
            continue
        if block.get("type") not in ("text", "list") and not recovered:
            continue
        regular = '\n'.join(_scheme_parts(block)) if block.get('type') in ('text', 'list') and index not in covered else ''
        value = '\n'.join(filter(None, [regular, *(p['text'] for p in recovered)]))
        if not value or not is_scheme_description(value):
            continue
        window = [(index, int(block.get("page_idx", 0)) + 1, value)]
        taken.add(index)
        for nxt in range(index + 1, len(blocks)):
            if (nxt in covered or nxt in taken or blocks[nxt].get("type") not in ("text", "list")):
                break
            text = '\n'.join(_scheme_parts(blocks[nxt]))
            if not text or heading_match(text) or SECTION_END.match(text):
                break
            if not (is_scheme_description(text) or (scheme_codes(text) and _SCHEME_TRANSFORM.search(text))):
                break
            window.append((nxt, int(blocks[nxt].get("page_idx", 0)) + 1, text))
            taken.add(nxt)
        raw = " ".join(item[2] for item in window)
        related_idx = []
        for start, step in ((window[0][0] - 1, -1), (window[-1][0] + 1, 1)):
            i = start
            if step == -1 and i >= 0 and re.match(r'^(?:\[\d+\]\s*)?其中', _block_text(blocks[i])):
                i -= 1
            while 0 <= i < len(blocks) and blocks[i].get("type") in ("image", "chart", "table"):
                if blocks[i].get('markdown_scheme_paragraphs'):
                    break
                related_idx.append(i)
                i += step
        if recovered and block.get('img_path'):
            related_idx.append(index)
        related = [blocks[i].get("img_path") for i in sorted(related_idx) if blocks[i].get("img_path")]
        base = f"scheme_{window[0][0]}"
        chunk_id, n = base, 2
        while chunk_id in taken or any(chunk_id == c["id"] for c in chunks):
            chunk_id = f"{base}_alt{n}"
            n += 1
        chunk = _finish_chunk({
            "id": chunk_id, "reference_id": base,
            "kind": "scheme", "number": window[0][0],
            "title": "", "source": source,
            "page_start": window[0][1], "page_end": window[-1][1],
            "block_start": window[0][0], "block_end": window[-1][0],
            "images": list(dict.fromkeys(related)), "tables": [],
            "block_indices": sorted({item[0] for item in window}),
            "procedure_raw": raw, "procedure": raw,
            "text_parts": [item[2] for item in window], "text": raw,
            "source_parts": ([dict(text=regular, block_idx=index, page_idx=block.get('page_idx'),
                                   source_role='procedure')] if recovered and regular else []) +
                            ([dict(p, block_idx=index, page_idx=block.get('page_idx'),
                                   source_role='markdown') for p in recovered] if recovered else
                             [dict(text=item[2], block_idx=item[0], page_idx=item[1] - 1,
                                   source_role='procedure') for item in window]) +
                            ([dict(text=item[2], block_idx=item[0], page_idx=item[1] - 1,
                                   source_role='procedure') for item in window[1:]] if recovered else []),
            "references": _references(raw), "has_procedure": True,
        })
        # A scheme narrative is a synthesis description by construction; the
        # amount-less text would classify as unverified.
        chunk["process_kind"] = "synthesis"
        chunks.append(chunk)
    return chunks


def _recover_markdown_schemes(blocks, source):
    """Recover missing numbered scheme prose without changing source indices."""
    suffix = '_content_list.json'
    if not source.name.endswith(suffix):
        return blocks
    markdown = source.with_name(source.name[:-len(suffix)] + '.md')
    if not markdown.exists():
        return blocks
    text = markdown.read_text(encoding='utf-8')
    markers = list(re.finditer(r'(?m)^\[(\d{4})\]\s*', text))
    existing = {}
    for index, block in enumerate(blocks):
        value = _block_text(block) + ' ' + ' '.join(_captions(block))
        for number in re.findall(r'\[(\d{4})\]', value):
            existing.setdefault(number, set()).add(index)
    for pos, marker in enumerate(markers):
        number = marker[1]
        if number in existing:
            continue
        end = markers[pos + 1].start() if pos + 1 < len(markers) else len(text)
        paragraph = re.split(r'\n\s*\n', text[marker.start():end], maxsplit=1)[0].strip()
        # Supplement reaction prose only. A strategy introduction may mention
        # several formula ranges without specifying any chemical operation.
        if not is_scheme_description(paragraph) or not re.search(
                r'反应.{0,30}(?:生成|得到)|取代.{0,60}得到|转化为|\breact(?:ing|s|ed)?\b', paragraph, re.I):
            continue
        before = next((m for m in reversed(markers[:pos]) if len(existing.get(m[1], ())) == 1), None)
        after = next((m for m in markers[pos + 1:] if len(existing.get(m[1], ())) == 1), None)
        if before is None or after is None or not int(before[1]) < int(number) < int(after[1]):
            continue
        anchor, following = next(iter(existing[before[1]])), next(iter(existing[after[1]]))
        if anchor >= following:
            continue
        blocks[anchor].setdefault('markdown_scheme_paragraphs', []).append({
            'text': paragraph, 'paragraph_number': number,
            'markdown_source': str(markdown), 'markdown_line': text.count('\n', 0, marker.start()) + 1,
            'anchor_paragraphs': [before[1], after[1]]})
    return blocks


def select_procedure_chunks(blocks: List[Dict[str, Any]], source: str,
                            segmented: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep titled units and separate experimental procedures before the first title."""
    from .step_context import attach_context
    if not segmented:
        fallback = fallback_procedures(blocks, source)
        scheme = scheme_chunks(blocks, source,
                               {i for c in fallback for i in c['block_indices']})
        return attach_context(fallback + scheme, blocks, source)
    contextual = attach_context(segmented, blocks, source)
    titled = [chunk for chunk in contextual if chunk["has_procedure"]]
    covered = {i for c in titled for i in c['block_indices']}
    # Preserve original indices while preventing fallback from crossing a titled unit.
    uncovered = [dict(b) if i not in covered else {'type': 'covered', '_covered': True,
                  'page_idx': b.get('page_idx', 0)} for i, b in enumerate(blocks)]
    extra = fallback_procedures(uncovered, source)
    extra = [c for c in extra if not covered.intersection(c['block_indices']) and c['process_kind'] == 'synthesis']
    first = min(c['block_start'] for c in segmented)
    starts = [i for i, b in enumerate(blocks[:first]) if re.search(
        r'通用程序|General Procedures?|^具体实施方式$|^EXAMPLES$|^Experimental(?: Section)?$', _block_text(b), re.I)]
    prefix_start = min(starts) if starts else first
    extra = [c for c in extra if c['block_end'] >= first or c['block_start'] >= prefix_start
             or (re.search(r'(?:afforded|obtained|yielded|得到|制得).{0,150}\d+(?:\.\d+)?\s*[%％]', c['text'], re.I)
                 and not re.search(r'所述|权利要求|wherein|present invention|in some embodiments', c['text'], re.I))
             or (re.search(r'\d+(?:\.\d+)?\s*(?:mg|g|mmol)\b', c['text'], re.I)
                 and not re.search(r'所述|权利要求|wherein|present invention|in some embodiments|can be|may be', c['text'], re.I)
                 and classify_process('', c['text']) == 'synthesis')]
    scheme = scheme_chunks(blocks, source,
                           covered | {i for c in extra for i in c['block_indices']})
    return sorted(titled + attach_context(extra + scheme, blocks, source),
                  key=lambda c: (c['page_start'], c['block_start']))


def coverage_gaps(blocks, chunks):
    covered = {i for c in chunks for i in c.get('block_indices', [])}
    gaps = []
    for i, b in enumerate(blocks):
        value = _block_text(b) or ' '.join(_captions(b))
        if i not in covered and not SECTION_END.match(value) and classify_process('', value) == 'synthesis':
            gaps.append({'block_idx': i, 'page_idx': b.get('page_idx'), 'reason': 'uncovered_experimental_text'})
    return gaps

def load_blocks(path: str) -> List[Dict[str, Any]]:
    """Load MinerU content_list.json or ChemReX pdf pages.json."""
    from .source_paths import resolve_source
    source = resolve_source(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return _recover_markdown_schemes(data, source)
    if isinstance(data, dict) and "pages" in data:
        blocks = []
        for page in data["pages"]:
            number = page["page_number"]
            for paragraph in re.split(r"\n\s*\n", page.get("text", "")):
                for line in paragraph.splitlines():
                    if line.strip():
                        blocks.append({"type": "text", "page_idx": number - 1, "text": line})
            for image in page.get("images", []):
                blocks.append({"type": "image", "page_idx": number - 1, "img_path": image})
        return blocks
    raise ValueError("source must be MinerU content_list.json or ChemReX pages.json")
