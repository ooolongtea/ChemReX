"""Local, bounded Chinese/English detection for the shared PDF worker.

Count prose scripts rather than document names or patent country codes. A
Chinese character carries roughly a word's worth of evidence; discounting
Latin letters avoids chemical names overwhelming otherwise Chinese prose.
No document text is sent to a model for language detection.
"""
import json
import re
from pathlib import Path
from typing import Optional

HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]')
LATIN = re.compile(r'[A-Za-z]')
MAX_TEXT = 128_000


def detect_text_language(text: str) -> Optional[str]:
    han = len(HAN.findall(text))
    latin = len(LATIN.findall(text))
    if han >= 8 and han * 4 >= latin:
        return 'zh'
    if latin >= 20:
        return 'en'
    return None


def detect_pdf_language(path: str) -> Optional[str]:
    """Probe embedded text across the document; sparse/scanned PDFs defer to OCR."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path, strict=False)
        if reader.is_encrypted and not reader.decrypt(''):
            return None
        count = len(reader.pages)
        indexes = sorted({round(i * (count - 1) / min(count - 1, 31))
                          for i in range(min(count, 32))}) if count > 1 else range(count)
        parts = []
        for index in indexes:
            text = (reader.pages[index].extract_text() or '')[:4000]
            if len(HAN.findall(text)) >= 40 or len(LATIN.findall(text)) >= 300:
                parts.append(text)
        text = '\n'.join(parts)
        # A scan's embedded page numbers/English headers are not enough to
        # route its Chinese body. Read the parsed body in that case.
        if len(HAN.findall(text)) < 40 and len(LATIN.findall(text)) < 300:
            return None
        return detect_text_language(text)
    except Exception:
        # Missing optional readers, encryption and broken text layers still
        # get the normal parser's full attempt.
        return None


def detect_parsed_language(path: str) -> Optional[str]:
    blocks = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(blocks, dict):
        if 'pages' in blocks:
            blocks = [{'type': 'text', 'text': page.get('text', '')} for page in blocks['pages']]
        else:
            blocks = blocks.get('content_list', blocks.get('blocks', []))
    if not isinstance(blocks, list):
        raise ValueError('Parsed PDF content must be a block list')
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        # Ignore image paths, table markup, formulas and parser metadata.
        text = block.get('text')
        if not text and isinstance(block.get('list_items'), list):
            text = '\n'.join(item for item in block['list_items'] if isinstance(item, str))
        if isinstance(text, str) and block.get('type') not in ('image', 'table', 'equation'):
            parts.append(text[:4000])
    # Sample throughout a long document instead of privileging its cover.
    if len(parts) > 64:
        parts = [parts[round(i * (len(parts) - 1) / 63)] for i in range(64)]
    return detect_text_language('\n'.join(parts)[:MAX_TEXT])
