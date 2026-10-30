"""Position matching — map procedure units and reference sentences back to
MinerU content_list blocks (page_idx + bbox).

Depends on the ordered blocks produced by ``segmenter.load_blocks`` (MinerU
``*_content_list.json`` or ChemReX pages.json). Two entry points:

    ``match_chunk(chunk)``    — position of a title+procedure unit. Primary path
                                 is the block-index range recorded by the
                                 segmenter; a normalized text match on the unit
                                 title is the fallback.
    ``match_reference(text)`` — position of an arbitrary reference sentence
                                 (exact match first, then most-specific
                                 containment match).

All returned ``page_idx`` values follow the MinerU 0-based convention.
"""

import re
from typing import Any, Dict, List, Optional

from .segmenter import HEADING, _block_text, experimental_footer


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _bibliographic(text: str) -> bool:
    return bool(re.search(r"et al\.|\bvol\.|\bdoi\b|Annual Meeting|"
                          r"Clinical Trials?:|https?://|clinicaltrials\.gov|"
                          r"\bNCT\d{6,}\b", text, re.I))


class PositionMatcher:
    """Match segmented units / reference text to content_list positions."""

    def __init__(self, blocks: List[Dict[str, Any]]):
        self.blocks = blocks or []
        # (block index, normalized text) for every content-bearing block,
        # skipping page furniture exactly like the segmenter does.
        self._text_blocks: List[List[Any]] = []
        for index, block in enumerate(self.blocks):
            if block.get('type') == 'page_number' or (block.get('type') == 'footer' and not experimental_footer(block)):
                continue
            if block.get("type") == "header" and not HEADING.match(_norm(_block_text(block))):
                continue
            text = _norm(_block_text(block))
            if text and not _bibliographic(text):
                self._text_blocks.append([index, text])

    # ── position of a title+procedure unit ──
    def match_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        indices = chunk.get("block_indices")
        positions = (self._positions_for_indices(indices)
                     if isinstance(indices, list) else
                     self._positions_in_range(chunk.get("block_start"), chunk.get("block_end")))
        method = "block_index" if positions else None

        if not positions:
            positions = self._match_chunk_by_text(chunk)
            method = "text" if positions else None

        first = positions[0] if positions else None
        pages = sorted({p["page_idx"] for p in positions if p.get("page_idx") is not None})
        fallback_page = max(int(chunk.get("page_start", 1)) - 1, 0)
        return {
            "page_idx": first["page_idx"] if first else fallback_page,
            "bbox": first.get("bbox") if first else None,
            "match_method": method or "none",
            "pages": pages or [fallback_page],
            "blocks": positions,
        }

    # ── position of a reference sentence (RetroParser-style) ──
    def match_reference(self, reference: str) -> Optional[Dict[str, Any]]:
        text = _norm(reference)
        if not text:
            return None
        for index, block_text in self._text_blocks:
            if block_text == text:
                return self._position(index, match="exact")
        # containment: first occurrence in reading order (patents often repeat
        # the abstract inside the description; the first hit is the source)
        for index, block_text in self._text_blocks:
            if text in block_text:
                return self._position(index, match="contains")
        return None

    # ── document head text (abstract region, before the first Example) ──
    def head_text(self, language: str = "en", max_chars: Optional[int] = None) -> str:
        """Use the abstract plus invention statements before the first Example."""
        limit = max_chars if max_chars is not None else 18000
        lead_limit = 4000
        lead: List[str] = []
        selected: List[str] = []
        priority: List[tuple] = []
        lead_size = selected_size = 0
        for block in self.blocks:
            if block.get("type") in ("footer", "page_number", "ref_text"):
                continue
            text = _norm(_block_text(block))
            if not text or HEADING.match(text):
                if text and HEADING.match(text):
                    break
                continue
            if block.get("type") == "header":
                continue
            if _bibliographic(text) or re.search(r"^References\b|\bExample\s+\d", text, re.I):
                continue
            score = (3 if re.search(r"(?:particularly|more)\s+preferred.{0,160}\bis\b|"
                                    r"preferred.{0,160}\bcompound\b", text, re.I) else
                     2 if re.search(r"also known as|化学名称|目标化合物|最终产物", text, re.I) else
                     1 if re.search(r"active ingredient|主要活性成分", text, re.I) else 0)
            if score:
                priority.append((score, text))
            if lead_size < lead_limit:
                lead.append(text)
                lead_size += len(text) + 1
            elif (re.search(r"invention|preferred|embodiment|final product|本发明|优选|目标化合物",
                            text, re.I) and selected_size < limit - lead_limit):
                selected.append(text)
                selected_size += len(text) + 1
        chosen = []
        size = 0
        seen = set(lead)
        for score, text in sorted(priority, key=lambda item: -item[0]):
            if text not in seen and size + len(text) + 1 <= min(6000, max(limit - lead_limit, 0)):
                chosen.append(text)
                seen.add(text)
                size += len(text) + 1
        return "\n".join(lead + chosen + selected)[:limit]

    # ── helpers ──
    def _positions_for_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        positions = []
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < len(self.blocks):
                continue
            block = self.blocks[index]
            if block.get('type') == 'page_number' or (block.get('type') == 'footer' and not experimental_footer(block)):
                continue
            if block.get("type") == "header" and not HEADING.match(_norm(_block_text(block))):
                continue
            positions.append(self._position(index))
        return positions

    def _positions_in_range(self, start: Any, end: Any) -> List[Dict[str, Any]]:
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            return []
        if start < 0 or end < start or start >= len(self.blocks):
            return []
        positions = []
        for index in range(start, min(end, len(self.blocks) - 1) + 1):
            block = self.blocks[index]
            if block.get('type') == 'page_number' or (block.get('type') == 'footer' and not experimental_footer(block)):
                continue
            if block.get("type") == "header" and not HEADING.match(_norm(_block_text(block))):
                continue
            positions.append(self._position(index))
        return positions

    def _match_chunk_by_text(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        title = _norm(chunk.get("title"))
        probe = title or _norm(chunk.get("procedure", ""))[:80]
        if not probe:
            return []
        for index, block_text in self._text_blocks:
            if probe == block_text or (len(probe) >= 20 and probe in block_text):
                return [self._position(index, match="title")]
        return []

    def _position(self, index: int, match: Optional[str] = None) -> Dict[str, Any]:
        block = self.blocks[index]
        info = {
            "block_idx": index,
            "page_idx": block.get("page_idx"),
            "bbox": block.get("bbox"),
            "type": block.get("type", "text"),
        }
        if match:
            info["match"] = match
        return info
