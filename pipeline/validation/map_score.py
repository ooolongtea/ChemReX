"""RXNMapper scoring wrapper."""

from __future__ import annotations

from typing import List

from rxnmapper import RXNMapper

from .build_candidates import ReactionCandidate


class MappingScorer:
    """Score candidate reaction expressions with RXNMapper."""

    def __init__(self):
        self.mapper = RXNMapper()

    def score(self, candidates: List[ReactionCandidate]) -> List[ReactionCandidate]:
        if not candidates:
            return candidates

        expressions = [candidate.expression for candidate in candidates]
        results = self.mapper.get_attention_guided_atom_maps(expressions)
        for candidate, result in zip(candidates, results):
            candidate.mapped_rxn = result.get("mapped_rxn", "")
            candidate.confidence = float(result.get("confidence") or 0.0)
        return candidates
