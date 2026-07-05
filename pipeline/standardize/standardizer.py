"""Standardizer — unify extraction NER+Action into structured JSON.

- compound fields (REACTION_PRODUCT/STARTING_MATERIAL/REAGENT_CATALYST/SOLVENT): list[str] → list[{name, standard_name, smiles}]
- scalar fields (TEMPERATURE/TIME/YIELD_PERCENT): str | null
- action: ordered list (unchanged order)
- null/empty/n-a normalized to null for scalars, [] for lists
"""

from typing import Any, Dict, List

from loguru import logger

from .compound_standardizer import CompoundStandardizer

COMPOUND_FIELDS = ("REACTION_PRODUCT", "STARTING_MATERIAL", "REAGENT_CATALYST", "SOLVENT")
SCALAR_FIELDS = ("TEMPERATURE", "TIME", "YIELD_PERCENT")


class Standardizer:
    """Produce unified structured JSON from extraction output."""

    def __init__(self, use_opsin: bool = True):
        try:
            import config
            use_opsin = getattr(config, "STANDARDIZE_USE_OPSIN", use_opsin)
        except ImportError:
            pass
        self.compound_std = CompoundStandardizer(use_opsin=use_opsin)
        logger.debug("Standardizer ready")

    def standardize(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        ner = extraction_result.get("ner", {}) or {}
        action = list(extraction_result.get("action", []) or [])
        language = extraction_result.get("language")
        task = extraction_result.get("task", "both")

        compound_names = self._collect_compound_names(ner)
        std_map = self.compound_std.standardize(compound_names) if compound_names else {}

        out_ner: Dict[str, Any] = {}
        for field in COMPOUND_FIELDS:
            raw = ner.get(field, [])
            if isinstance(raw, str):
                raw = [raw] if raw.strip() else []
            out_ner[field] = [self._enrich_compound(n, std_map) for n in (raw or [])]
        for field in SCALAR_FIELDS:
            val = ner.get(field)
            out_ner[field] = self._norm_scalar(val)

        result = {
            "ner": out_ner,
            "action": action,
            "language": language,
            "task": task,
        }
        logger.info(
            f"Standardize done | compounds enriched={sum(1 for v in std_map.values() if v.get('smiles'))} "
            f"actions={len(action)}"
        )
        return result

    @staticmethod
    def _collect_compound_names(ner: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        for field in COMPOUND_FIELDS:
            raw = ner.get(field, [])
            if isinstance(raw, str):
                raw = [raw] if raw.strip() else []
            for n in (raw or []):
                if isinstance(n, str) and n.strip():
                    names.append(n.strip())
        return list(dict.fromkeys(names))

    @staticmethod
    def _enrich_compound(name: str, std_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        info = std_map.get(name, {})
        return {
            "name": name,
            "standard_name": info.get("standard_name"),
            "smiles": info.get("smiles"),
        }

    @staticmethod
    def _norm_scalar(val: Any) -> Any:
        if val is None:
            return None
        s = str(val).strip()
        if not s or s.lower() in ("null", "none", "n/a", "not specified", "notspecified"):
            return None
        return s
