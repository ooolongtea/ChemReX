"""ChemReX Module 3 — Data standardization.

Consumes extraction NER+Action output, produces a unified structured JSON:
- field names normalized (NER_FIELDS)
- multi-value fields → list, scalar fields → str | null
- empty/null/n/a → null
- action → ordered list
- compound names → {name, standard_name, smiles} via LLM standardize + OPSIN + RDKit cascade
"""
from .standardizer import Standardizer

__all__ = ["Standardizer"]
