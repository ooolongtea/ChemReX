"""ChemReX Module 4 — Coreference resolution (ID→SMILES mapping).

Resolves compound identifiers (e.g. "化合物(3)", "Compound I") in NER fields to
SMILES from the image-side compound dictionary. Language-layer coreference is
out of scope.
"""

from .coref_resolver import CorefResolver

__all__ = ["CorefResolver"]
