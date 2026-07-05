"""ChemReX Module 2 — Image mining (compound dictionary + reaction structures).

Self-contained: MolDetect/ and RxnScribe/ are bundled here (external projects, independently refactored).
Only a supplementary evidence source — produces dict_result (ID→SMILES) and image_result (reactions).
No PDF layout parsing, no page_idx/bbox position linking.
"""
from .image_pipeline import ImagePipeline

__all__ = ["ImagePipeline"]
