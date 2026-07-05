"""MolScribe wrapper — batch molecule image → SMILES.

Adapted from RetroParser/ImgAgent/mol_ocr.py:
- model path configurable via config.MOLSCRIBE_CKPT (no hardcoded path)
- self-contained import (no sys.path hacks)
"""

import os
from typing import Dict, List, Optional

import torch

from molscribe import MolScribe


def _resolve_model_path() -> str:
    try:
        import config
        return getattr(config, "MOLSCRIBE_CKPT", "")
    except ImportError:
        return os.getenv("MOLSCRIBE_CKPT", "")


class MolOCR:
    """MolScribe molecule recognition (batch via MolScribe.predict_image_files)."""

    def __init__(self, device: str = "cpu", model_path: Optional[str] = None):
        self.device = torch.device(device)
        self.model_path = model_path or _resolve_model_path()
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"MolScribe model not found: {self.model_path}\n"
                f"Set MOLSCRIBE_CKPT env var or config.MOLSCRIBE_CKPT."
            )
        try:
            self.model = MolScribe(self.model_path, device=self.device)
        except Exception as e:
            raise RuntimeError(f"MolScribe init failed: {e}")

    def predict_image_file(self, image_path: str, return_atoms_bonds: bool = False,
                           return_confidence: bool = False) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.model.predict_image_file(
            image_path, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence
        )

    def predict_image(self, image_path: str, return_atoms_bonds: bool = False,
                      return_confidence: bool = False) -> Optional[str]:
        try:
            result = self.predict_image_file(image_path, return_atoms_bonds, return_confidence)
            if isinstance(result, dict):
                return result.get("smiles")
            return result if isinstance(result, str) else None
        except Exception as e:
            print(f"MolOCR predict error: {e}")
            return None

    def predict_batch(self, image_paths: List[str], return_atoms_bonds: bool = False,
                      return_confidence: bool = False) -> List[Dict]:
        if not image_paths:
            return []
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing images: {missing}")
        try:
            outputs = self.model.predict_image_files(
                image_paths, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence
            )
            return [{"image_path": p, "smiles": r.get("smiles")} for p, r in zip(image_paths, outputs)]
        except Exception as e:
            raise RuntimeError(f"Batch predict failed: {e}")
