"""RxnOCR — RxnScribe detection + external MolOCR + Paddle OCR.

Adapted from RetroParser/ImgAgent/rxn_ocr.py:
- local import from .RxnScribe.rxnscribe.interface (no sys.path hacks)
- model path configurable via config.RXNSCRIBE_CKPT
- paddle_ocr.py co-located in this package
- no hardcoded GPU id
"""

import os
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import torch
from loguru import logger

from .RxnScribe.rxnscribe.interface import RxnScribe

try:
    from rdkit import Chem
except ImportError:
    Chem = None

from .mol_ocr import MolOCR


def _resolve_rxnscribe_ckpt() -> str:
    try:
        import config
        return getattr(config, "RXNSCRIBE_CKPT", "")
    except ImportError:
        return os.getenv("RXNSCRIBE_CKPT", "")


class RxnOCR:
    """RxnScribe-based reaction recognition with external MolOCR + Paddle OCR."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        molocr_device: Optional[str] = None,
        keep_crops: bool = False,
        use_paddle_ocr: bool = True,
        paddle_env: str = "paddle",
    ):
        self.ckpt_path = _resolve_rxnscribe_ckpt()
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"RxnScribe ckpt not found: {self.ckpt_path}\n"
                f"Set RXNSCRIBE_CKPT env var or config.RXNSCRIBE_CKPT."
            )
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = RxnScribe(self.ckpt_path, device=self.device)
        self._image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
        self.keep_crops = keep_crops

        molocr_device_str = molocr_device or (str(self.device) if device else "cuda:0")
        self.mol_ocr = MolOCR(device=molocr_device_str)

        self.use_paddle_ocr = use_paddle_ocr
        self.paddle_env = paddle_env
        self.paddle_ocr_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paddle_ocr.py")
        if use_paddle_ocr and not os.path.exists(self.paddle_ocr_script):
            raise FileNotFoundError(f"Paddle OCR script not found: {self.paddle_ocr_script}")

    def _is_image(self, p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in self._image_exts

    def _filter_image_paths(self, paths: Iterable[str]) -> List[str]:
        return [str(Path(x)) for x in paths if self._is_image(Path(x))]

    def _bbox_norm_to_px(self, bbox, w: int, h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(w - 1, int(round(bbox[0] * w))))
        y1 = max(0, min(h - 1, int(round(bbox[1] * h))))
        x2 = max(0, min(w - 1, int(round(bbox[2] * w))))
        y2 = max(0, min(h - 1, int(round(bbox[3] * h))))
        if x2 <= x1: x2 = min(w - 1, x1 + 1)
        if y2 <= y1: y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    def _bbox_to_key(self, bbox, img_idx: int = 0) -> Optional[Tuple]:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return (img_idx, tuple(round(float(x), 6) for x in bbox))
        return None

    def _canon_smiles(self, smi: str) -> str:
        if not smi or Chem is None:
            return smi
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass
        return smi

    def _call_paddle_ocr(self, image_paths: List[str]) -> List[str]:
        try:
            env = os.environ.copy()
            paddle_gpu = os.environ.get("PADDLE_OCR_GPU")
            if paddle_gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(paddle_gpu)
            cmd = ["conda", "run", "-n", self.paddle_env, "python", self.paddle_ocr_script, "--json"] + image_paths
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            return json.loads(result.stdout.strip())
        except Exception as e:
            logger.error(f"Paddle OCR failed: {e}")
            return ["UNKNOWN"] * len(image_paths)

    def predict_images(
        self,
        image_paths: Iterable[str],
        batch_size: Optional[int] = None,
        molscribe: bool = False,
        ocr_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        files = self._filter_image_paths(image_paths)
        if not files:
            return []
        if batch_size is None:
            batch_size = 16
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        try:
            preds = self.model.predict_image_files(files, batch_size=batch_size, molscribe=False, ocr_type=None)
            if len(preds) != len(files):
                raise RuntimeError(f"Batch count mismatch: {len(preds)} != {len(files)}")
        except (RuntimeError, KeyError, ValueError, IndexError) as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.error(f"Batch detection failed: {str(e)[:200]}")
            return [{"image": f, "result": []} for f in files]

        mol_bbox_map: Dict = {}
        txt_bbox_map: Dict = {}
        global_mol_id = 0
        global_txt_id = 0
        tmp_dir = tempfile.mkdtemp(prefix="rxnocr_batch_")
        position_to_bbox: List[Tuple] = []

        for img_idx, (image_path, pred) in enumerate(zip(files, preds)):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                for rxn_idx, rxn in enumerate(pred):
                    for r_idx, reactant in enumerate(rxn.get("reactants", [])):
                        bbox = reactant.get("bbox")
                        bbox_key = self._bbox_to_key(bbox, img_idx)
                        if bbox_key and bbox_key not in mol_bbox_map:
                            x1, y1, x2, y2 = self._bbox_norm_to_px(bbox, w, h)
                            if x2 - x1 >= 5 and y2 - y1 >= 5:
                                crop = img[y1:y2, x1:x2]
                                crop_path = os.path.join(tmp_dir, f"mol_{global_mol_id}.png")
                                if cv2.imwrite(crop_path, crop):
                                    mol_bbox_map[bbox_key] = (crop_path, global_mol_id)
                                    global_mol_id += 1
                        if bbox_key:
                            position_to_bbox.append((img_idx, "reactant", rxn_idx, r_idx, "mol", bbox_key))
                    for p_idx, product in enumerate(rxn.get("products", [])):
                        bbox = product.get("bbox")
                        bbox_key = self._bbox_to_key(bbox, img_idx)
                        if bbox_key and bbox_key not in mol_bbox_map:
                            x1, y1, x2, y2 = self._bbox_norm_to_px(bbox, w, h)
                            if x2 - x1 >= 5 and y2 - y1 >= 5:
                                crop = img[y1:y2, x1:x2]
                                crop_path = os.path.join(tmp_dir, f"mol_{global_mol_id}.png")
                                if cv2.imwrite(crop_path, crop):
                                    mol_bbox_map[bbox_key] = (crop_path, global_mol_id)
                                    global_mol_id += 1
                        if bbox_key:
                            position_to_bbox.append((img_idx, "product", rxn_idx, p_idx, "mol", bbox_key))
                    for c_idx, condition in enumerate(rxn.get("conditions", [])):
                        bbox = condition.get("bbox")
                        bbox_key = self._bbox_to_key(bbox, img_idx)
                        category = condition.get("category", "")
                        if "[Mol]" in category or "molecule" in category.lower():
                            if bbox_key and bbox_key not in mol_bbox_map:
                                x1, y1, x2, y2 = self._bbox_norm_to_px(bbox, w, h)
                                if x2 - x1 >= 5 and y2 - y1 >= 5:
                                    crop = img[y1:y2, x1:x2]
                                    crop_path = os.path.join(tmp_dir, f"mol_{global_mol_id}.png")
                                    if cv2.imwrite(crop_path, crop):
                                        mol_bbox_map[bbox_key] = (crop_path, global_mol_id)
                                        global_mol_id += 1
                            if bbox_key:
                                position_to_bbox.append((img_idx, "condition", rxn_idx, c_idx, "mol", bbox_key))
                        elif "[Txt]" in category or "text" in category.lower():
                            if bbox_key and bbox_key not in txt_bbox_map:
                                x1, y1, x2, y2 = self._bbox_norm_to_px(bbox, w, h)
                                if x2 - x1 >= 5 and y2 - y1 >= 5:
                                    crop = img[y1:y2, x1:x2]
                                    crop_path = os.path.join(tmp_dir, f"txt_{global_txt_id}.png")
                                    if cv2.imwrite(crop_path, crop):
                                        txt_bbox_map[bbox_key] = (crop_path, global_txt_id)
                                        global_txt_id += 1
                            if bbox_key:
                                position_to_bbox.append((img_idx, "condition", rxn_idx, c_idx, "txt", bbox_key))
            except Exception as e:
                logger.error(f"Image [{img_idx}] error: {str(e)[:150]}")
                continue

        mol_results: Dict = {}
        if mol_bbox_map:
            try:
                sorted_items = sorted(mol_bbox_map.items(), key=lambda x: x[1][1])
                mol_crops = [it[1][0] for it in sorted_items]
                mol_keys = [it[0] for it in sorted_items]
                mol_ocr_results = self.mol_ocr.predict_batch(mol_crops, return_atoms_bonds=False, return_confidence=False)
                if len(mol_ocr_results) == len(mol_crops):
                    for bbox_key, result in zip(mol_keys, mol_ocr_results):
                        smiles = result.get("smiles")
                        if smiles:
                            mol_results[bbox_key] = self._canon_smiles(smiles)
            except Exception as e:
                logger.error(f"Batch MolOCR failed: {str(e)[:150]}")

        txt_results: Dict = {}
        if txt_bbox_map and self.use_paddle_ocr:
            try:
                sorted_items = sorted(txt_bbox_map.items(), key=lambda x: x[1][1])
                txt_crops = [it[1][0] for it in sorted_items]
                txt_keys = [it[0] for it in sorted_items]
                txt_ocr_results = self._call_paddle_ocr(txt_crops)
                for bbox_key, text in zip(txt_keys, txt_ocr_results):
                    if text and text.upper() != "UNKNOWN":
                        txt_results[bbox_key] = text.strip()
            except Exception as e:
                logger.error(f"Batch OCR failed: {str(e)[:150]}")

        for img_idx, category, rxn_idx, item_idx, type_, bbox_key in position_to_bbox:
            try:
                if type_ == "mol":
                    smiles = mol_results.get(bbox_key)
                    if smiles:
                        if category == "reactant":
                            preds[img_idx][rxn_idx]["reactants"][item_idx]["smiles"] = smiles
                        elif category == "product":
                            preds[img_idx][rxn_idx]["products"][item_idx]["smiles"] = smiles
                        elif category == "condition":
                            preds[img_idx][rxn_idx]["conditions"][item_idx]["smiles"] = smiles
                elif type_ == "txt":
                    text = txt_results.get(bbox_key)
                    if text:
                        preds[img_idx][rxn_idx]["conditions"][item_idx]["text"] = [text]
            except (IndexError, KeyError) as e:
                logger.error(f"Fill failed [img{img_idx}]: {e}")

        if not self.keep_crops:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        total_time = time.time() - start_time
        logger.debug(f"RxnOCR done | {total_time:.1f}s | imgs={len(files)} | mol={len(mol_results)} | txt={len(txt_results)}")
        return [{"image": f, "result": r} for f, r in zip(files, preds)]

    def predict_dir(self, dir_path: str, batch_size: Optional[int] = None, recursive: bool = True,
                    patterns: Optional[List[str]] = None, molscribe: bool = False,
                    ocr_type: Optional[str] = None) -> List[Dict[str, Any]]:
        d = Path(dir_path)
        if not d.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")
        candidates: List[Path] = []
        if patterns:
            for pat in patterns:
                candidates.extend(d.rglob(pat) if recursive else d.glob(pat))
        else:
            it = d.rglob("*") if recursive else d.glob("*")
            for p in it:
                if self._is_image(p):
                    candidates.append(p)
        files = sorted({str(p) for p in candidates})
        if not files:
            return []
        return self.predict_images(files, batch_size=batch_size, molscribe=molscribe, ocr_type=ocr_type)


def flatten_reactions(raw_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flatten reaction results to {reactants, products, conditions} string dicts."""
    flattened = []
    for item in raw_results:
        for rxn in item.get("result", []):
            react_smiles = [r.get("smiles") for r in rxn.get("reactants", []) if r.get("smiles")]
            prod_smiles = [p.get("smiles") for p in rxn.get("products", []) if p.get("smiles")]
            cond_items = []
            for c in rxn.get("conditions", []):
                if c.get("smiles"):
                    cond_items.append(c["smiles"])
                txt = c.get("text")
                if isinstance(txt, list):
                    s = " ".join(str(t) for t in txt if t)
                    if s:
                        cond_items.append(s)
                elif isinstance(txt, str) and txt.strip():
                    cond_items.append(txt.strip())
            flattened.append({
                "reactants": ";".join(react_smiles),
                "products": ";".join(prod_smiles),
                "conditions": ";".join(cond_items),
            })
    return flattened
