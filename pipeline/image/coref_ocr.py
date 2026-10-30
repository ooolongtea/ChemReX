"""DictGenerator — MolDetect detection + external MolOCR + Paddle OCR → ID→SMILES dictionary.

Adapted from RetroParser/ImgAgent/coref_ocr.py:
- local import from .MolDetect.rxnscribe.interface (no sys.path hacks)
- removed TextAgent.GPTAPI and TextAgent_refactor.llm_ocr dependencies (paddle-only path)
- model path configurable via config.MOLDETECT_CKPT
- no hardcoded GPU id
"""

import os
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from loguru import logger

from .MolDetect.rxnscribe.interface import MolDetect

try:
    from rdkit import Chem
except ImportError:
    Chem = None

from .mol_ocr import MolOCR
from pipeline.image_batch_retry import predict_with_backoff
from pipeline.chemical_quality import normalize_image_label, valid_image_smiles


def _resolve_moldetect_ckpt() -> str:
    try:
        import config
        return getattr(config, "MOLDETECT_CKPT", "")
    except ImportError:
        return os.getenv("MOLDETECT_CKPT", "")


class CorefOCR:
    """Lightweight MolDetect wrapper."""

    def __init__(self, device: Optional[torch.device] = None, coref: bool = False, num_workers: int = 1):
        ckpt_path = _resolve_moldetect_ckpt()
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"MolDetect ckpt not found: {ckpt_path}\n"
                f"Set MOLDETECT_CKPT env var or config.MOLDETECT_CKPT."
            )
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.coref_default = coref
        self.model = MolDetect(ckpt_path, device=self.device, coref=self.coref_default)
        self._exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}

    def predict_image_file(self, image_path: str, *, molscribe: bool = False,
                           coref: Optional[bool] = None, ocr: bool = False) -> Any:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        use_coref = self.coref_default if coref is None else coref
        return self.model.predict_image_file(image_file=image_path, molscribe=molscribe, coref=use_coref, ocr=ocr)

    def predict_batch(self, image_paths: List[str], *, batch_size: Optional[int] = None,
                      molscribe: bool = False, coref: Optional[bool] = None, ocr: bool = False) -> List[Any]:
        if not image_paths:
            return []
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing images: {missing}")
        if batch_size is None:
            n = len(image_paths)
            batch_size = 16 if n <= 50 else (32 if n <= 100 else (48 if n <= 200 else 64))
        use_coref = self.coref_default if coref is None else coref
        return self.model.predict_image_files(image_files=image_paths, batch_size=batch_size,
                                              molscribe=molscribe, coref=use_coref, ocr=ocr)


class DictGenerator:
    """Generate identifier → SMILES dictionary from images (MolDetect + MolOCR + Paddle OCR)."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        idt_category_id: int = 3,
        mol_category_id: int = 1,
        keep_crops: bool = False,
        use_paddle_ocr: bool = True,
        paddle_env: str = "paddle",
        molocr_device: Optional[str] = None,
    ):
        self.coref_ocr = CorefOCR(device=device, coref=True)
        molocr_device_str = molocr_device or (str(device) if device else "cuda:0")
        self.mol_ocr = MolOCR(device=molocr_device_str)
        self.use_paddle_ocr = use_paddle_ocr
        self.paddle_env = paddle_env
        self.paddle_ocr_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paddle_ocr.py")
        if use_paddle_ocr and not os.path.exists(self.paddle_ocr_script):
            raise FileNotFoundError(f"Paddle OCR script not found: {self.paddle_ocr_script}")
        self.idt_category_id = idt_category_id
        self.mol_category_id = mol_category_id
        self.keep_crops = keep_crops

    def _bbox_norm_to_px(self, bbox, w: int, h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(w - 1, int(round(bbox[0] * w))))
        y1 = max(0, min(h - 1, int(round(bbox[1] * h))))
        x2 = max(0, min(w - 1, int(round(bbox[2] * w))))
        y2 = max(0, min(h - 1, int(round(bbox[3] * h))))
        if x2 <= x1: x2 = min(w - 1, x1 + 1)
        if y2 <= y1: y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    def _canon_smiles(self, smi: str, allow_markush: bool = False) -> str:
        if not smi or Chem is None:
            return smi
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol and valid_image_smiles(smi, allow_markush=allow_markush):
                return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass
        return ""

    def _call_paddle_ocr(self, image_paths: List[str]) -> List[str]:
        try:
            env = os.environ.copy()
            paddle_gpu = os.environ.get("PADDLE_OCR_GPU")
            if paddle_gpu and not env.get("CUDA_VISIBLE_DEVICES"):
                env["CUDA_VISIBLE_DEVICES"] = str(paddle_gpu)
            paddle_python = os.environ.get("PADDLE_PYTHON")
            cmd = ([paddle_python] if paddle_python else
                   ["conda", "run", "-n", self.paddle_env, "python"])
            cmd += [self.paddle_ocr_script, "--json"] + image_paths
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            return json.loads(result.stdout.strip())
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()[-600:]
            raise RuntimeError(f"Paddle OCR subprocess failed: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Paddle OCR failed: {exc}") from exc

    def generate_batch_fused(
        self,
        image_paths: List[str],
        batch_size: Optional[int] = None,
        keep_duplicates: bool = False,
        return_per_image: bool = False,
    ) -> Dict[str, Any]:
        """Batch-detect all images, then batch-recognize molecules + identifiers, fuse into one dict."""
        import random
        if not image_paths:
            return {"map": {}, "total_pairs": 0, "failed_count": 0, "source_count": 0}
        if batch_size is None:
            batch_size = 16
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        def clear_cache():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        detections, failed_images = predict_with_backoff(
            image_paths,
            lambda subset, size: self.coref_ocr.predict_batch(
                subset, batch_size=size, coref=True, molscribe=False, ocr=False),
            batch_size or 16, {"bboxes": [], "corefs": []}, clear_cache,
        )
        for failure in failed_images:
            logger.error("dictionary detection failed for {}: {}",
                         failure["image"], failure["error"])

        all_mol_crops: List[str] = []
        all_idt_crops: List[str] = []
        mol_to_result: List[Tuple[int, int]] = []
        idt_to_result: List[Tuple[int, int, int]] = []
        mol_bbox_records: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
        idt_bbox_records: List[Tuple[int, int, int, Tuple[int, int, int, int]]] = []
        image_meta: Dict[int, Tuple[str, int, int]] = {}
        tmp_dir = tempfile.mkdtemp(prefix="dict_gen_batch_")
        global_mol_id = 0
        global_idt_id = 0
        failed_count = 0

        for img_idx, (image_path, detection) in enumerate(zip(image_paths, detections)):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    failed_images.append({"image": image_path, "error": "unreadable image"})
                    continue
                h, w = img.shape[:2]
                image_meta[img_idx] = (os.path.basename(image_path), w, h)
                bboxes = detection.get("bboxes", [])
                corefs = detection.get("corefs", [])
                paired_molecules = set()
                for pair_idx, pair in enumerate(corefs):
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    i, j = pair
                    if i < 0 or i >= len(bboxes) or j < 0 or j >= len(bboxes):
                        continue
                    a, b = bboxes[i], bboxes[j]
                    idt_rec = mol_rec = None
                    local_idt_idx = local_mol_idx = -1
                    if a.get("category_id") == self.idt_category_id and b.get("category_id") == self.mol_category_id:
                        idt_rec, mol_rec = a, b
                        local_idt_idx, local_mol_idx = i, j
                    elif b.get("category_id") == self.idt_category_id and a.get("category_id") == self.mol_category_id:
                        idt_rec, mol_rec = b, a
                        local_idt_idx, local_mol_idx = j, i
                    else:
                        failed_count += 1
                        continue
                    current_mol_id = global_mol_id
                    current_idt_id = global_idt_id
                    mol_bbox = mol_rec.get("bbox")
                    if not isinstance(mol_bbox, (list, tuple)) or len(mol_bbox) != 4:
                        failed_count += 1
                        continue
                    x1, y1, x2, y2 = self._bbox_norm_to_px(mol_bbox, w, h)
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        failed_count += 1
                        continue
                    mol_bbox_px = (x1, y1, x2, y2)
                    mol_crop = img[y1:y2, x1:x2]
                    mol_crop_path = os.path.join(tmp_dir, f"img{img_idx:04d}_p{pair_idx:03d}_m{local_mol_idx:03d}.png")
                    mol_saved = cv2.imwrite(mol_crop_path, mol_crop)
                    idt_bbox = idt_rec.get("bbox")
                    if not isinstance(idt_bbox, (list, tuple)) or len(idt_bbox) != 4:
                        failed_count += 1
                        continue
                    x1, y1, x2, y2 = self._bbox_norm_to_px(idt_bbox, w, h)
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        failed_count += 1
                        continue
                    idt_bbox_px = (x1, y1, x2, y2)
                    idt_crop = img[y1:y2, x1:x2]
                    idt_crop_path = os.path.join(tmp_dir, f"img{img_idx:04d}_p{pair_idx:03d}_i{local_idt_idx:03d}.png")
                    idt_saved = cv2.imwrite(idt_crop_path, idt_crop)
                    if mol_saved and idt_saved:
                        all_mol_crops.append(mol_crop_path)
                        all_idt_crops.append(idt_crop_path)
                        mol_to_result.append((img_idx, current_mol_id))
                        idt_to_result.append((img_idx, current_idt_id, current_mol_id))
                        mol_bbox_records.append((img_idx, current_mol_id, mol_bbox_px))
                        idt_bbox_records.append((img_idx, current_idt_id, current_mol_id, idt_bbox_px))
                        image_meta[img_idx] = (os.path.basename(image_path), w, h)
                        global_mol_id += 1
                        global_idt_id += 1
                        paired_molecules.add(local_mol_idx)
                # A molecule need not have a printed identifier. Coreference
                # links define dictionary entries, not the existence of a
                # molecule detection (US8927710, Example 210).
                for local_idx, mol_rec in enumerate(bboxes):
                    if mol_rec.get('category_id') != self.mol_category_id or local_idx in paired_molecules:
                        continue
                    bbox = mol_rec.get('bbox')
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = self._bbox_norm_to_px(bbox, w, h)
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue
                    crop_path = os.path.join(tmp_dir, f'img{img_idx:04d}_unpaired_m{local_idx:03d}.png')
                    if cv2.imwrite(crop_path, img[y1:y2, x1:x2]):
                        all_mol_crops.append(crop_path)
                        mol_to_result.append((img_idx, global_mol_id))
                        mol_bbox_records.append((img_idx, global_mol_id, (x1, y1, x2, y2)))
                        global_mol_id += 1
            except Exception as e:
                logger.error(f"Image [{img_idx}] error: {str(e)[:150]}")
                continue

        mol_results: Dict[int, str] = {}
        # Markush fallback: a scheme figure's general structure fails the
        # strict gate but stays valid as an image-side identifier. Keep it
        # for molecule_boxes (the document stage rebuilds label→structure
        # from the geometry); the dictionaries stay concrete-only.
        mol_markush: Dict[int, str] = {}
        if all_mol_crops:
            try:
                mol_ocr_results = self.mol_ocr.predict_batch(all_mol_crops, return_atoms_bonds=False, return_confidence=False)
                for (img_idx, gmid), result in zip(mol_to_result, mol_ocr_results):
                    smiles = result.get("smiles")
                    if smiles:
                        canon = self._canon_smiles(smiles)
                        mol_results[gmid] = canon
                        if not canon:
                            fallback = self._canon_smiles(smiles, allow_markush=True)
                            if fallback:
                                mol_markush[gmid] = fallback
            except Exception as e:
                logger.error(f"Batch MolOCR failed: {str(e)[:150]}")

        idt_ocr_texts: List[str] = []
        if all_idt_crops and self.use_paddle_ocr:
            try:
                idt_ocr_texts = self._call_paddle_ocr(all_idt_crops)
            except Exception as exc:
                raise RuntimeError(f"identifier OCR failed: {exc}") from exc

        fused_map: Dict[str, Any] = {}
        total_pairs = 0
        for (img_idx, gidt_id, gmid), ocr_text in zip(idt_to_result, idt_ocr_texts):
            if not ocr_text or ocr_text.upper() == "UNKNOWN":
                failed_count += 1
                continue
            smiles = mol_results.get(gmid)
            if not smiles:
                failed_count += 1
                continue
            ocr_text = normalize_image_label(ocr_text)
            if not ocr_text:
                failed_count += 1
                continue
            total_pairs += 1
            if keep_duplicates:
                if ocr_text in fused_map:
                    if isinstance(fused_map[ocr_text], list):
                        fused_map[ocr_text].append(smiles)
                    else:
                        fused_map[ocr_text] = [fused_map[ocr_text], smiles]
                else:
                    fused_map[ocr_text] = [smiles]
            else:
                fused_map.setdefault(ocr_text, smiles)

        if not self.keep_crops:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        per_image: List[Dict[str, Any]] = []
        if return_per_image:
            mol_by_img: Dict[int, List[Tuple[int, Tuple[int, int, int, int], str]]] = {}
            for (mi, gmid, bbox) in mol_bbox_records:
                mol_by_img.setdefault(mi, []).append(
                    (gmid, bbox, mol_results.get(gmid) or mol_markush.get(gmid, "")))
            idt_by_img: Dict[int, List[Tuple[int, int, Tuple[int, int, int, int], str]]] = {}
            for (mi, gidt, gmid, bbox), ocr_text in zip(idt_bbox_records, idt_ocr_texts):
                idt_by_img.setdefault(mi, []).append((gidt, gmid, bbox, ocr_text or ""))
            for img_idx in range(len(image_paths)):
                fname = os.path.basename(image_paths[img_idx])
                meta = image_meta.get(img_idx)
                if meta:
                    w, h = meta[1], meta[2]
                else:
                    image = cv2.imread(image_paths[img_idx])
                    h, w = image.shape[:2] if image is not None else (0, 0)
                mols_data = mol_by_img.get(img_idx, [])
                idts_data = idt_by_img.get(img_idx, [])
                molecule_boxes = []
                mol_gmid_to_idx: Dict[int, int] = {}
                for gmid, bbox, smiles in mols_data:
                    mol_gmid_to_idx[gmid] = len(molecule_boxes)
                    molecule_boxes.append({"bbox_xyxy": list(bbox), "smiles": smiles})
                figure_labels = []
                label_mol_links = []
                dict_res: Dict[str, str] = {}
                for gidt, gmid, bbox, ocr_text in idts_data:
                    if not ocr_text or ocr_text.upper() == "UNKNOWN":
                        continue
                    ocr_text = normalize_image_label(ocr_text)
                    if not ocr_text:
                        continue
                    label_idx = len(figure_labels)
                    figure_labels.append({"bbox_xyxy": list(bbox), "text": ocr_text})
                    if gmid in mol_gmid_to_idx:
                        label_mol_links.append({"label_idx": label_idx, "mol_idx": mol_gmid_to_idx[gmid]})
                    smi = mol_results.get(gmid)
                    if smi:
                        dict_res[ocr_text] = smi
                per_image.append({
                    "image_file_name": fname,
                    "image_size": {"width": w, "height": h},
                    "molecule_boxes": molecule_boxes,
                    "figure_labels": figure_labels,
                    "label_mol_links": label_mol_links,
                    "dict_result": dict_res,
                })

        total_time = time.time() - start_time
        logger.debug(f"DictGen done | {total_time:.1f}s | imgs={len(image_paths)} | map={len(fused_map)} | failed={failed_count}")
        result = {
            "map": fused_map,
            "total_pairs": total_pairs,
            "failed_count": failed_count,
            "source_count": len(image_paths),
            "failed_images": failed_images,
        }
        if return_per_image:
            result["per_image"] = per_image
        return result
