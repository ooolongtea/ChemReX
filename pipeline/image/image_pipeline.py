"""Image pipeline — compound dictionary + reaction recognition (no PDF position linking).

Adapted from old pipeline/image_pipeline.py:
- removed content_list.json page_idx/bbox position linking
- uses local .rxn_ocr / .coref_ocr wrappers (self-contained, no RetroParser dependency)
- conda envs configurable via config.IMGPARSER_CONDA_ENV / config.PADDLE_CONDA_ENV
- output: {dict_result: {id→smiles}, image_result: [{image_file_name, reactions}]}
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from .rxn_ocr import RxnOCR, flatten_reactions
from .coref_ocr import DictGenerator


def _get_available_gpus(min_free_memory_mb: int = 10000) -> List[Tuple[int, int]]:
    import subprocess
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        all_gpus = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                gpu_id = int(parts[0].strip())
                free_mb = int(parts[1].strip())
                all_gpus.append((gpu_id, free_mb))
    except Exception as e:
        logger.warning(f"Cannot query GPU info: {e}")
        return []

    if cuda_visible:
        visible_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        logical_gpus = []
        for logical_idx, physical_id in enumerate(visible_ids):
            for gid, free_mb in all_gpus:
                if gid == physical_id and free_mb >= min_free_memory_mb:
                    logical_gpus.append((logical_idx, free_mb))
                    break
        logical_gpus.sort(key=lambda x: x[1], reverse=True)
        if logical_gpus:
            logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible} → logical GPU {logical_gpus[0][0]} (physical {visible_ids[logical_gpus[0][0]]})")
        return logical_gpus

    gpus = [(gid, free_mb) for gid, free_mb in all_gpus if free_mb >= min_free_memory_mb]
    gpus.sort(key=lambda x: x[1], reverse=True)
    return gpus


def _try_initialize_model(model_class, gpu_ids: List[int], **kwargs) -> Tuple[Any, Optional[torch.device]]:
    for gpu_id in gpu_ids:
        try:
            device = torch.device(f"cuda:{gpu_id}")
            model = model_class(device=device, **kwargs)
            return model, device
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"GPU {gpu_id} init failed for {model_class.__name__}: {str(e)[:100]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    try:
        device = torch.device("cpu")
        model = model_class(device=device, **kwargs)
        return model, device
    except Exception as e:
        logger.error(f"{model_class.__name__} init failed: {e}")
        return None, None


def _find_image_dir(base_path: str) -> Optional[str]:
    """Find an images directory under base_path."""
    candidates = [os.path.join(base_path, "images"), os.path.join(base_path, "auto", "images")]
    for path in candidates:
        if os.path.isdir(path):
            return path
    for root, dirs, _ in os.walk(base_path):
        if "images" in dirs:
            return os.path.join(root, "images")
    return None


class ImagePipeline:
    """Sequential image processing: dict generation → reaction recognition. No position linking."""

    def __init__(self):
        try:
            import config
            self.paddle_env = getattr(config, "PADDLE_CONDA_ENV", "paddle")
        except ImportError:
            self.paddle_env = os.getenv("PADDLE_CONDA_ENV", "paddle")

    def run(self, image_base_path: str, return_per_image: bool = False) -> Dict[str, Any]:
        """Run image mining on a directory of images.

        Args:
            image_base_path: directory containing images (or an images/ subdir).
            return_per_image: if True, include ``per_image_detections`` in the
                output (per-image molecule boxes, figure labels, links, dict).
        Returns:
            {"dict_result": {id→smiles}, "image_result": [{image_file_name, reactions}]}
            and optionally ``per_image_detections`` when *return_per_image* is True.
        """
        start_time = time.time()
        request_id = f"local_{int(time.time() * 1000)}"

        available_gpus = _get_available_gpus(min_free_memory_mb=10000)
        gpu_ids = [g[0] for g in available_gpus]
        paddle_gpu = gpu_ids[0] if gpu_ids else None

        img_dir = _find_image_dir(image_base_path)
        if not img_dir:
            img_dir = image_base_path if os.path.isdir(image_base_path) else None
        if not img_dir:
            logger.error(f"Image directory not found under: {image_base_path}")
            return {"dict_result": {}, "image_result": []}

        rxn_images: List[str] = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff")):
                    rxn_images.append(os.path.join(root, f))
        rxn_images = sorted(rxn_images)
        if not rxn_images:
            logger.warning(f"No images found in {img_dir}")
            return {"dict_result": {}, "image_result": []}

        logger.info(f"[{request_id}] Processing {len(rxn_images)} images...")

        # Step 1: compound dictionary (MolDetect + MolOCR + PaddleOCR)
        compound_dict: Dict[str, str] = {}
        per_image_detections: List[Dict[str, Any]] = []
        dict_generator = None
        try:
            if paddle_gpu is not None:
                os.environ["PADDLE_OCR_GPU"] = str(paddle_gpu)
            dict_generator, _ = _try_initialize_model(
                DictGenerator, gpu_ids, use_paddle_ocr=True, paddle_env=self.paddle_env,
            )
            if dict_generator is not None:
                dict_result = dict_generator.generate_batch_fused(
                    rxn_images, batch_size=None, keep_duplicates=False,
                    return_per_image=return_per_image,
                )
                compound_dict = dict_result.get("map", {})
                if return_per_image:
                    per_image_detections = dict_result.get("per_image", [])
        except Exception as e:
            logger.error(f"[{request_id}] Dict generation failed: {e}")
            compound_dict = {}
        finally:
            if dict_generator is not None:
                del dict_generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Step 2: reaction recognition (RxnScribe + MolOCR + PaddleOCR)
        rxn_batch: List[Dict[str, Any]] = []
        rxn_ocr = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if paddle_gpu is not None:
                os.environ["PADDLE_OCR_GPU"] = str(paddle_gpu)
            rxn_ocr, _ = _try_initialize_model(
                RxnOCR, gpu_ids, keep_crops=False, use_paddle_ocr=True, paddle_env=self.paddle_env,
            )
            if rxn_ocr is not None:
                rxn_batch = rxn_ocr.predict_images(rxn_images, batch_size=None, molscribe=True, ocr_type="paddle")
        except Exception as e:
            logger.error(f"[{request_id}] Reaction recognition failed: {e}")
            rxn_batch = []
        finally:
            if rxn_ocr is not None:
                del rxn_ocr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Flatten & format (NO position linking — no page_idx/bbox)
        image_result = []
        for item in rxn_batch:
            fname = os.path.basename(item.get("image", ""))
            reactions = flatten_reactions([item])
            if reactions:
                image_result.append({"image_file_name": fname, "reactions": reactions})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Done | {elapsed:.1f}s | dict={len(compound_dict)} | image_rxns={len(image_result)}")
        result = {"dict_result": compound_dict, "image_result": image_result}
        if return_per_image:
            result["per_image_detections"] = per_image_detections
        return result
