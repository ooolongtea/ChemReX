"""Batch image processing — DictGenerator per-image detections with incremental saving.

Processes all images through DictGenerator (MolDetect + MolOCR + PaddleOCR),
producing per-image molecule boxes, figure labels, label-mol links, and dict_result.
Supports resume (skips already-processed images).

Usage:
    CUDA_VISIBLE_DEVICES=6 python -m pipeline.image.run_image_batch \
        --image-base data/raw/reference_sheets \
        --output-dir data/outputs/m2

Outputs:
    <output-dir>/per_image_detections.jsonl  (one line per image)
    <output-dir>/m2_image_result.json        (merged result, written at end)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from loguru import logger

from .coref_ocr import DictGenerator
from .image_pipeline import _get_available_gpus, _try_initialize_model


def _collect_images(image_base: str) -> List[str]:
    imgs: List[str] = []
    for root, _, files in os.walk(image_base):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff")):
                imgs.append(os.path.join(root, f))
    return sorted(imgs)


def _load_done(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fname = rec.get("image_file_name")
                if fname:
                    done.add(fname)
            except json.JSONDecodeError:
                continue
    return done


def _append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-base", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Detection batch size (default: auto based on image count)")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip already-processed images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "per_image_detections.jsonl"
    merged_path = output_dir / "m2_image_result.json"

    all_images = _collect_images(args.image_base)
    if not all_images:
        logger.error(f"No images found in {args.image_base}")
        sys.exit(1)
    logger.info(f"Found {len(all_images)} images")

    done = set() if args.no_resume else _load_done(jsonl_path)
    if done:
        logger.info(f"Resume: {len(done)} images already processed, skipping")
    pending = [p for p in all_images if os.path.basename(p) not in done]
    logger.info(f"Pending: {len(pending)} images")

    if not pending:
        logger.info("Nothing to do — all images already processed")
    else:
        available_gpus = _get_available_gpus(min_free_memory_mb=10000)
        gpu_ids = [g[0] for g in available_gpus]
        if not gpu_ids:
            logger.warning("No GPU with enough free memory, trying CPU")
            gpu_ids = []

        paddle_gpu = gpu_ids[0] if gpu_ids else None
        if paddle_gpu is not None:
            os.environ["PADDLE_OCR_GPU"] = str(paddle_gpu)

        try:
            import config
            paddle_env = getattr(config, "PADDLE_CONDA_ENV", "paddle")
        except ImportError:
            paddle_env = os.getenv("PADDLE_CONDA_ENV", "paddle")

        logger.info(f"Initializing DictGenerator (gpus={gpu_ids}, paddle_env={paddle_env})...")
        dict_generator, device = _try_initialize_model(
            DictGenerator, gpu_ids, use_paddle_ocr=True, paddle_env=paddle_env,
        )
        if dict_generator is None:
            logger.error("Failed to initialize DictGenerator")
            sys.exit(1)

        t0 = time.time()
        try:
            result = dict_generator.generate_batch_fused(
                pending,
                batch_size=args.batch_size,
                keep_duplicates=False,
                return_per_image=True,
            )
        except Exception as e:
            logger.error(f"DictGenerator failed: {e}")
            sys.exit(1)
        finally:
            del dict_generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        elapsed = time.time() - t0
        per_image = result.get("per_image", [])
        logger.info(f"DictGen done | {elapsed:.1f}s | imgs={len(pending)} | per_image={len(per_image)}")

        per_image_by_name = {r["image_file_name"]: r for r in per_image}
        records = []
        for img_path in pending:
            name = os.path.basename(img_path)
            if name in per_image_by_name:
                records.append(per_image_by_name[name])
            else:
                records.append({
                    "image_file_name": name,
                    "image_size": {},
                    "molecule_boxes": [],
                    "figure_labels": [],
                    "label_mol_links": [],
                    "dict_result": {},
                })
        _append_jsonl(jsonl_path, records)
        logger.info(f"Saved {len(records)} per-image records -> {jsonl_path}")

    merged = _merge_results(jsonl_path)
    with merged_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logger.info(f"Merged result saved -> {merged_path}")
    logger.info(f"  dict_result: {len(merged.get('dict_result', {}))} entries")
    logger.info(f"  per_image_detections: {len(merged.get('per_image_detections', []))} images")


def _merge_results(jsonl_path: Path) -> Dict[str, Any]:
    per_image: List[Dict[str, Any]] = []
    global_dict: Dict[str, str] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            per_image.append(rec)
            for k, v in (rec.get("dict_result") or {}).items():
                if k not in global_dict:
                    global_dict[k] = v

    return {
        "dict_result": global_dict,
        "image_result": [],
        "per_image_detections": per_image,
    }


if __name__ == "__main__":
    main()
