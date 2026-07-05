"""CLI for Module 2 — Image mining.

Usage:
    python -m pipeline.image.run_image --image-base path/to/images
    python -m pipeline.image.run_image --image-base path/to/images --output result.json
"""

import argparse
import json
import os

from loguru import logger

from .image_pipeline import ImagePipeline


def main():
    parser = argparse.ArgumentParser(description="ChemReX Module 2 — Image mining")
    parser.add_argument("--image-base", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--return-per-image", action="store_true",
                        help="Include per-image detections (molecule boxes, figure labels, links)")
    args = parser.parse_args()

    pipeline = ImagePipeline()
    result = pipeline.run(args.image_base, return_per_image=args.return_per_image)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
