"""CLI for PDF text and image preprocessing."""

import argparse
import json
import os
from pathlib import Path

from .pdf_pipeline import DEFAULT_MINERU_BACKEND, MINERU_BACKENDS, process_pdf, process_pdf_mineru


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF page text and images for ChemReX")
    parser.add_argument("--pdf", required=True, help="Input PDF file")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted files")
    parser.add_argument("--language", required=True, choices=("en", "zh"))
    parser.add_argument("--backend", choices=("mineru", "poppler"),
                        default=os.getenv("PDF_BACKEND", "mineru"),
                        help="PDF processing backend (default: mineru; set PDF_BACKEND to override)")
    parser.add_argument("--mineru-backend", default=DEFAULT_MINERU_BACKEND, choices=MINERU_BACKENDS,
                        help="MinerU parse backend (default: vlm-auto-engine; set MINERU_BACKEND to override)")
    parser.add_argument("--mineru-url", help="Server URL for vlm/hybrid-http-client MinerU backends")
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--last-page", type=int)
    parser.add_argument("--render-pages", action="store_true",
                        help="Poppler only: also render pages as images for vector drawings")
    parser.add_argument("--run-image", action="store_true",
                        help="Run M2 on extracted images (requires imgparser environment and weights)")
    args = parser.parse_args()
    if args.backend == "mineru":
        result = process_pdf_mineru(args.pdf, args.output_dir, args.language,
                                    backend=args.mineru_backend, server_url=args.mineru_url,
                                    first_page=args.first_page, last_page=args.last_page)
    else:
        result = process_pdf(args.pdf, args.output_dir, args.language,
                             render_pages=args.render_pages,
                             first_page=args.first_page, last_page=args.last_page)
    if args.run_image:
        from pipeline.image.image_pipeline import ImagePipeline
        image_result = ImagePipeline().run(result["image_base"])
        image_file = Path(args.output_dir) / "image_result.json"
        image_file.write_text(json.dumps(image_result, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        result["image_result_file"] = str(image_file.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
