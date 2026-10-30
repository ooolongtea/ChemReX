"""Resident MinerU parsing service — run inside the ``mineru`` conda env.

    conda run -n mineru python batch/mineru_service.py --port 9246 [--gpu 3]

Loads the layout/VLM models once on the first request (MinerU keeps them in a
process-level singleton) and parses any number of PDFs afterwards without
reloading — the key cost saving for large batches versus per-PDF CLI calls.

Endpoints:
    GET  /health   — service status (models loaded, busy, parsed count)
    POST /parse    — {pdf_path, output_dir, backend, language, first_page, last_page}
                     → {parse_dir, content_list_file, md_file, image_base, images}

Output layout mirrors the MinerU CLI (and ChemReX's _mineru_parse_dir):
    <output_dir>/<pdf stem>/vlm/            (vlm-* backends)
    <output_dir>/<pdf stem>/<parse_method>/ (pipeline backend)
"""

import argparse
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s mineru-service %(levelname)s %(message)s")
log = logging.getLogger("mineru-service")

from mineru.cli.common import do_parse, read_fn  # noqa: E402
from mineru.cli.output_paths import build_parse_dir  # noqa: E402

DEFAULT_BACKEND = os.getenv("MINERU_BACKEND", "vlm-auto-engine")
_mem_util = os.getenv("MINERU_VLM_MEM_UTIL")  # e.g. "0.15" for shared GPUs
DEFAULT_GPU_MEM_UTIL = float(_mem_util) if _mem_util else None


def select_best_gpu(min_free_mb: int = 10000) -> Optional[int]:
    """Pick the GPU with the most free memory; sets CUDA_VISIBLE_DEVICES."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
    except Exception as exc:
        log.warning("GPU query failed: %s", exc)
        return None
    best, best_free = None, -1
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        gpu_id, free_mb = int(parts[0]), int(parts[1])
        if free_mb > best_free:
            best, best_free = gpu_id, free_mb
    if best is not None and best_free >= min_free_mb:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
        log.info("Using GPU %d (%d MiB free)", best, best_free)
        return best
    log.warning("No GPU with >=%d MiB free; continuing on CPU/default", min_free_mb)
    return None


class ParseRequest(BaseModel):
    pdf_path: str
    output_dir: str
    backend: str = DEFAULT_BACKEND
    language: str = "en"          # "en" | "zh" (mapped to mineru "ch")
    parse_method: str = "auto"    # auto | txt | ocr (pipeline backend only)
    first_page: int = 1
    last_page: Optional[int] = None
    server_url: Optional[str] = None       # for vlm-http-client backends
    gpu_memory_utilization: Optional[float] = DEFAULT_GPU_MEM_UTIL


class ServiceState:
    def __init__(self):
        self.lock = threading.Lock()   # GPU serialisation: one PDF at a time
        self.models_loaded = False
        self.parsed_count = 0
        self.failed_count = 0
        self.busy = False
        self.last_error: Optional[str] = None


state = ServiceState()
app = FastAPI(title="ChemReX batch MinerU service")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": state.models_loaded,
        "busy": state.busy,
        "parsed_count": state.parsed_count,
        "failed_count": state.failed_count,
        "backend_default": DEFAULT_BACKEND,
        "last_error": state.last_error,
    }


@app.post("/parse")
def parse(req: ParseRequest):
    pdf = Path(req.pdf_path).expanduser().resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        return {"status": "error", "error": f"PDF file not found: {pdf}"}
    output = Path(req.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with state.lock:
        state.busy = True
        started = time.time()
        try:
            lang = "ch" if req.language.lower().startswith("zh") else "en"
            kwargs = {}
            if req.server_url:
                kwargs["server_url"] = req.server_url
            if req.gpu_memory_utilization is not None:
                # vLLM engines pre-allocate a fraction of TOTAL GPU memory;
                # shared GPUs need this lowered to fit the free portion.
                kwargs["gpu_memory_utilization"] = req.gpu_memory_utilization
            do_parse(
                str(output),
                [pdf.stem],
                [read_fn(pdf)],
                [lang],
                backend=req.backend,
                parse_method=req.parse_method,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=True,
                start_page_id=req.first_page - 1,
                end_page_id=(req.last_page - 1) if req.last_page is not None else None,
                **kwargs,
            )
            state.models_loaded = True
            parse_dir = build_parse_dir(str(output), pdf.stem, req.backend, req.parse_method)
            content_list = parse_dir / f"{pdf.stem}_content_list.json"
            if not content_list.is_file():
                raise RuntimeError(f"MinerU produced no content_list at {content_list}")
            image_base = parse_dir / "images"
            state.parsed_count += 1
            return {
                "status": "ok",
                "parse_dir": str(parse_dir),
                "content_list_file": str(content_list),
                "md_file": str(parse_dir / f"{pdf.stem}.md"),
                "image_base": str(image_base) if image_base.is_dir() else None,
                "images": len(list(image_base.glob("*"))) if image_base.is_dir() else 0,
                "elapsed_seconds": round(time.time() - started, 1),
            }
        except Exception as exc:
            state.failed_count += 1
            state.last_error = f"{type(exc).__name__}: {exc}"
            log.exception("parse failed for %s", pdf.name)
            return {"status": "error", "error": state.last_error,
                    "elapsed_seconds": round(time.time() - started, 1)}
        finally:
            state.busy = False


def main():
    global DEFAULT_BACKEND
    parser = argparse.ArgumentParser(description="Resident MinerU parsing service")
    parser.add_argument("--host", default=os.getenv("MINERU_SERVICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MINERU_SERVICE_PORT", "9246")))
    parser.add_argument("--gpu", type=int, default=None,
                        help="Pin to a specific GPU id (default: auto-select freest)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help=f"Default parse backend (default: {DEFAULT_BACKEND})")
    args = parser.parse_args()

    allowed = {int(part) for part in os.getenv(
        "MINERU_ALLOWED_GPUS", "0,1,2,3,4").split(",") if part.strip().isdigit()}
    if args.gpu is not None and allowed and args.gpu not in allowed:
        parser.error(f"--gpu {args.gpu} is not in MINERU_ALLOWED_GPUS={sorted(allowed)}")

    DEFAULT_BACKEND = args.backend

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        log.info("Pinned to GPU %d", args.gpu)
    else:
        select_best_gpu()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
