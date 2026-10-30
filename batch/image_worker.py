"""Resident image-mining worker — run inside the ``imgparser`` conda env.

Started and supervised by the batch scheduler (batch/image_client.py). Loads
the OCR/OCSR models once (``ImagePipeline(resident_models=True)``) and then
serves any number of documents over a line protocol:

    stdin : one JSON request per line
            {"request_id": "...", "image_base": "<dir>", "return_per_image": true,
             "result_path": "<absolute json path>"}
    stdout: one JSON response per line (in request order)
            {"request_id": "...", "status": "ok", "result_file": "<path>"}
            {"request_id": "...", "status": "ok", "result": {...}}  # no result_path
            {"request_id": "...", "status": "error", "error": "..."}
    stderr: loguru logs (inherited by the scheduler's log)

    The file-backed response keeps multi-megabyte per-image JSON off stdout:
    a full pipe used to block the worker while the scheduler reader was busy.

Exit on stdin EOF. A blank line is ignored. The scheduler serialises
requests, so the worker processes exactly one document at a time.
"""

import json
import os
import sys
import time
import faulthandler
from pathlib import Path

# Set these before importing OCR/torch/numpy. Resident OpenMP workers otherwise
# spin between documents even while the protocol thread is blocked on stdin.
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('KMP_BLOCKTIME', '0')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ChemReX root

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="INFO")

_protocol_out = None


def _open_protocol_channel() -> None:
    """Keep fd 1 exclusively for the JSON protocol: anything the model
    libraries print to stdout (paddle/torch/...) is redirected to stderr, and
    responses are written to the saved original stdout."""
    global _protocol_out
    _protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w",
                              encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


def _send(response: dict) -> None:
    _protocol_out.write(json.dumps(response, ensure_ascii=True) + "\n")
    _protocol_out.flush()


def _write_result_file(path: Path, result: dict) -> None:
    """Atomically persist the image result so only its path crosses stdout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main() -> None:
    _open_protocol_channel()
    from pipeline.image.image_pipeline import ImagePipeline

    pipeline = ImagePipeline(resident_models=True)
    logger.info("image worker ready (resident models)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.error("malformed request line: {}", exc)
            continue
        request_id = request.get("request_id")
        started = time.time()
        logger.info('request {} started: {}', request_id, request.get('image_base'))
        faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
        try:
            result = pipeline.run(
                request["image_base"],
                return_per_image=bool(request.get("return_per_image", True)),
            )
            errors = result.get("errors") if isinstance(result, dict) else None
            if errors:
                # A partial failure must not be reported as a successful empty
                # result: surface it so the scheduler retries or fails the task.
                response = {"request_id": request_id, "status": "error",
                            "error": "; ".join(str(e) for e in errors),
                            "elapsed_seconds": round(time.time() - started, 1)}
                logger.error("request {} partial failure: {}", request_id, errors)
            else:
                result_path = request.get("result_path")
                if result_path:
                    _write_result_file(Path(result_path), result)
                    response = {"request_id": request_id, "status": "ok",
                                "result_file": str(result_path),
                                "elapsed_seconds": round(time.time() - started, 1)}
                else:
                    response = {"request_id": request_id, "status": "ok",
                                "result": result,
                                "elapsed_seconds": round(time.time() - started, 1)}
        except Exception as exc:
            response = {"request_id": request_id, "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": round(time.time() - started, 1)}
            logger.exception("request {} failed", request_id)
        finally:
            faulthandler.cancel_dump_traceback_later()
        logger.info('request {} finished: {} ({}s)', request_id, response['status'],
                    response['elapsed_seconds'])
        _send(response)
    logger.info("stdin closed; image worker exiting")
    pipeline.release()


if __name__ == "__main__":
    main()
