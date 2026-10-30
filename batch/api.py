"""Upload API over the existing persistent batch queue; no model imports here."""

import asyncio
import filecmp
import fcntl
import json
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import Response
from starlette.concurrency import run_in_threadpool

from . import config as batch_config
from .store import TaskStore
from .api_models import RESULT_RESPONSES, TaskStatus


@dataclass
class ApiConfig:
    data_root: Path
    max_upload_bytes: int = 200 * 1024 * 1024
    poll_interval: float = 0.5

    def __post_init__(self):
        self.data_root = self.data_root.expanduser().resolve()

    @classmethod
    def from_env(cls):
        return cls(
            data_root=Path(os.getenv("CHEMREX_API_DATA_ROOT",
                "/data/patents/chemrex_batch/api")),
            max_upload_bytes=int(os.getenv("CHEMREX_API_MAX_UPLOAD_MB", "200")) * 1024 * 1024,
        )

    @property
    def db_path(self):
        return self.data_root / "tasks.db"


class UploadBoundary:
    """Bound request bytes before multipart parsing/spooling."""

    def __init__(self, app, config):
        self.app = app
        self.config = config

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if scope["method"] != "POST" or scope["path"] != "/v1/pdf":
            return await self.app(scope, receive, send)
        # Allow multipart headers/form fields in addition to the PDF itself.
        limit = self.config.max_upload_bytes + 1024 * 1024
        try:
            too_large = int(headers.get(b"content-length", b"0")) > limit
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        if too_large:
            return await JSONResponse({"detail": "Upload exceeds the request size limit"}, 413)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(413, "Upload exceeds the request size limit")
            return message

        return await self.app(scope, limited_receive, send)


def worker_running(db_path):
    """Observe the existing scheduler lock, without changing its ownership."""
    lock_path = db_path.with_suffix(".lock")
    if not lock_path.is_file():
        return False
    with lock_path.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def task_view(task, base_url=""):
    retry_pending = task['status'] == 'FAILED' and bool(task.get('retry_pending', False))
    return {
        "task_id": task["uuid"],
        "filename": Path(task["pdf_path"]).name,
        "language": task["language"],
        "requested_language": task.get('requested_language', task['language']),
        "language_source": task.get('language_source', 'explicit'),
        "status": task["status"],
        "stage": task["stage"],
        "retry_pending": retry_pending,
        "finished": task["status"] in ("DONE", "CANCELLED") or (task["status"] == "FAILED" and not retry_pending),
        "attempts": task["attempt_count"],
        "chunks_total": task["chunks_total"],
        "chunks_done": task["chunks_done"],
        "error": task["error"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "elapsed_seconds": task["elapsed_seconds"],
        "result_url": f"{base_url}/v1/tasks/{quote(task['uuid'], safe='')}/result",
    }


def cleanup_deleted(store, config, task_id=None):
    """Clean only API-owned paths after execution acknowledgment, independently of models.

    The transaction prevents another frontend from completing deletion and
    reusing this task's id/path while a previous cleanup is still running.
    """
    with store._lock:
        store._conn.execute('BEGIN IMMEDIATE')
        try:
            sql = "SELECT * FROM pdf_task WHERE status='DELETING' AND execution_active=0"
            params = ()
            if task_id is not None:
                sql += ' AND uuid=?'
                params = (task_id,)
            for row in store._conn.execute(sql, params).fetchall():
                try:
                    root = config.data_root
                    output = Path(row['output_dir']).absolute()
                    upload = Path(row['pdf_path']).absolute().parent
                    if (output != root / 'results' / row['uuid'] or output.is_symlink()
                            or (root / 'results').is_symlink()
                            or upload.parent != root / 'uploads' or upload.is_symlink()
                            or (root / 'uploads').is_symlink()):
                        raise ValueError('Refusing to delete paths outside task-owned API directories')
                    for folder in (output, upload):
                        if folder.exists():
                            shutil.rmtree(folder)
                    # Worker/model logs are shared and deliberately retained.
                    store._conn.execute("DELETE FROM pdf_task WHERE id=? AND status='DELETING' AND execution_active=0", (row['id'],))
                except (OSError, ValueError) as exc:
                    store._conn.execute('UPDATE pdf_task SET error=?, updated_at=? WHERE id=?',
                                        (f'Deletion failed: {exc}', time.time(), row['id']))
            store._conn.execute('COMMIT')
        except Exception:
            store._conn.execute('ROLLBACK')
            raise


def source_links(task, result, base_url):
    """Project existing provenance into client URLs without changing delivery files."""
    root = Path(task["output_dir"]).resolve()
    prefix = f"{base_url}/v1/tasks/{quote(task['uuid'], safe='')}/files/"

    def local_path(value, parent=root):
        if not value or not isinstance(value, str):
            return None
        path = (parent / value).resolve()
        return path if path.is_relative_to(root) and path.is_file() else None

    def file_url(path):
        return prefix + quote(path.relative_to(root).as_posix(), safe="/") if path else None

    parse = result["parse"] or {}
    content_path = local_path(parse.get("content_list_file"))
    names = {"reactions": "delivery/reactions.jsonl", "routes": "delivery/routes.jsonl",
             "pdf": "delivery/artifacts/source.pdf", "chunks": "chunks.jsonl",
             "document": "document_result.json", "summary": "summary.json",
             "records": "results.jsonl", "images": "image_result.json",
             "final_product": "final_product.json", "delivery_manifest": "delivery/artifacts/manifest.json"}
    downloads = {key: file_url(local_path(name)) for key, name in names.items()}
    downloads["content_list"] = file_url(content_path)
    chunks = {c["id"]: c for c in result["chunks"]}
    records = {r["id"]: r for r in result["records"]}
    content_cache = {}

    def content_blocks(path):
        if path not in content_cache:
            try:
                data = json.loads(path.read_text(encoding="utf-8")) if path else []
                content_cache[path] = data if isinstance(data, list) else []
            except (OSError, ValueError) as exc:
                result["warnings"].append(f"source content list: {type(exc).__name__}: {exc}")
                content_cache[path] = []
        return content_cache[path]

    def block_info(block, index=None):
        return {"block_idx": block.get("block_idx", index), "page_idx": block.get("page_idx"),
                "bbox": block.get("bbox"), "type": block.get("type")}

    sources = []
    for reaction in result["reactions"]:
        source = reaction["source"]
        chunk_id = source["chunk_id"]
        chunk = chunks.get(chunk_id, {})
        record = records.get(chunk_id, {})
        path = local_path(chunk.get("source")) or content_path
        blocks = content_blocks(path)
        raw = next((r for r in record.get("reactions") or []
                    if r.get("index") == source["reaction_index"]), {})
        context = raw.get("step_source") or next(
            ((raw.get(layer) or {}).get("step_source") for layer in ("validation", "coref", "extraction")
             if (raw.get(layer) or {}).get("step_source")), {})
        positions = [block_info(b) for b in context.get("source_parts") or [] if isinstance(b, dict)]
        scope = "reaction" if positions else "unavailable"
        if not positions:
            positions = [block_info(blocks[i], i) for i in chunk.get("block_indices") or []
                         if type(i) is int and 0 <= i < len(blocks) and isinstance(blocks[i], dict)]
            if not positions:
                positions = [block_info(b) for b in (chunk.get("position") or {}).get("blocks") or []
                             if isinstance(b, dict)]
            if positions:
                scope = "chunk"
        images, seen = [], set()
        # Use exact paths, never basename matching across different images.
        image_paths = {local_path(name, path.parent) for name in chunk.get("images") or []} if path else set()
        image_paths.discard(None)
        for i, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            image_path = local_path(block.get("img_path"), path.parent)
            if image_path not in image_paths or image_path in seen:
                continue
            seen.add(image_path)
            images.append({**block_info(block, i), "url": file_url(image_path)})
        for image_path in sorted(image_paths - seen):
            images.append({"url": file_url(image_path), "block_idx": None, "page_idx": None,
                           "bbox": None, "type": None})
        sources.append({"reaction_id": reaction["reaction_id"], "chunk_id": chunk_id,
                        "content_list_url": file_url(path), "position_scope": scope,
                        "blocks": positions, "images": images})
    return downloads, sources


def collect_result(task, base_url=""):
    """Return existing structured outputs, including unqualified/partial records.

    Never read the append-only history as if every line were a current record.
    Delivery export errors are exposed alongside the raw extraction outputs.
    """
    from pipeline.document.results import latest_by_id

    root = Path(task["output_dir"])
    warnings = []

    def read_json(relative, default=None):
        path = root / relative
        if not path.is_file():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warnings.append(f"{relative}: {type(exc).__name__}: {exc}")
            return default

    def read_lines(relative):
        path = root / relative
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    warnings.append(f"{relative}:{number}: invalid JSON")
        return rows

    chunks = read_lines("chunks.jsonl")
    fingerprints = {c["id"]: c.get("chunk_fingerprint") for c in chunks}
    records = latest_by_id(root / "results.jsonl", fingerprints)
    result = {
        "document": read_json("document_result.json"),
        "summary": read_json("summary.json"),
        "manifest": read_json("manifest.json"),
        "parse": read_json("pdf_prepared/parse_result.json"),
        "images": read_json("image_result.json"),
        "final_product": read_json("final_product.json"),
        "chunks": chunks,
        "links": read_json("links.json", []),
        "records": [records[c["id"]] for c in chunks if c["id"] in records],
        "reactions": read_lines("delivery/reactions.jsonl"),
        "routes": read_lines("delivery/routes.jsonl"),
        "delivery_manifest": read_json("delivery/artifacts/manifest.json"),
        "warnings": warnings,
        "files_url": f"{base_url}/v1/tasks/{quote(task['uuid'], safe='')}/files/",
    }
    result["downloads"], result["reaction_sources"] = source_links(task, result, base_url)
    return {**task_view(task, base_url), "result": result}


def create_app(config: Optional[ApiConfig] = None):
    config = config or ApiConfig.from_env()
    config.data_root = config.data_root.expanduser().resolve()
    if config.max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = TaskStore(config.db_path)
        async def clean_loop():
            while True:
                cleanup = asyncio.create_task(run_in_threadpool(cleanup_deleted, app.state.store, config))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise
                await asyncio.sleep(config.poll_interval)
        cleaner = asyncio.create_task(clean_loop())
        try:
            yield
        finally:
            cleaner.cancel()
            # Do not close SQLite while a cancellation-shielded cleanup thread
            # still owns the transaction.
            try:
                await cleaner
            except asyncio.CancelledError:
                pass
            app.state.store.close()

    app = FastAPI(title="ChemReX PDF API", lifespan=lifespan, description=(
        "上传 PDF 后轮询 result_url，以 finished 判断终态。业务入库使用 reactions / routes；"
        "downloads 和 reaction_sources 提供现成 URL 与来源坐标。document / summary / delivery_manifest "
        "中已说明的状态、统计字段可用于展示。其余中间结果不稳定，不依赖其结构入库；"
        "展开响应 Schema 可查询当前内部字段。响应保留未列出的内部字段，不进行过滤。"))
    app.add_middleware(UploadBoundary, config=config)

    async def get_task(task_id):
        task = await run_in_threadpool(app.state.store.get, task_id)
        if task is None:
            raise HTTPException(404, "Unknown task")
        return task

    async def respond(task, request, wait):
        base_url = str(request.base_url).rstrip("/")
        while wait and not task_view(task)["finished"] and task['status'] != 'PAUSED':
            if await request.is_disconnected():
                # Disconnect only ends the HTTP wait; the durable job continues.
                return JSONResponse(task_view(task, base_url), status_code=202)
            await asyncio.sleep(config.poll_interval)
            task = await get_task(task["uuid"])
        headers = {"Location": f"/v1/tasks/{task['uuid']}/result",
                   "X-Task-ID": task["uuid"]}
        if task_view(task)["finished"] or task['status'] == 'PAUSED':
            # Large patent responses must not block other requests while JSON
            # is read and encoded.
            return await run_in_threadpool(lambda: JSONResponse(collect_result(task, base_url),
                status_code=202 if task['status'] == 'PAUSED' else 200, headers=headers))
        return JSONResponse(task_view(task, base_url), status_code=202,
                            headers={**headers, "Retry-After": "2"})

    @app.get("/health")
    async def health():
        return {"status": "ok", "queue": await run_in_threadpool(app.state.store.counts)}

    @app.get("/ready")
    async def ready():
        from .mineru_client import MineruClient
        worker = await run_in_threadpool(worker_running, config.db_path)
        try:
            parser = await run_in_threadpool(MineruClient(batch_config.MINERU_SERVICE_URL).health, timeout=2)
            parser_ready = parser.get("status") == "ok"
        except Exception:
            parser_ready = False
        healthy = worker and parser_ready
        return JSONResponse({"status": "ready" if healthy else "degraded",
                             "worker_running": worker, "parser_reachable": parser_ready},
                            status_code=200 if healthy else 503)

    @app.post("/v1/pdf", responses=RESULT_RESPONSES)
    async def upload_pdf(
        request: Request,
        file: UploadFile = File(...),
        language: Literal["auto", "en", "zh"] = Form("auto"),
        wait: bool = Query(True),
        idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    ):
        if idempotency_key is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", idempotency_key):
            raise HTTPException(400, "Idempotency-Key must contain 1-128 letters, digits, _ or -")
        filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if (not filename.lower().endswith(".pdf") or len(filename.encode("utf-8")) > 240
                or any(ord(c) < 32 for c in filename)):
            raise HTTPException(400, "A PDF filename is required")

        task_id = "api_" + (idempotency_key or uuid.uuid4().hex)
        upload_dir = config.data_root / "uploads" / uuid.uuid4().hex
        path = upload_dir / filename
        def persist():
            registered = False
            try:
                upload_dir.mkdir(parents=True)
                # Keep disk writes and comparisons out of the event loop.
                with path.open("wb") as target:
                    size = 0
                    prefix = b""
                    while block := file.file.read(1024 * 1024):
                        size += len(block)
                        if size > config.max_upload_bytes:
                            raise HTTPException(413, "PDF exceeds the upload size limit")
                        if len(prefix) < 1024:
                            prefix += block[:1024 - len(prefix)]
                        target.write(block)
                    if b"%PDF-" not in prefix:
                        raise HTTPException(400, "File does not contain a PDF header")
                    target.flush()
                    os.fsync(target.fileno())
                store = app.state.store
                # Registration and input comparison must exclude deletion:
                # a concurrent cleanup can otherwise remove the PDF/row
                # between add_task, get, and filecmp.
                with store._lock:
                    store._conn.execute('BEGIN IMMEDIATE')
                    try:
                        _, created = store.add_task(
                            task_id, str(path), str(config.data_root / "results" / task_id), language)
                        task = store.get(task_id)
                        if not created:
                            if task['status'] == 'DELETING':
                                raise HTTPException(409, 'Task deletion is still in progress')
                            same_input = (task["requested_language"] == language
                                          and Path(task["pdf_path"]).name == filename
                                          and filecmp.cmp(path, task["pdf_path"], shallow=False))
                            if not same_input:
                                raise HTTPException(409, "Idempotency-Key is already used for a different input")
                        store._conn.execute('COMMIT')
                        registered = created
                    except Exception:
                        store._conn.execute('ROLLBACK')
                        raise
                return task
            finally:
                if not registered:
                    shutil.rmtree(upload_dir, ignore_errors=True)

        # A server shutdown can cancel the coroutine while SQLite commits in a
        # thread. Finish registration before closing the upload; never delete a
        # file that a committed job now owns.
        registration = asyncio.create_task(run_in_threadpool(persist))
        try:
            task = await asyncio.shield(registration)
        except asyncio.CancelledError:
            await registration
            raise
        finally:
            await file.close()
        return await respond(task, request, wait)

    @app.get("/v1/tasks/{task_id}", responses={200: {"model": TaskStatus}})
    async def status(task_id: str, request: Request):
        return task_view(await get_task(task_id), str(request.base_url).rstrip("/"))

    @app.get("/v1/tasks/{task_id}/result", responses=RESULT_RESPONSES)
    async def result(task_id: str, request: Request, wait: bool = False):
        return await respond(await get_task(task_id), request, wait)

    async def control(task_id, request, action):
        try:
            task = await run_in_threadpool(app.state.store.request_stop, task_id, action)
        except KeyError:
            if action == 'delete':
                return Response(status_code=204)
            raise HTTPException(404, 'Unknown task')
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        if action == 'delete':
            await run_in_threadpool(cleanup_deleted, app.state.store, config, task_id)
            task = await run_in_threadpool(app.state.store.get, task_id)
            if task is None:
                return Response(status_code=204)
        code = 202 if task['status'] in ('PAUSING', 'CANCELLING', 'DELETING') else 200
        return JSONResponse(task_view(task, str(request.base_url).rstrip('/')), status_code=code)

    control_responses = {200: {'model': TaskStatus}, 202: {'model': TaskStatus},
                         404: {'description': 'Unknown task'},
                         409: {'description': 'Operation conflicts with the current task state'}}

    @app.post('/v1/tasks/{task_id}/pause', responses=control_responses)
    async def pause(task_id: str, request: Request):
        return await control(task_id, request, 'pause')

    @app.post('/v1/tasks/{task_id}/cancel', responses=control_responses)
    async def cancel(task_id: str, request: Request):
        return await control(task_id, request, 'cancel')

    @app.delete('/v1/tasks/{task_id}', status_code=204, response_class=Response,
                responses={202: {'model': TaskStatus}, 204: {'description': 'Task deleted'}})
    async def delete(task_id: str, request: Request):
        return await control(task_id, request, 'delete')

    @app.post('/v1/tasks/{task_id}/resume', responses=control_responses)
    # Legacy alias: new clients and OpenAPI use resume for every recovery.
    @app.post('/v1/tasks/{task_id}/retry', responses=control_responses, include_in_schema=False)
    async def resume(task_id: str, request: Request):
        try:
            task = await run_in_threadpool(app.state.store.resume, task_id)
        except KeyError:
            raise HTTPException(404, 'Unknown task')
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return task_view(task, str(request.base_url).rstrip('/'))

    @app.get("/v1/tasks/{task_id}/files/{relative:path}", response_class=FileResponse, responses={
        200: {"description": "已有产物的原始字节；失败终态及安全暂停后也可下载，不保证所有文件存在。",
              "content": {kind: {"schema": {"type": "string", "format": "binary"}}
                          for kind in ("application/x-ndjson", "application/json", "application/pdf", "image/jpeg", "image/png", "application/octet-stream")}},
        404: {"description": "任务或文件不存在，或文件路径超出任务目录。"},
        409: {"description": "任务仍在处理、停止中或等待自动重试；安全暂停后允许下载。"},
    })
    async def artifact(task_id: str, relative: str):
        task = await get_task(task_id)
        if not task_view(task)["finished"] and task['status'] != 'PAUSED':
            raise HTTPException(409, "Task is still processing")
        root = Path(task["output_dir"]).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, "Unknown result file")
        media_type = "application/x-ndjson" if path.suffix.lower() == ".jsonl" else None
        return FileResponse(path, media_type=media_type)

    return app
