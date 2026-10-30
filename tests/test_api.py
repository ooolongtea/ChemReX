"""Exercise the HTTP boundary with the real SQLite queue and fake model stages."""

import asyncio
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from batch.api import ApiConfig, create_app
from batch import config as batch_config


PDF = b"%PDF-1.4\nAPI test fixture\n%%EOF"


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = ApiConfig(self.root, poll_interval=0.01)
        self.app = create_app(self.config)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.store = self.app.state.store

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.tmp.cleanup()

    def upload(self, **kwargs):
        return self.client.post("/v1/pdf?wait=false", files={"file": ("patent.pdf", PDF)}, **kwargs)

    def complete(self, task_id, failed=False):
        task = self.store.claim_next("test")
        self.assertEqual(task["uuid"], task_id)
        root = Path(task["output_dir"])
        root.mkdir(parents=True)
        chunk = {"id": "example_1", "chunk_fingerprint": "current"}
        (root / "chunks.jsonl").write_text(json.dumps(chunk) + "\n")
        rows = [
            {**chunk, "status": "error", "error": "previous attempt"},
            {**chunk, "status": "needs_review", "quality": {"admission_eligible": False}},
            {"id": "retired", "chunk_fingerprint": "old", "status": "ok"},
        ]
        (root / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        (root / "summary.json").write_text(json.dumps({"delivery": {"status": "error"}}))
        if failed:
            self.store._execute("UPDATE pdf_task SET attempt_count=? WHERE uuid=?",
                                (batch_config.MAX_ATTEMPTS, task_id))
            self.store.mark_failed(task["id"], "extraction failed", run_id="test")
        else:
            self.store.mark_done(task["id"], 1, run_id="test")

    def test_upload_persists_and_returns_recoverable_task(self):
        response = self.upload(data={"language": "zh"})
        self.assertEqual(response.status_code, 202)
        body = response.json()
        task = self.store.get(body["task_id"])
        self.assertEqual(task["language"], "zh")
        self.assertEqual(Path(task["pdf_path"]).read_bytes(), PDF)
        self.assertEqual(response.headers["x-task-id"], body["task_id"])
        self.assertEqual(self.client.get(response.headers["location"]).status_code, 202)

    def test_sync_request_waits_and_returns_unqualified_results(self):
        stop = threading.Event()
        errors = []

        def worker():
            try:
                while not stop.wait(0.01):
                    tasks = self.store.list_tasks()
                    if tasks:
                        self.complete(tasks[0]["uuid"])
                        return
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            response = self.client.post("/v1/pdf", files={"file": ("patent.pdf", PDF)})
        finally:
            stop.set()
            thread.join(2)
        self.assertFalse(errors)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "DONE")
        self.assertEqual(len(body["result"]["records"]), 1)
        record = body["result"]["records"][0]
        self.assertEqual(record["status"], "needs_review")
        self.assertFalse(record["quality"]["admission_eligible"])
        self.assertEqual(body["result"]["summary"]["delivery"]["status"], "error")

    def test_concurrent_idempotency_preserves_one_task_and_one_upload(self):
        def submit(_):
            return self.upload(headers={"Idempotency-Key": "system-document-42"})
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(submit, range(4)))
        self.assertTrue(all(r.status_code == 202 for r in responses))
        self.assertEqual(len({r.json()["task_id"] for r in responses}), 1)
        self.assertEqual(self.store.counts()["TOTAL"], 1)
        self.assertEqual(len(list((self.root / "uploads").glob("*/*.pdf"))), 1)

    def test_idempotency_conflict_checks_actual_bytes_and_language(self):
        self.upload(headers={"Idempotency-Key": "same"})
        changed = self.client.post("/v1/pdf?wait=false", headers={"Idempotency-Key": "same"},
                                  files={"file": ("patent.pdf", PDF + b"changed")})
        self.assertEqual(changed.status_code, 409)
        self.assertEqual(self.upload(headers={"Idempotency-Key": "same"},
                                     data={"language": "zh"}).status_code, 409)
        self.assertEqual(len(list((self.root / "uploads").glob("*/*.pdf"))), 1)

    def test_concurrent_distinct_uploads_are_not_dropped(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda _: self.upload(), range(8)))
        self.assertTrue(all(r.status_code == 202 for r in responses))
        self.assertEqual(self.store.counts()["PENDING"], 8)
        claimed = [self.store.claim_next("test")["uuid"] for _ in range(8)]
        self.assertEqual(len(set(claimed)), 8)

    def test_retry_backoff_is_not_a_terminal_failure(self):
        task_id = self.upload().json()["task_id"]
        task = self.store.claim_next("test")
        self.store.mark_failed(task["id"], "transient", run_id="test", retry_pending=True)
        result = self.client.get(f"/v1/tasks/{task_id}/result")
        self.assertEqual(result.status_code, 202)
        self.assertTrue(result.json()["retry_pending"])

    def test_exhausted_failure_returns_partial_results(self):
        task_id = self.upload().json()["task_id"]
        self.complete(task_id, failed=True)
        response = self.client.get(f"/v1/tasks/{task_id}/result")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "FAILED")
        self.assertEqual(len(response.json()["result"]["records"]), 1)

    def source_artifacts(self, task_id):
        root = Path(self.store.get(task_id)["output_dir"])
        prepared = root / "pdf_prepared" / "专利 #1" / "vlm"
        prepared.mkdir(parents=True)
        image = prepared / "images" / "反应 #1%.jpg"
        image.parent.mkdir()
        image.write_bytes(b"test image")
        content = prepared / "content list.json"
        blocks = [{"type": "text", "page_idx": 0, "bbox": [1, 2, 3, 4]},
                  {"type": "image", "page_idx": 0, "bbox": [5, 6, 7, 8],
                   "img_path": "images/反应 #1%.jpg"},
                  {"type": "text", "page_idx": 1, "bbox": [9, 10, 11, 12]}]
        content.write_text(json.dumps(blocks))
        chunk = {"id": "example_1", "chunk_fingerprint": "current", "source": str(content),
                 "images": ["images/反应 #1%.jpg"], "block_indices": [0, 1]}
        (root / "chunks.jsonl").write_text(json.dumps(chunk) + "\n")
        (root / "pdf_prepared" / "parse_result.json").write_text(json.dumps({"content_list_file": str(content)}))
        record = {"id": "example_1", "chunk_fingerprint": "current", "status": "ok",
                  "reactions": [{"index": 1, "step_source": {"source_parts": [
                      {"block_idx": 2, "page_idx": 1, "bbox": [9, 10, 11, 12]}]}}]}
        (root / "results.jsonl").write_text(json.dumps(record) + "\n")
        delivery = root / "delivery"
        delivery.mkdir()
        rows = [{"reaction_id": f"doc.pdf::example_1::{index}",
                 "source": {"chunk_id": "example_1", "reaction_index": index}} for index in [1, 2]]
        original = "".join(json.dumps(r) + "\n" for r in rows).encode()
        (delivery / "reactions.jsonl").write_bytes(original)
        (delivery / "routes.jsonl").write_bytes(b"")
        return root, original

    def test_urls_resolve_and_sources_preserve_reaction_and_chunk_scope(self):
        task_id = self.upload().json()["task_id"]
        self.complete(task_id)
        root, original = self.source_artifacts(task_id)
        body = self.client.get(f"/v1/tasks/{task_id}/result").json()
        result = body["result"]
        self.assertTrue(body["result_url"].startswith("http://testserver/"))
        self.assertIsNone(result["downloads"]["pdf"])
        for name in ["reactions", "routes", "chunks", "content_list", "records"]:
            url = result["downloads"][name]
            self.assertTrue(url.startswith("http://testserver/"))
            self.assertEqual(self.client.get(url).status_code, 200)
        sources = result["reaction_sources"]
        self.assertEqual(sources[0]["position_scope"], "reaction")
        self.assertEqual(sources[0]["blocks"][0]["block_idx"], 2)
        self.assertEqual(sources[1]["position_scope"], "chunk")
        self.assertEqual([b["block_idx"] for b in sources[1]["blocks"]], [0, 1])
        image = sources[0]["images"][0]
        self.assertEqual(image["bbox"], [5, 6, 7, 8])
        self.assertIn("%23", image["url"])
        self.assertIn("%25", image["url"])
        self.assertEqual(self.client.get(image["url"]).content, b"test image")
        response = self.client.get(result["downloads"]["reactions"])
        self.assertEqual(response.headers["content-type"], "application/x-ndjson")
        self.assertEqual(response.content, original)
        self.assertEqual((root / "delivery/reactions.jsonl").read_bytes(), original)
        self.assertEqual(result["reactions"], [json.loads(l) for l in original.splitlines()])

    def test_failure_downloads_only_advertise_existing_safe_files(self):
        task_id = self.upload().json()["task_id"]
        self.complete(task_id, failed=True)
        root = Path(self.store.get(task_id)["output_dir"])
        (root / "delivery/artifacts").mkdir(parents=True)
        (root / "delivery/artifacts/source.pdf").symlink_to(self.root / "tasks.db")
        result = self.client.get(f"/v1/tasks/{task_id}/result").json()["result"]
        self.assertIsNone(result["downloads"]["reactions"])
        self.assertIsNone(result["downloads"]["routes"])
        self.assertIsNone(result["downloads"]["pdf"])
        self.assertEqual(result["reaction_sources"], [])
        self.assertEqual(self.client.get(result["downloads"]["summary"]).status_code, 200)
        self.assertEqual(self.client.get(f"/v1/tasks/{task_id}/files/delivery/routes.jsonl").status_code, 404)

    def test_broken_source_metadata_keeps_business_results_and_blocks_image_escape(self):
        task_id = self.upload().json()["task_id"]
        self.complete(task_id)
        root, _ = self.source_artifacts(task_id)
        content = root / "pdf_prepared/专利 #1/vlm/content list.json"
        content.write_text("not json")
        response = self.client.get(f"/v1/tasks/{task_id}/result")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["result"]["reactions"]), 2)
        self.assertTrue(response.json()["result"]["warnings"])
        image = content.parent / "images/反应 #1%.jpg"
        image.unlink()
        image.symlink_to(self.root / "tasks.db")
        content.write_text(json.dumps([{"img_path": "images/反应 #1%.jpg"}]))
        result = self.client.get(f"/v1/tasks/{task_id}/result").json()["result"]
        self.assertEqual(result["reaction_sources"][0]["images"], [])

    def test_swagger_describes_internal_results_and_both_poll_responses(self):
        schema = self.client.get("/openapi.json").json()
        models = schema["components"]["schemas"]
        for path, method in [("/v1/pdf", "post"), ("/v1/tasks/{task_id}/result", "get")]:
            responses = schema["paths"][path][method]["responses"]
            for code, name in [("200", "TaskResult"), ("202", "TaskStatus")]:
                self.assertEqual(responses[code]["content"]["application/json"]["schema"]["$ref"],
                                 f"#/components/schemas/{name}")
        for name in ["InternalRecord", "InternalChunk", "InternalImages", "InternalParse", "InternalManifest"]:
            self.assertIn("不稳定", models[name]["description"])
            self.assertTrue(models[name]["additionalProperties"])
            self.assertTrue(models[name]["properties"])
        self.assertIn("delivery", models["SummaryResult"]["properties"])
        self.assertIn("reaction_sources", models["ResultData"]["properties"])
        self.assertIn("application/x-ndjson", schema["paths"]["/v1/tasks/{task_id}/files/{relative}"]["get"]["responses"]["200"]["content"])

    def test_absolute_links_include_application_root_path(self):
        with TestClient(create_app(self.config), base_url="https://example.test", root_path="/chemrex") as client:
            response = client.post("/v1/pdf?wait=false", files={"file": ("patent.pdf", PDF)})
            task_id = response.json()["task_id"]
            self.assertEqual(response.json()["result_url"], f"https://example.test/chemrex/v1/tasks/{task_id}/result")

    def test_invalid_uploads_do_not_enter_queue_or_leave_files(self):
        for name, content, code in [("not.txt", PDF, 400), ("bad.pdf", b"not PDF", 400)]:
            response = self.client.post("/v1/pdf?wait=false", files={"file": (name, content)})
            self.assertEqual(response.status_code, code)
        self.config.max_upload_bytes = len(PDF) - 1
        self.assertEqual(self.upload().status_code, 413)
        self.assertEqual(self.store.counts()["TOTAL"], 0)
        self.assertEqual(list((self.root / "uploads").glob("*")), [])

    def test_filename_traversal_is_removed(self):
        response = self.client.post("/v1/pdf?wait=false", files={"file": ("../../patent.pdf", PDF)})
        task = self.store.get(response.json()["task_id"])
        self.assertTrue(Path(task["pdf_path"]).is_relative_to(self.root / "uploads"))
        self.assertEqual(Path(task["pdf_path"]).name, "patent.pdf")

    def test_artifact_traversal_and_symlink_escape_are_rejected(self):
        task_id = self.upload().json()["task_id"]
        self.assertEqual(self.client.get(f"/v1/tasks/{task_id}/files/summary.json").status_code, 409)
        self.complete(task_id)
        root = Path(self.store.get(task_id)["output_dir"])
        (root / "escape.json").symlink_to(self.root / "tasks.db")
        self.assertEqual(self.client.get(f"/v1/tasks/{task_id}/files/summary.json").status_code, 200)
        self.assertEqual(self.client.get(f"/v1/tasks/{task_id}/files/escape.json").status_code, 404)
        self.assertEqual(self.client.get(f"/v1/tasks/{task_id}/files/%2E%2E/%2E%2E/tasks.db").status_code, 404)

    def test_queue_survives_api_restart(self):
        task_id = self.upload().json()["task_id"]
        with TestClient(create_app(ApiConfig(self.root))) as restarted:
            response = restarted.get(f"/v1/tasks/{task_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "PENDING")

    def test_routes_do_not_require_authentication(self):
        for headers in ({}, {"Authorization": "Bearer obsolete-value"}):
            self.assertEqual(self.client.get("/health", headers=headers).status_code, 200)
            self.assertEqual(self.client.get("/docs", headers=headers).status_code, 200)
            self.assertEqual(self.client.get("/openapi.json", headers=headers).status_code, 200)
            self.assertEqual(self.client.get("/v1/tasks/missing", headers=headers).status_code, 404)
        self.assertEqual(self.client.post("/v1/pdf", content=b"not multipart").status_code, 422)

    def test_legacy_token_setting_does_not_enable_authentication(self):
        with mock.patch.dict("os.environ", {
                "CHEMREX_API_TOKEN": "obsolete-setting",
                "CHEMREX_API_DATA_ROOT": str(self.root / "legacy-env"),
        }):
            with TestClient(create_app()) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                response = client.post("/v1/pdf?wait=false", files={"file": ("patent.pdf", PDF)})
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                self.assertEqual(client.get(f"/v1/tasks/{task_id}").status_code, 200)
                self.assertEqual(client.get(f"/v1/tasks/{task_id}/result").status_code, 202)

    def test_request_body_limit_without_content_length(self):
        self.config.max_upload_bytes = 1
        # A streaming body omits Content-Length, so check received bytes too.
        response = self.client.post("/v1/pdf", headers={
            "Content-Type": "multipart/form-data; boundary=test"},
            content=iter([b"--test\r\n", b"x" * (1024 * 1024 + 1)]))
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.store.counts()["TOTAL"], 0)

    def test_readiness_checks_worker_and_parser(self):
        with mock.patch("batch.api.worker_running", return_value=False), mock.patch(
                "batch.mineru_client.MineruClient.health", return_value={"status": "ok"}):
            self.assertEqual(self.client.get("/ready").status_code, 503)
        with mock.patch("batch.api.worker_running", return_value=True), mock.patch(
                "batch.mineru_client.MineruClient.health", return_value={"status": "ok"}):
            self.assertEqual(self.client.get("/ready").status_code, 200)

    def test_worker_launcher_preserves_full_pipeline_and_uses_api_queue(self):
        import run_api
        with mock.patch("batch.api.ApiConfig.from_env", return_value=self.config), mock.patch(
                "batch.cli.main", return_value=0) as batch_main, mock.patch.dict("os.environ", {}), mock.patch.object(
                batch_config, "BATCH_LOG_DIR", self.root / "logs"):
            self.assertEqual(run_api.main(["worker"]), 0)
            args = batch_main.call_args.args[0]
            self.assertEqual(args[:4], ["--db", str(self.config.db_path), "run", "--daemon"])
            self.assertEqual(args[args.index("--task-workers") + 1], "1")
            self.assertNotIn("--no-image", args)
            self.assertNotIn("--no-delivery", args)
            self.assertNotIn("--skip-final-product", args)

    def test_disconnect_keeps_registered_job(self):
        task_id = self.upload().json()["task_id"]

        # Drive the ASGI app directly, disconnecting the result wait immediately.
        async def exercise():
            messages = []
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                     "method": "GET", "scheme": "http", "path": f"/v1/tasks/{task_id}/result",
                     "raw_path": f"/v1/tasks/{task_id}/result".encode(),
                     "query_string": b"wait=true", "headers": [],
                     "server": ("test", 80), "client": ("test", 1), "root_path": ""}

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                messages.append(message)

            await self.app(scope, receive, send)
            return messages

        messages = asyncio.run(exercise())
        self.assertEqual(messages[0]["status"], 202)
        self.assertEqual(self.store.get(task_id)["status"], "PENDING")

    def test_shutdown_during_registration_keeps_committed_pdf(self):
        import httpx
        entered = threading.Event()
        release = threading.Event()
        original_add = self.store.add_task

        def delayed_add(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release registration")
            return original_add(*args)

        async def exercise():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                         base_url="http://test") as client:
                request = asyncio.create_task(client.post("/v1/pdf?wait=false",
                    headers={"Idempotency-Key": "cancel-test"}, files={"file": ("patent.pdf", PDF)}))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                    request.cancel()
                    await asyncio.sleep(0)
                finally:
                    release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await request

        with mock.patch.object(self.store, "add_task", side_effect=delayed_add):
            asyncio.run(exercise())
        task = self.store.get("api_cancel-test")
        self.assertIsNotNone(task)
        self.assertEqual(Path(task["pdf_path"]).read_bytes(), PDF)


if __name__ == "__main__":
    unittest.main()
