"""Tests for batch/document_processor.py — segmentation + concurrent extraction."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

BLOCKS = [
    {"type": "text", "text": "Example 1: Compound A", "page_idx": 0},
    {"type": "text", "text": "A mixture was stirred at 25 C for 2 h.", "page_idx": 0},
    {"type": "image", "img_path": "images/a.jpg", "page_idx": 0},
    {"type": "text", "text": "Example 2: Compound B", "page_idx": 1},
    {"type": "text", "text": "The solution was heated to reflux.", "page_idx": 1},
]


class FakeModule:
    def extract(self, text, language):
        # Echo this chunk's product title; a hard-coded A for Example 2 would
        # now correctly trigger the terminal-product coverage guard.
        title = text.splitlines()[0] if text.startswith("Compound") else "Compound A"
        ner = {"REACTION_PRODUCT": [title],
               "STARTING_MATERIAL": ["Compound B"]}
        return {"ner": ner, "ner_records": [ner], "action": ["stir"],
                "raw": {"ner": '{"REACTION_PRODUCT": "Compound A"}',
                        "action": "stir"},
                "parse_status": {"ner": "ok", "action": "ok"}}

    def resolve(self, payload):
        return {"ner": payload["ner"], "standardized": True,
                "dict_result": payload.get("dict_result")}

    def validate_record(self, record):
        return {"rxn_validation": {"status": "passed", "diagnosis": "pass"}}


class FakePipeline:
    def __init__(self, enable_image=True):
        self.enable_image = enable_image

    def _get_extraction(self):
        return FakeModule()

    def _get_coref(self):
        return FakeModule()

    def _get_validation(self):
        return FakeModule()


def make_processor(**kwargs):
    from batch.document_processor import DocumentProcessor

    defaults = dict(language="en", chunk_workers=4, skip_final_product=True)
    defaults.update(kwargs)
    with mock.patch("pipeline.pipeline.ChemReXPipeline", FakePipeline):
        return DocumentProcessor(**defaults)


class ConfiguredFakeModule(FakeModule):
    """Fake M1 that exposes the provenance configuration used by resume."""

    def configuration(self, language):
        return {"model": "qwen3-lora-08", "preset": "qwen3-lora-08",
                "prompts": {"ner": {"version": "v1", "content": "ner"},
                            "action": {"version": "v1", "content": "action"}}}

    def extract(self, text, language):
        result = super().extract(text, language)
        result["extraction_configuration"] = self.configuration(language)
        return result


class ConfiguredFakePipeline(FakePipeline):
    def _get_extraction(self):
        return ConfiguredFakeModule()


class DocumentProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.content_list = self.root / "content_list.json"
        self.content_list.write_text(json.dumps(BLOCKS), encoding="utf-8")
        self.out = self.root / "doc_out"
        self.image_data = {
            "dict_result": {"1": "CCO"},
            "image_result": [],
            "per_image_detections": [
                {"image_file_name": "a.jpg", "dict_result": {"2": "CCC"}}],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepare_segments_and_writes_manifest(self):
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        self.assertEqual(len(prepared.chunks), 2)
        self.assertTrue(all(c["has_procedure"] for c in prepared.chunks))
        self.assertEqual(prepared.chunks[0]["images"], ["images/a.jpg"])
        self.assertTrue((self.out / "chunks.jsonl").is_file())
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["total_chunks"], 2)
        self.assertEqual(manifest["with_procedure"], 2)

    def test_no_reaction_document_has_explicit_result(self):
        self.content_list.write_text(json.dumps([
            {"type": "text", "text": "Patent certificate for compound A",
             "page_idx": 0},
        ]), encoding="utf-8")
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        summary = processor.extract(prepared, self.out, {})
        result = json.loads((self.out / "document_result.json").read_text())
        self.assertEqual(summary["with_procedure"], 0)
        self.assertEqual(result["status"], "no_reaction_found")
        self.assertEqual((self.out / "results.jsonl").stat().st_size, 0)

    def test_extract_writes_ok_records_with_merged_dict(self):
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        progress = []
        summary = processor.extract(prepared, self.out, self.image_data,
                                    progress_cb=lambda d, t: progress.append((d, t)))
        records = [json.loads(line)
                   for line in (self.out / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["status"] == "ok" for r in records))
        # example_1 has local image dict {"2": ...} merged over global {"1": ...}
        first = next(r for r in records if r["id"] == "example_1")
        scope = first["image"]["dict_scope"]
        self.assertEqual(scope["local_entries"], 1)
        self.assertEqual(scope["global_entries"], 1)
        self.assertEqual(scope["merged_entries"], 2)
        self.assertEqual(summary["result_status"], {"ok": 2})
        self.assertEqual(progress[-1], (2, 2))
        self.assertTrue((self.out / "summary.json").is_file())

    def test_extract_resume_skips_completed(self):
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        processor.extract(prepared, self.out, self.image_data)
        # second run: nothing pending, no duplicate records
        progress = []
        summary = processor.extract(prepared, self.out, self.image_data,
                                    progress_cb=lambda d, t: progress.append((d, t)))
        records = [json.loads(line)
                   for line in (self.out / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["result_status"], {"ok": 2})
        self.assertEqual(progress, [])

    def test_model_config_change_reprocesses_completed_chunks(self):
        from batch.document_processor import DocumentProcessor

        with mock.patch("pipeline.pipeline.ChemReXPipeline", ConfiguredFakePipeline):
            processor = DocumentProcessor(language="zh", chunk_workers=2,
                                          skip_final_product=True)
        prepared = processor.prepare(str(self.content_list), self.out)
        processor.extract(prepared, self.out, self.image_data)
        path = self.out / "results.jsonl"
        self.assertEqual(len(path.read_text().splitlines()), 2)

        # Same M1 configuration: resume skips both completed chunks.
        processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(len(path.read_text().splitlines()), 2)

        # Stored records were produced by a different model/prompt: both must
        # be re-extracted instead of silently reused.
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        old_configuration = {
            "model": "deepseek-v4.1-flash", "preset": "huawei-maas",
            "prompts": {"ner": {"version": "v2", "content": "ner"},
                        "action": {"version": "v2", "content": "action"}},
        }
        for row in rows:
            row["extraction"] = {"extraction_configuration": old_configuration}
            for reaction in row.get("reactions") or []:
                reaction["extraction"] = {
                    "extraction_configuration": old_configuration}
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(len(path.read_text().splitlines()), 4)

    def test_changed_text_with_same_id_is_reprocessed(self):
        processor = make_processor()
        first = processor.prepare(str(self.content_list), self.out)
        processor.extract(first, self.out, self.image_data)
        changed = [dict(block) for block in BLOCKS]
        changed[1]["text"] = "A mixture was stirred at 40 C for 4 h."
        self.content_list.write_text(json.dumps(changed), encoding="utf-8")
        second = processor.prepare(str(self.content_list), self.out)
        self.assertEqual(first.chunks[0]["id"], second.chunks[0]["id"])
        self.assertNotEqual(first.chunks[0]["chunk_fingerprint"],
                            second.chunks[0]["chunk_fingerprint"])
        summary = processor.extract(second, self.out, self.image_data)
        records = [json.loads(line) for line in
                   (self.out / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual(summary["result_status"], {"ok": 2})

    def test_legacy_ok_is_reprocessed_with_matching_fingerprint(self):
        processor = make_processor(language="zh")
        prepared = processor.prepare(str(self.content_list), self.out)
        old = {"id": prepared.chunks[0]["id"], "status": "ok",
               "chunk_fingerprint": prepared.chunks[0]["chunk_fingerprint"],
               "validation": {"rxn_validation": {"best_expression": "CCO>>CC=O"}}}
        path = self.out / "results.jsonl"
        path.write_text(json.dumps(old) + "\n")
        processor.extract(prepared, self.out, self.image_data)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0], old)
        self.assertTrue(all(record["reactions"] for record in records[1:]))
        processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(len(path.read_text().splitlines()), 3)

    def test_standalone_resume_reprocesses_legacy_completed_result(self):
        from pipeline.document.run_document import main
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        chunk = prepared.chunks[0]
        path = self.out / "results.jsonl"
        path.write_text(json.dumps({"id": chunk["id"], "status": "needs_review",
                                   "reactions": [], "chunk_fingerprint": chunk["chunk_fingerprint"]}) + "\n")
        with mock.patch("pipeline.pipeline.ChemReXPipeline", FakePipeline), mock.patch(
                "sys.argv", ["run_document", "--content-list", str(self.content_list),
                             "--output-dir", str(self.out), "--language", "zh",
                             "--skip-final-product", "--no-delivery"]):
            main()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["reactions"] for row in rows[1:]))

    def test_extract_ignores_records_without_id(self):
        # M5: a valid-JSON line missing "id" must not crash resume parsing
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        with (self.out / "results.jsonl").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"status": "ok"}) + "\n")
            stream.write(json.dumps({"id": "example_1", "status": "ok"}) + "\n")
        summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary["result_status"], {"ok": 2})

    def test_extract_ignores_stale_chunk_ids(self):
        # review round-2 H1: an error record for a chunk id that no longer
        # exists must neither skip work nor fail the summary.
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        with (self.out / "results.jsonl").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": "removed_example_9",
                                     "status": "error", "error": "old"}) + "\n")
        summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary["result_status"], {"ok": 2})

    def test_extract_completed_filter_keeps_progress_honest(self):
        # review round-2 minor: ok records of stale ids must not inflate the
        # resume progress counters.
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        with (self.out / "results.jsonl").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": "removed_example_9",
                                     "status": "ok"}) + "\n")
        progress = []
        processor.extract(prepared, self.out, self.image_data,
                          progress_cb=lambda done, total: progress.append((done, total)))
        # both current chunks ran; the stale ok record counted for nothing
        self.assertEqual(progress, [(1, 2), (2, 2)])

    def test_standardize_failure_fails_chunk(self):
        # review round-2 H4: coref returning standardize_status="failed"
        # must fail the chunk (error record → scheduler retry), not ok.
        from batch.document_processor import DocumentProcessor

        class FailedStandardizeCoref(FakeModule):
            def resolve(self, payload):
                return {"ner": payload["ner"],
                        "standardize_status": "failed"}

        class FailedStandardizePipeline(FakePipeline):
            def _get_coref(self):
                return FailedStandardizeCoref()

        with mock.patch("pipeline.pipeline.ChemReXPipeline", FailedStandardizePipeline):
            processor = DocumentProcessor(language="en", chunk_workers=2,
                                          skip_final_product=True)
        prepared = processor.prepare(str(self.content_list), self.out)
        summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary["result_status"], {"error": 2})
        records = [json.loads(line)
                   for line in (self.out / "results.jsonl").read_text().splitlines()]
        self.assertIn("standardization failed", records[0]["error"])

    def test_dict_conflicts_are_reported(self):
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        image_data = {
            "dict_result": {"1": "CCO", "2": "CCN"},
            "image_result": [],
            "per_image_detections": [
                {"image_file_name": "a.jpg", "dict_result": {"2": "CCC"}}],
        }
        processor.extract(prepared, self.out, image_data)
        records = [json.loads(line)
                   for line in (self.out / "results.jsonl").read_text().splitlines()]
        first = next(r for r in records if r["id"] == "example_1")
        self.assertEqual(first["image"]["dict_scope"]["id_conflicts"], 1)
        # local value wins over the global fallback
        self.assertEqual(first["coref"]["dict_result"]["2"], "CCC")

    def test_extract_retry_errors_reprocesses_failed_chunks(self):
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        with (self.out / "results.jsonl").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": "example_1", "status": "error",
                                     "chunk_fingerprint": prepared.chunks[0]["chunk_fingerprint"],
                                     "error": "x"}) + "\n")
        # without retry_errors: the attempted error record is skipped
        summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary["result_status"], {"error": 1, "ok": 1})
        # with retry_errors: example_1 is reprocessed
        summary = processor.extract(prepared, self.out, self.image_data,
                                    retry_errors=True)
        self.assertEqual(summary["result_status"], {"ok": 2})

    def test_extract_failure_recorded_per_chunk(self):
        from batch.document_processor import DocumentProcessor

        class ExplodingExtraction(FakeModule):
            def extract(self, text, language):
                raise RuntimeError("api down")

        class ExplodingPipeline(FakePipeline):
            def _get_extraction(self):
                return ExplodingExtraction()

        with mock.patch("pipeline.pipeline.ChemReXPipeline", ExplodingPipeline):
            processor = DocumentProcessor(language="en", chunk_workers=2,
                                          skip_final_product=True)
        prepared = processor.prepare(str(self.content_list), self.out)
        summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary["result_status"], {"error": 2})
        records = [json.loads(line)
                   for line in (self.out / "results.jsonl").read_text().splitlines()]
        self.assertIn("api down", records[0]["error"])

    def test_delivery_default_disabled_and_failure_does_not_repeat_extraction(self):
        processor = make_processor()
        prepared = processor.prepare(str(self.content_list), self.out)
        (self.out / "original.pdf").write_bytes(b"%PDF-1.4\nprocessor")
        summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary["delivery"]["status"], "ready")
        self.assertTrue((self.out / "delivery/reactions.jsonl").is_file())
        self.assertTrue((self.out / "delivery/routes.jsonl").is_file())
        before = (self.out / "results.jsonl").read_bytes()
        with mock.patch("pipeline.document.delivery.export_document", side_effect=OSError("disk full")):
            summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary["delivery"]["status"], "error")
        self.assertEqual(before, (self.out / "results.jsonl").read_bytes())
        disabled = make_processor(no_delivery=True)
        separate = self.root / "disabled"
        prepared = disabled.prepare(str(self.content_list), separate)
        summary = disabled.extract(prepared, separate, self.image_data)
        self.assertNotIn("delivery", summary)
        self.assertFalse((separate / "delivery").exists())

    def test_final_product_cached(self):
        import pipeline.document.target_extractor  # noqa: F401 (mock target)
        from batch.document_processor import DocumentProcessor

        class FakeTargetExtractor:
            def extract(self, head, language):
                return {"product": "Target X", "reference": "Example 1: Compound A"}

        processor = make_processor(skip_final_product=False)
        prepared = processor.prepare(str(self.content_list), self.out)
        with mock.patch("pipeline.document.target_extractor.TargetExtractor",
                        FakeTargetExtractor):
            summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary["final_product"]["product"], "Target X")
        self.assertTrue((self.out / "final_product.json").is_file())
        # second run reuses the cache without the extractor
        summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary["final_product"]["product"], "Target X")

    def test_final_product_runs_during_chunk_processing(self):
        chunk_started, target_started = threading.Event(), threading.Event()
        class FakeTargetExtractor:
            def extract(self, head, language):
                target_started.set()
                if not chunk_started.wait(3):
                    raise RuntimeError('chunk did not overlap')
                return {'product': 'Target X', 'reference': 'Example 1: Compound A'}
        processor = make_processor(skip_final_product=False, chunk_workers=1, no_delivery=True)
        original = processor._process_chunk
        def run_chunk(*args):
            chunk_started.set()
            if not target_started.wait(3):
                raise RuntimeError('target did not overlap')
            return original(*args)
        processor._process_chunk = run_chunk
        prepared = processor.prepare(str(self.content_list), self.out)
        with mock.patch('pipeline.document.target_extractor.TargetExtractor', FakeTargetExtractor):
            summary = processor.extract(prepared, self.out, self.image_data)
        self.assertEqual(summary['final_product']['product'], 'Target X')
        self.assertNotIn('error', summary['result_status'])

    def test_final_product_failure_is_not_cached_and_recovers(self):
        processor = make_processor(skip_final_product=False, no_delivery=True)
        prepared = processor.prepare(str(self.content_list), self.out)
        with mock.patch('pipeline.document.target_extractor.TargetExtractor') as cls:
            cls.return_value.extract.side_effect = [RuntimeError('temporary failure'),
                {'product': 'Target X', 'reference': 'Example 1: Compound A'}]
            first = processor.extract(prepared, self.out, self.image_data)
            self.assertIn('temporary failure', first['final_product']['error'])
            self.assertFalse((self.out / 'final_product.json').exists())
            second = processor.extract(prepared, self.out, self.image_data)
            self.assertEqual(second['final_product']['product'], 'Target X')
            self.assertEqual(cls.return_value.extract.call_count, 2)


if __name__ == "__main__":
    unittest.main()
