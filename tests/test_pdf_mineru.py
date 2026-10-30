import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.pdf import pdf_pipeline


def fake_mineru_dir(root: Path, stem: str, backend_dir: str = "vlm") -> Path:
    parse_dir = root / stem / backend_dir
    (parse_dir / "images").mkdir(parents=True)
    (parse_dir / "images" / "a.jpg").write_bytes(b"img")
    blocks = [
        {"type": "text", "text": "Example 1: 2-(anilino) pyrimidine", "page_idx": 39},
        {"type": "list", "list_items": ["CH2Cl2, ", "CH3OH"], "page_idx": 39},
        {"type": "image", "img_path": "images/a.jpg", "page_idx": 39},
        {"type": "page_number", "text": "40", "page_idx": 39},
    ]
    (parse_dir / f"{stem}_content_list.json").write_text(
        json.dumps(blocks, ensure_ascii=False), encoding="utf-8")
    return parse_dir


class MineruCommandTest(unittest.TestCase):
    def setUp(self):
        pdf_pipeline._MINERU_COMMAND = None

    def tearDown(self):
        pdf_pipeline._MINERU_COMMAND = None
    def test_resolves_mineru_bin_env(self):
        with mock.patch.dict(os.environ, {"MINERU_BIN": "/opt/mineru/bin/mineru"}):
            with mock.patch.object(subprocess, "run") as run:
                run.return_value = mock.Mock(returncode=0)
                self.assertEqual(pdf_pipeline._mineru_command(), ["/opt/mineru/bin/mineru"])
                run.assert_called_once_with(
                    ["/opt/mineru/bin/mineru", "--version"],
                    capture_output=True, text=True)

    def test_missing_mineru_raises(self):
        with mock.patch.dict(os.environ, {"MINERU_BIN": ""}):
            with mock.patch.object(pdf_pipeline.shutil, "which", return_value=None):
                with self.assertRaises(RuntimeError):
                    pdf_pipeline._mineru_command()

    def test_falls_back_to_sibling_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "mineru"
            script.write_text("#!/usr/bin/env python3.10\n")
            (Path(tmp) / "python").write_text("")
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return mock.Mock(returncode=0 if len(calls) > 1 else 126)

            with mock.patch.dict(os.environ, {"MINERU_BIN": str(script)}):
                with mock.patch.object(subprocess, "run", side_effect=fake_run):
                    prefix = pdf_pipeline._mineru_command()
            self.assertEqual(prefix, [str(Path(tmp) / "python"), str(script)])


class ProcessPdfMineruTest(unittest.TestCase):
    def _run(self, tmp, **kwargs):
        pdf = Path(tmp) / "patent.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        output = Path(tmp) / "out"
        defaults = dict(pdf_path=str(pdf), output_dir=str(output), language="en")
        return pdf, output, pdf_pipeline.process_pdf_mineru(**{**defaults, **kwargs})

    def test_vlm_default_and_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                fake_mineru_dir(Path(tmp) / "out", "patent")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(pdf_pipeline.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(pdf_pipeline, "_mineru_command", return_value=["mineru"]):
                    pdf, output, result = self._run(tmp)
            self.assertEqual(result["backend"], "mineru")
            self.assertEqual(result["mineru_backend"], "vlm-auto-engine")
            self.assertTrue(result["content_list_file"].endswith("/patent/vlm/patent_content_list.json"))
            self.assertEqual(result["images"], 1)
            segments = [json.loads(line) for line in
                        (output / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([s["text"] for s in segments],
                             ["Example 1: 2-(anilino) pyrimidine", "CH2Cl2, CH3OH"])
            self.assertEqual([s["page_number"] for s in segments], [40, 40])
            self.assertEqual([s["segment_number"] for s in segments], [1, 2])
            self.assertTrue((Path(tmp) / "out" / "mineru.log").is_file())

    def test_page_range_is_zero_based(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = {}

            def fake_run(command, **kwargs):
                seen["args"] = command
                fake_mineru_dir(Path(tmp) / "out", "patent")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(pdf_pipeline.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(pdf_pipeline, "_mineru_command", return_value=["mineru"]):
                    self._run(tmp, first_page=2, last_page=5)
            args = seen["args"]
            self.assertEqual(args[args.index("-s") + 1], "1")
            self.assertEqual(args[args.index("-e") + 1], "4")
            self.assertNotIn("-l", args)

    def test_pipeline_backend_gets_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = {}

            def fake_run(command, **kwargs):
                seen["args"] = command
                fake_mineru_dir(Path(tmp) / "out", "patent", backend_dir="auto")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(pdf_pipeline.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(pdf_pipeline, "_mineru_command", return_value=["mineru"]):
                    _, _, result = self._run(tmp, backend="pipeline", language="zh")
            args = seen["args"]
            self.assertEqual(args[args.index("-l") + 1], "ch")
            self.assertTrue(result["content_list_file"].endswith("/patent/auto/patent_content_list.json"))

    def test_http_client_requires_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"MINERU_URL": ""}):
                with self.assertRaises(ValueError):
                    self._run(tmp, backend="vlm-http-client")

    def test_http_client_passes_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = {}

            def fake_run(command, **kwargs):
                seen["args"] = command
                fake_mineru_dir(Path(tmp) / "out", "patent")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(pdf_pipeline.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(pdf_pipeline, "_mineru_command", return_value=["mineru"]):
                    self._run(tmp, backend="vlm-http-client", server_url="http://127.0.0.1:30000")
            args = seen["args"]
            self.assertEqual(args[args.index("-u") + 1], "http://127.0.0.1:30000")

    def test_mineru_failure_raises_with_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                return mock.Mock(returncode=1, stdout="", stderr="boom")

            with mock.patch.object(pdf_pipeline.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(pdf_pipeline, "_mineru_command", return_value=["mineru"]):
                    with self.assertRaises(RuntimeError) as ctx:
                        self._run(tmp)
            self.assertIn("boom", str(ctx.exception))

    def test_missing_content_list_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(pdf_pipeline.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(pdf_pipeline, "_mineru_command", return_value=["mineru"]):
                    with self.assertRaises(RuntimeError):
                        self._run(tmp)


if __name__ == "__main__":
    unittest.main()
