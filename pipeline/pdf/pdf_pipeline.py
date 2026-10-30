"""Extract page text and images from a PDF.

Two backends are available:

- MinerU (default): runs the ``mineru`` CLI with a VLM backend. It restores the
  reading order of two-column patent scans and crops reaction images, writing
  an ordered ``*_content_list.json`` plus an ``images/`` directory.
- Poppler: plain page text and embedded images via the Poppler tools. Column
  order can interleave on two-column scans; kept as a lightweight fallback.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

MINERU_BACKENDS = (
    "pipeline", "vlm-http-client", "hybrid-http-client",
    "vlm-auto-engine", "hybrid-auto-engine",
)
DEFAULT_MINERU_BACKEND = os.getenv("MINERU_BACKEND", "vlm-auto-engine")


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is required; install Poppler utilities (poppler-utils)")
    return path


def _run(args: List[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed: {result.stderr.strip()}")
    return result.stdout


def _paragraphs(text: str) -> List[str]:
    """Split only on blank lines; retain chemical spelling and line breaks."""
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def process_pdf(
    pdf_path: str,
    output_dir: str,
    language: str,
    render_pages: bool = False,
    first_page: int = 1,
    last_page: Optional[int] = None,
) -> Dict[str, Any]:
    """Write pages.json, segments.jsonl and extracted images under output_dir.

    ``segments.jsonl`` has the ``text``/``language`` fields accepted by M1.
    PDF layout cannot identify experimental paragraphs reliably, so callers
    should review the segments before sending them to an extraction model.
    """
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError(f"PDF file not found: {pdf}")
    if language not in ("en", "zh"):
        raise ValueError("language must be 'en' or 'zh'")
    info = _run([_tool("pdfinfo"), str(pdf)])
    match = re.search(r"^Pages:\s*(\d+)\s*$", info, re.MULTILINE)
    if not match:
        raise RuntimeError("Cannot determine PDF page count from pdfinfo")
    page_count = int(match.group(1))
    end = page_count if last_page is None else last_page
    if first_page < 1 or end < first_page or end > page_count:
        raise ValueError(f"page range must be within 1..{page_count}")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)
    # A second run with a narrower page range must not feed stale images to M2.
    for pattern in ("page_*_image-*", "page_*_render.png"):
        for previous in images.glob(pattern):
            if previous.is_file():
                previous.unlink()
    pages: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = []
    image_names: List[str] = []
    for page_number in range(first_page, end + 1):
        text = _run([
            _tool("pdftotext"), "-f", str(page_number), "-l", str(page_number),
            "-layout", str(pdf), "-",
        ]).replace("\f", "").strip()
        page_images: List[str] = []
        prefix = images / f"page_{page_number:04d}_image"
        _run([_tool("pdfimages"), "-f", str(page_number), "-l", str(page_number),
              "-png", str(pdf), str(prefix)])
        page_images.extend(p.name for p in sorted(images.glob(prefix.name + "-*")))
        if render_pages:
            render_prefix = images / f"page_{page_number:04d}_render"
            _run([_tool("pdftoppm"), "-f", str(page_number), "-l", str(page_number),
                  "-singlefile", "-scale-to", "2000", "-png", str(pdf), str(render_prefix)])
            page_images.append(render_prefix.name + ".png")
        image_names.extend(page_images)
        pages.append({"page_number": page_number, "text": text, "images": page_images})
        for part_number, part in enumerate(_paragraphs(text), start=1):
            segments.append({
                "source_pdf": pdf.name, "page_number": page_number,
                "segment_number": part_number, "language": language, "text": part,
            })

    (output / "pages.json").write_text(
        json.dumps({"source_pdf": str(pdf), "page_count": page_count, "pages": pages},
                   ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (output / "segments.jsonl").open("w", encoding="utf-8") as stream:
        for segment in segments:
            stream.write(json.dumps(segment, ensure_ascii=False) + "\n")
    return {
        "source_pdf": str(pdf), "page_count": page_count,
        "pages_processed": len(pages), "segments": len(segments),
        "images": len(image_names), "pages_file": str(output / "pages.json"),
        "segments_file": str(output / "segments.jsonl"), "image_base": str(images),
    }


_MINERU_COMMAND: Optional[List[str]] = None


def _mineru_command() -> List[str]:
    """Return the argv prefix that runs the MinerU CLI.

    ``MINERU_BIN`` overrides the lookup. Conda entry points can carry a
    shebang the current environment cannot execute, so a sibling interpreter
    is probed as a fallback before giving up.
    """
    global _MINERU_COMMAND
    if _MINERU_COMMAND is not None:
        return _MINERU_COMMAND
    configured = os.getenv("MINERU_BIN")
    script = configured or shutil.which("mineru")
    if not script:
        raise RuntimeError(
            "MinerU executable not found; install MinerU or set MINERU_BIN to the mineru CLI")
    script = str(Path(script).expanduser().resolve())
    candidates: List[List[str]] = [[script]]
    sibling = Path(script).with_name("python")
    if sibling.is_file():
        candidates.append([str(sibling), script])
    for prefix in candidates:
        try:
            probe = subprocess.run(prefix + ["--version"], capture_output=True, text=True)
        except OSError:
            continue
        if probe.returncode == 0:
            _MINERU_COMMAND = prefix
            return prefix
    raise RuntimeError(
        f"MinerU CLI at {script} is not executable; set MINERU_BIN to a working mineru command")


def _mineru_parse_dir(output_dir: Path, pdf_stem: str, backend: str,
                      parse_method: str = "auto") -> Path:
    """Mirror MinerU's output layout: <output>/<stem>/<backend dir>/."""
    if backend.startswith("pipeline"):
        return output_dir / pdf_stem / parse_method
    if backend.startswith("vlm"):
        return output_dir / pdf_stem / "vlm"
    if backend.startswith("hybrid"):
        return output_dir / pdf_stem / f"hybrid_{parse_method}"
    raise ValueError(f"unknown MinerU backend: {backend}")


def _content_list_segments(blocks: List[Dict[str, Any]], source_pdf: str,
                           language: str) -> List[Dict[str, Any]]:
    """Turn ordered content-list blocks into M1-compatible segments."""
    segments: List[Dict[str, Any]] = []
    for block in blocks:
        if block.get("type") not in ("text", "list"):
            continue
        value = block.get("text")
        if value is None and block.get("type") == "list":
            value = block.get("list_items", [])
        if isinstance(value, list):
            value = " ".join(str(part) for part in value)
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            continue
        segments.append({
            "source_pdf": source_pdf,
            "page_number": int(block.get("page_idx", 0)) + 1,
            "segment_number": len(segments) + 1,
            "language": language, "text": text,
        })
    return segments


def process_pdf_mineru(
    pdf_path: str,
    output_dir: str,
    language: str,
    backend: Optional[str] = None,
    server_url: Optional[str] = None,
    first_page: int = 1,
    last_page: Optional[int] = None,
) -> Dict[str, Any]:
    """Run MinerU over a PDF and write M1/M2-ready outputs under output_dir.

    MinerU writes ``<output_dir>/<pdf stem>/<backend dir>/`` with the ordered
    ``<stem>_content_list.json``, markdown and cropped ``images/``. This
    function additionally writes ``segments.jsonl`` at the top of output_dir
    so the extraction module can consume the ordered blocks directly.
    """
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError(f"PDF file not found: {pdf}")
    if language not in ("en", "zh"):
        raise ValueError("language must be 'en' or 'zh'")
    backend = backend or DEFAULT_MINERU_BACKEND
    if backend not in MINERU_BACKENDS:
        raise ValueError(f"backend must be one of {MINERU_BACKENDS}")
    server_url = server_url or os.getenv("MINERU_URL")
    if backend.endswith("http-client") and not server_url:
        raise ValueError(f"backend {backend} requires --mineru-url (or MINERU_URL)")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    args = ["-p", str(pdf), "-o", str(output), "-b", backend]
    if backend.endswith("http-client"):
        args += ["-u", server_url]
    if first_page != 1 or last_page is not None:
        # MinerU page ids are 0-based and inclusive.
        args += ["-s", str(first_page - 1)]
        if last_page is not None:
            args += ["-e", str(last_page - 1)]
    if backend.startswith(("pipeline", "hybrid")):
        args += ["-l", "ch" if language == "zh" else "en"]

    command = _mineru_command() + args
    result = subprocess.run(command, capture_output=True, text=True)
    log_file = output / "mineru.log"
    log_file.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"mineru failed (exit {result.returncode}); see {log_file}: "
            f"{(result.stderr or result.stdout).strip()[:500]}")

    parse_dir = _mineru_parse_dir(output, pdf.stem, backend)
    content_list_file = parse_dir / f"{pdf.stem}_content_list.json"
    if not content_list_file.is_file():
        raise RuntimeError(
            f"MinerU output not found at {content_list_file}; see {log_file}")
    blocks = json.loads(content_list_file.read_text(encoding="utf-8"))
    segments = _content_list_segments(blocks, pdf.name, language)
    segments_file = output / "segments.jsonl"
    with segments_file.open("w", encoding="utf-8") as stream:
        for segment in segments:
            stream.write(json.dumps(segment, ensure_ascii=False) + "\n")

    image_base = parse_dir / "images"
    md_file = parse_dir / f"{pdf.stem}.md"
    return {
        "source_pdf": str(pdf), "backend": "mineru", "mineru_backend": backend,
        "mineru_output_dir": str(parse_dir),
        "content_list_file": str(content_list_file),
        "md_file": str(md_file) if md_file.is_file() else None,
        "segments": len(segments), "segments_file": str(segments_file),
        "images": len(list(image_base.glob("*"))) if image_base.is_dir() else 0,
        "image_base": str(image_base), "log_file": str(log_file),
    }
