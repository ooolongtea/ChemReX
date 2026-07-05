"""CLI for Module 1 — Text extraction (NER + Action).

Usage:
    # single text (默认 Qwen3-8B + LoRA 0.8，需本地 vLLM 挂载 qwen3-lora-08)
    python -m pipeline.extraction.run_extraction --text "..." --task both --language zh

    # JSONL batch (each line: {text|input, task_type?, language, output?, source?})
    python -m pipeline.extraction.run_extraction --input segs.jsonl --output preds.jsonl

    # 显式指定 LoRA 0.8 预设并并发 32（与 eval 主实验对齐）
    python -m pipeline.extraction.run_extraction \
        --preset qwen3-lora-08 --input segs.jsonl --output preds.jsonl --max-workers 32

    # 临时使用其它 OpenAI 兼容模型（未部署 vLLM 时）
    python -m pipeline.extraction.run_extraction --text "..." --task both --language zh --model deepseek-v3.2

    # 列出可用预设
    python -m pipeline.extraction.run_extraction --list-presets
"""

import argparse
import json
import os
from typing import Dict, Any, List, Optional

from loguru import logger

from .extractor import ChemReXExtractor


def _norm_language(lang: str) -> str:
    s = (lang or "").strip().lower()
    if s in ("zh", "zh-cn", "chinese", "cn"):
        return "zh"
    if s in ("en", "english"):
        return "en"
    raise ValueError(f"language must be 'zh' or 'en' (provided by caller, not detected); got {lang!r}")


def _infer_task_from_type(task_type: Any) -> str:
    if isinstance(task_type, int):
        return "ner" if task_type == 0 else "action"
    if isinstance(task_type, str):
        v = task_type.strip().lower()
        if v in ("0", "ner"):
            return "ner"
        if v in ("1", "action"):
            return "action"
    return "both"


def run_single(extractor: ChemReXExtractor, text: str, language: str, task: str) -> Dict[str, Any]:
    return extractor.extract(text=text, language=language, task=task)


def _extract_row(extractor: ChemReXExtractor, row: Dict[str, Any], idx: int) -> tuple:
    """Extract one row. Returns (result_or_None, error_str_or_None)."""
    text = (row.get("text") or row.get("input") or "").strip()
    if not text:
        return None, "empty"
    language = _norm_language(row.get("language", ""))
    task = _infer_task_from_type(row.get("task_type")) if row.get("task_type") is not None else "both"
    try:
        return extractor.extract(text=text, language=language, task=task), None
    except Exception as e:
        return {"ner": {}, "action": [], "raw": {}, "error": str(e)}, str(e)


def _attach_prediction(out_row: Dict[str, Any], row: Dict[str, Any], res, err: Optional[str]) -> None:
    if err == "empty":
        out_row["_prediction"] = {"ner": {}, "action": [], "raw": {}}
    else:
        out_row["_prediction"] = res
    if "output" in row:
        out_row["_reference"] = row["output"]


def run_batch(extractor: ChemReXExtractor, input_path: str, output_path: str, max_workers: int = 1) -> None:
    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total = len(rows)
    out_rows: List[Optional[Dict[str, Any]]] = [None] * total
    done = 0

    if max_workers and max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_extract_row, extractor, row, i): i for i, row in enumerate(rows)}
            for fut in as_completed(futures):
                i = futures[fut]
                res, err = fut.result()
                out_row = dict(rows[i])
                _attach_prediction(out_row, rows[i], res, err)
                if err and err != "empty":
                    logger.error(f"row {i} failed: {err}")
                elif err == "empty":
                    logger.warning(f"row {i}: empty text, skipping")
                out_rows[i] = out_row
                done += 1
                if done % 100 == 0 or done == total:
                    logger.info(f"progress: {done}/{total}")
    else:
        for i, row in enumerate(rows):
            res, err = _extract_row(extractor, row, i)
            out_row = dict(row)
            _attach_prediction(out_row, row, res, err)
            if err and err != "empty":
                logger.error(f"row {i} failed: {err}")
            elif err == "empty":
                logger.warning(f"row {i}: empty text, skipping")
            out_rows[i] = out_row
            done += 1
            if done % 100 == 0 or done == total:
                logger.info(f"progress: {done}/{total}")

    written = [r for r in out_rows if r is not None]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in written:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Batch done: {len(written)} rows -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ChemReX Module 1 — Text extraction (NER + Action)")
    parser.add_argument("--text", type=str, default=None, help="Single text segment")
    parser.add_argument("--input", type=str, default=None, help="JSONL batch input")
    parser.add_argument("--output", type=str, default=None, help="JSONL batch output")
    parser.add_argument("--task", type=str, default="both", choices=["ner", "action", "both"])
    parser.add_argument("--language", type=str, default=None, help="zh or en (required for --text)")
    parser.add_argument("--save", type=str, default=None, help="Save single result to JSON file")
    # ── 模型选择（对齐 eval/vllm_lora_reproduce.md）──
    parser.add_argument("--preset", type=str, default=None,
                        help="命名模型预设，默认 qwen3-lora-08（Qwen3-8B + LoRA 0.8, vLLM）。"
                             "预设覆盖 model/base_url/api_key/temperature/max_tokens/enable_thinking。")
    parser.add_argument("--model", type=str, default=None, help="覆盖模型名（如 qwen3-lora-08 / Qwen3-8B）")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI 兼容 API 地址（如 http://localhost:8050/v1）")
    parser.add_argument("--api-key", type=str, default=None, help="API key（如 dummy-token）")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="批处理并发数（1=顺序）。对齐 eval 用 32。")
    parser.add_argument("--list-presets", action="store_true", help="列出可用预设后退出")
    args = parser.parse_args()

    if args.list_presets:
        try:
            import config
            presets = getattr(config, "MODEL_PRESETS", {})
        except ImportError:
            presets = {}
        if not presets:
            print("(no presets configured)")
        for name, p in presets.items():
            print(f"{name:<20} model={p.get('model')} base_url={p.get('base_url')} — {p.get('description','')}")
        return

    extractor = ChemReXExtractor(
        preset=args.preset,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    logger.info(
        f"Extractor configured | preset={args.preset or 'default'} "
        f"model={extractor.client.model} base_url={extractor.client.base_url} "
        f"temperature={extractor.client.temperature} max_tokens={extractor.client.max_tokens} "
        f"enable_thinking={extractor.client.enable_thinking}"
    )

    if args.input:
        if not args.output:
            args.output = os.path.splitext(args.input)[0] + "_predictions.jsonl"
        run_batch(extractor, args.input, args.output, max_workers=args.max_workers)
    elif args.text:
        if not args.language:
            parser.error("--language is required when using --text")
        language = _norm_language(args.language)
        result = run_single(extractor, args.text, language, args.task)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved: {args.save}")
    else:
        parser.error("either --text or --input must be provided")


if __name__ == "__main__":
    main()
