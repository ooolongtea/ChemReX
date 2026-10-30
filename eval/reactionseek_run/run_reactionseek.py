#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a faithful ReactionSeek pipeline on ChemReX NER data using a vLLM-served
Qwen3-8B base model via OpenAI-compatible API.

Pipeline (matching ReactionSeek reference repo):
1. extract_gpt.py-style table prompt -> Markdown table (_prediction_raw)
2. structurelize.py-style parsing -> structured record (_table_record)
3. name_to_smiles.py-style compound standardization (optional)
4. time_standardlize.py-style time normalization
5. Map table fields to ChemReX 7 NER JSON fields (_prediction)
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
from tqdm import tqdm

from prompt_template import build_table_prompt
from postprocess import postprocess_predictions


def clean_procedure(text: str) -> str:
    """Remove paragraph IDs like [0257] commonly found in patent procedures."""
    if not text:
        return ""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def call_api(client: OpenAI, model: str, prompt: str, max_tokens: int, temperature: float, enable_thinking: bool = False, retries: int = 2):
    """Call chat completions API with retries."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(2 ** attempt)
            continue
    return f"__ERROR__: {last_error}"


def main():
    parser = argparse.ArgumentParser(description="ReactionSeek faithful pipeline with Qwen3-8B base")
    parser.add_argument("--data", default=str(REPO_ROOT / "data/seed-67/test.json"), help="Path to ChemReX test.json")
    parser.add_argument("--output_dir", default=None, help="Output directory for predictions.jsonl")
    parser.add_argument("--model", default="Qwen3-8B", help="Served model name")
    parser.add_argument("--base_url", default="http://localhost:8050/v1", help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--api_key", default="token-abc123", help="API key")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_workers", type=int, default=32, help="Concurrent API workers")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N NER samples (0 = all)")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3 thinking mode (default: off)")
    parser.add_argument("--skip_standardize_names", action="store_true", help="Skip name_to_smiles standardization")
    parser.add_argument("--skip_standardize_time", action="store_true", help="Skip time standardization")
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    ner_samples = [s for s in data if str(s.get("task_type", "")).upper() == "NER"]
    print(f"Total samples: {len(data)}, NER samples: {len(ner_samples)}")

    if args.limit > 0:
        ner_samples = ner_samples[: args.limit]
        print(f"Limited to first {args.limit} NER samples")

    # Setup output directory
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "results" / f"reactionseek_{args.model}_{ts}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "predictions_raw.jsonl"
    pred_path = output_dir / "predictions.jsonl"
    config_path = output_dir / "config.json"

    config = {
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "max_workers": args.max_workers,
        "limit": args.limit,
        "enable_thinking": args.enable_thinking,
        "num_samples": len(ner_samples),
        "prompt_style": "reactionseek_table_7col",
        "skip_standardize_names": args.skip_standardize_names,
        "skip_standardize_time": args.skip_standardize_time,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=300)

    def process_one(args_pair):
        idx, sample = args_pair
        procedure = clean_procedure(sample.get("input") or "")
        language = sample.get("language", "EN")
        prompt = build_table_prompt(procedure, language)
        pred = call_api(
            client,
            args.model,
            prompt,
            args.max_tokens,
            args.temperature,
            args.enable_thinking,
        )
        row = dict(sample)
        row["_task_type_eval"] = "ner"
        row["_prediction_raw"] = pred
        return row

    print(f"Starting inference with {args.max_workers} workers, max_tokens={args.max_tokens}")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_one, enumerate(ner_samples)),
                total=len(ner_samples),
                desc="ReactionSeek table extraction",
            )
        )

    # Write raw markdown predictions
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    empty = sum(1 for r in results if not (r.get("_prediction_raw") or "").strip())
    errors = sum(1 for r in results if (r.get("_prediction_raw") or "").startswith("__ERROR__"))
    print(f"\nRaw predictions written to {raw_path}")
    print(f"Total: {len(results)} | Empty predictions: {empty} | API errors: {errors}")

    # Run ReactionSeek post-processing pipeline
    print("\nRunning ReactionSeek post-processing pipeline (structurelize + standardize + field mapping)...")
    postprocess_predictions(
        raw_path=str(raw_path),
        out_path=str(pred_path),
        client=client,
        model=args.model,
        standardize_names=not args.skip_standardize_names,
        standardize_time=not args.skip_standardize_time,
    )

    print(f"\nFinal ChemReX predictions written to {pred_path}")


if __name__ == "__main__":
    main()
