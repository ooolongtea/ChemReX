"""Build ChemReX dataset: format → split → stats."""

import argparse
import json
from pathlib import Path

import config

from .format_data import format_data
from .split_data import load_jsonl, save_jsonl, split_train_val_test
from .stats import load_records, compute_field_coverage, compute_text_lengths, compute_patent_distribution


def build(input_path: Path, output_dir: Path, seed: int = 67):
    """Full dataset construction pipeline.

    Args:
        input_path: Path to raw_data.json.
        output_dir: Output directory for processed data.
        seed: Random seed for splitting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(input_path)

    # 1. Format
    format_data(input_path, output_dir)

    # 2. Split (using mixed data for multi-task training)
    mixed_path = output_dir / "data_mixed.jsonl"
    if mixed_path.exists():
        samples = load_jsonl(str(mixed_path))
        train, val, test = split_train_val_test(samples, 0.8, 0.1, 0.1, seed)
        save_jsonl(str(output_dir / "train.jsonl"), train)
        save_jsonl(str(output_dir / "val.jsonl"), val)
        save_jsonl(str(output_dir / "test.jsonl"), test)
        print(f"Split: train={len(train)}, val={len(val)}, test={len(test)} (seed={seed})")

    # 3. Stats
    field_stats = compute_field_coverage(records)
    text_stats = compute_text_lengths(records)
    patent_stats = compute_patent_distribution(records)

    stats_out = {
        "total_records": len(records),
        "field_coverage": field_stats,
        "text_lengths": text_stats,
        "patents": patent_stats,
        "split": {
            "seed": seed,
            "train": len(train) if 'train' in dir() else 0,
            "val": len(val) if 'val' in dir() else 0,
            "test": len(test) if 'test' in dir() else 0,
        },
    }
    stats_path = output_dir / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)
    print(f"Stats saved -> {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build ChemReX dataset")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw_data.json")
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=67)
    args = parser.parse_args()

    build(args.input, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
