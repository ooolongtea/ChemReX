"""Split dataset into train/val/test with given seed and ratios.

Uses sklearn.train_test_split for deterministic splitting.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from sklearn.model_selection import train_test_split


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def save_jsonl(path: str, samples: List[Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def split_train_val_test(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Three-way split using sklearn."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train, temp = train_test_split(samples, train_size=train_ratio, random_state=seed, shuffle=True)

    if test_ratio > 0:
        val_size = val_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(temp, train_size=val_size, random_state=seed, shuffle=True)
    else:
        val, test = temp, []

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Split dataset JSONL files")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--no_split", action="store_true", help="All data into train")
    args = parser.parse_args()

    output_dir = args.output_dir or f"split_seed_{args.seed}"

    all_samples = []
    for p in args.input:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")
        data = load_jsonl(p)
        print(f"Loaded {len(data)} from {p}")
        all_samples.extend(data)

    if args.no_split:
        train, val, test = all_samples, [], []
    else:
        train, val, test = split_train_val_test(
            all_samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )

    save_jsonl(os.path.join(output_dir, "train.jsonl"), train)
    save_jsonl(os.path.join(output_dir, "val.jsonl"), val)
    save_jsonl(os.path.join(output_dir, "test.jsonl"), test)

    meta = {
        "seed": args.seed,
        "total": len(all_samples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    save_json(os.path.join(output_dir, "split_meta.json"), meta)

    print(f"Split done: train={len(train)}, val={len(val)}, test={len(test)} (seed={args.seed})")


if __name__ == "__main__":
    main()
