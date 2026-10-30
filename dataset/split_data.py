"""Split dataset into train/val/test with given seed and ratios.

Default behavior is group-level splitting.  Samples with the same ``group_id``
or ``record_id`` stay in the same split, preventing NER/Action text leakage and
patent/procedure leakage.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from sklearn.model_selection import GroupShuffleSplit, train_test_split


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


def get_group(sample: Dict[str, Any], group_key: str = "group_id") -> str:
    group = sample.get(group_key)
    if group:
        return str(group)
    record_id = sample.get("record_id")
    if record_id:
        return f"record:{record_id}"
    text = sample.get("input")
    if text:
        return f"text:{text}"
    return json.dumps(sample, ensure_ascii=False, sort_keys=True)


def split_train_val_test_by_group(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_key: str = "group_id",
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Three-way split by group_id/record_id/input, not by task instance."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    groups = [get_group(sample, group_key) for sample in samples]
    train_idx, temp_idx = next(
        GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed).split(samples, groups=groups)
    )
    train = [samples[i] for i in train_idx]
    temp = [samples[i] for i in temp_idx]
    temp_groups = [groups[i] for i in temp_idx]

    if test_ratio > 0:
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = next(
            GroupShuffleSplit(n_splits=1, train_size=val_size, random_state=seed).split(temp, groups=temp_groups)
        )
        val = [temp[i] for i in val_idx]
        test = [temp[i] for i in test_idx]
    else:
        val, test = temp, []

    return train, val, test


def leakage_audit(train: List[Dict], val: List[Dict], test: List[Dict]) -> Dict[str, Any]:
    datasets = {"train": train, "val": val, "test": test}

    def overlap(field: str) -> Dict[str, int]:
        values = {
            name: {item.get(field) for item in rows if item.get(field)}
            for name, rows in datasets.items()
        }
        return {
            "train-val": len(values["train"] & values["val"]),
            "train-test": len(values["train"] & values["test"]),
            "val-test": len(values["val"] & values["test"]),
        }

    return {
        "split_method": "GroupShuffleSplit on group_id",
        "text_overlap": overlap("input"),
        "record_overlap": overlap("record_id"),
        "group_overlap": overlap("group_id"),
        "patent_overlap": overlap("patent_id"),
    }


def main():
    parser = argparse.ArgumentParser(description="Split dataset JSONL files")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--no_split", action="store_true", help="All data into train")
    parser.add_argument("--split_unit", choices=["group", "sample"], default="group")
    parser.add_argument("--group_key", type=str, default="group_id")
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
    elif args.split_unit == "sample":
        train, val, test = split_train_val_test(
            all_samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    else:
        train, val, test = split_train_val_test_by_group(
            all_samples,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
            args.group_key,
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
        "split_unit": args.split_unit,
        "group_key": args.group_key,
        "leakage_audit": leakage_audit(train, val, test),
    }
    save_json(os.path.join(output_dir, "split_meta.json"), meta)

    print(f"Split done: train={len(train)}, val={len(val)}, test={len(test)} (seed={args.seed})")


if __name__ == "__main__":
    main()
