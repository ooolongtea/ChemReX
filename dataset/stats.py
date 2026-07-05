"""Dataset statistics: per-patent distribution, per-field coverage."""

import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


FIELDS = [
    "REACTION_PRODUCT", "STARTING_MATERIAL", "REAGENT_CATALYST",
    "SOLVENT", "TEMPERATURE", "TIME", "YIELD_PERCENT",
]


def load_records(data_path: Path) -> List[Dict]:
    with data_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Data must be a top-level list")
    return raw


def compute_field_coverage(records: List[Dict]) -> Dict[str, Dict]:
    total = len(records)
    stats = {}
    for field in FIELDS:
        filled = sum(1 for r in records if r.get(field) and str(r[field]).strip() not in ("", "None"))
        stats[field] = {"filled": filled, "total": total, "ratio": filled / total if total else 0}
    return stats


def compute_text_lengths(records: List[Dict]) -> Dict:
    lengths = [len(r.get("text", "")) for r in records if r.get("text")]
    if not lengths:
        return {"count": 0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / len(lengths),
    }


def compute_patent_distribution(records: List[Dict]) -> Dict:
    counts = Counter()
    missing = 0
    for r in records:
        pid = (r.get("patent_number") or "").strip() or (r.get("file_name") or "").strip()
        if pid:
            counts[pid] += 1
        else:
            missing += 1
    dist = Counter(counts.values())
    return {
        "total_patents": len(counts),
        "missing_patent_id": missing,
        "distribution": dict(sorted(dist.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument("--data", type=Path, required=True, help="Path to raw data JSON")
    args = parser.parse_args()

    records = load_records(args.data)
    print(f"Total records: {len(records)}")

    field_stats = compute_field_coverage(records)
    print("\nField coverage:")
    for f, s in field_stats.items():
        print(f"  {f}: {s['filled']}/{s['total']} ({s['ratio']:.1%})")

    text_stats = compute_text_lengths(records)
    print(f"\nText lengths: min={text_stats.get('min')}, max={text_stats.get('max')}, "
          f"avg={text_stats.get('avg', 0):.0f}")

    patent_stats = compute_patent_distribution(records)
    print(f"\nPatents: {patent_stats['total_patents']} unique")
    print(f"Distribution: {patent_stats['distribution']}")


if __name__ == "__main__":
    main()
