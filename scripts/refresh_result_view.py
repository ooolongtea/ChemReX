"""Maintain a clean view of completed PDFs that have nonempty reaction results.

The original task directories hold resumable MinerU and image caches. This
script only creates/removes symlinks inside results_ready; it never touches
the task output directories or the SQLite task rows.
"""

import argparse
import os
import sqlite3
from pathlib import Path


def refresh(root: Path) -> int:
    view = root / "results_ready"
    desired = {}
    for db_name in ("tasks.db", "stack2_tasks.db"):
        db = root / db_name
        if not db.is_file():
            continue
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for uuid, output_dir in connection.execute(
                    "SELECT uuid, output_dir FROM pdf_task WHERE status='DONE'"):
                output = Path(output_dir)
                result = output / "results.jsonl"
                if result.is_file() and result.stat().st_size > 0:
                    desired[(output.parent.name, uuid)] = output
        finally:
            connection.close()

    for source, uuid in desired:
        category = view / source
        category.mkdir(parents=True, exist_ok=True)
        link = category / uuid
        target = os.path.relpath(desired[(source, uuid)], category)
        if link.is_symlink():
            if os.readlink(link) == target:
                continue
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"Refusing to replace non-symlink: {link}")
        link.symlink_to(target, target_is_directory=True)

    if view.exists():
        for category in view.iterdir():
            if not category.is_dir() or category.is_symlink():
                continue
            for link in category.iterdir():
                if link.is_symlink() and (category.name, link.name) not in desired:
                    link.unlink()
    return len(desired)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(f"{refresh(args.root)} result directories visible")
