"""ChemReX pipeline dispatcher — subcommands for the 4 primary modules.

Each subcommand forwards to the corresponding module CLI (single source of truth
for arguments). Modules are imported lazily so that, e.g., running ``extract`` does
not pull in the heavy image dependencies.

Subcommands:
    extract      — M1: NER (7 fields) + Action (verb sequence)
    image        — M2: image dictionary (ID→SMILES) + reaction structures
    coref        — M3: compound standardization + ID→SMILES mapping
    validate     — M4: field-constraint check + RXNMapper mapping screening
    standardize  — compatibility entry for standalone compound standardization
    chain        — run multiple modules on a single text segment

Examples:
    python run_pipeline.py extract --text "..." --task both --language zh
    python run_pipeline.py extract --input segs.jsonl --output preds.jsonl
    python run_pipeline.py image --image-base <dir>
    python run_pipeline.py standardize --input extraction.json --output std.json
    python run_pipeline.py coref --ner-result ner.json --dict-result dict.json
    python run_pipeline.py validate --ner-result std.json
    python run_pipeline.py chain --text "..." --language zh \
        --chain extract+image+coref+validate --image-base <dir>

Default is NOT chained (each subcommand is independent). Missing inputs warn and skip.
"""

import argparse
import importlib
import json
import os
import sys
from typing import List

from loguru import logger


# subcommand → module path (imported lazily)
_FORWARD = {
    "extract": "pipeline.extraction.run_extraction",
    "image": "pipeline.image.run_image",
    "standardize": "pipeline.standardize.run_standardize",
    "coref": "pipeline.coref.run_coref",
    "validate": "pipeline.validation.run_validation",
}

_CHAIN_MODULES = ("extract", "image", "coref", "validate", "standardize")
# map chain tokens to orchestrator module names
_CHAIN_TOKEN_TO_MODULE = {
    "extract": "extraction",
    "image": "image",
    "standardize": "standardize",
    "coref": "coref",
    "validate": "validation",
}


def _forward(cmd: str, rest: List[str]) -> None:
    """Forward argv to a module CLI's ``main()``."""
    module = importlib.import_module(_FORWARD[cmd])
    saved = sys.argv
    sys.argv = [f"pipeline.{cmd}"] + rest
    try:
        module.main()
    finally:
        sys.argv = saved


def _chain_main(rest: List[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py chain",
        description="Run chained modules on a single clean text segment.",
    )
    parser.add_argument("--text", required=True, help="Clean experimental paragraph")
    parser.add_argument("--language", required=True, help="zh or en (provided by caller)")
    parser.add_argument("--task", default="both", choices=["ner", "action", "both"])
    parser.add_argument("--image-base", default=None, help="Image directory (enables image module)")
    parser.add_argument(
        "--chain",
        default="extract+image+coref+validate",
        help="Modules to run, joined by '+'. Default: extract+image+coref+validate. "
             "Coref standardizes even when image dict_result is unavailable.",
    )
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args(rest)

    tokens = [t.strip() for t in args.chain.split("+") if t.strip()]
    chain: List[str] = []
    for t in tokens:
        if t not in _CHAIN_TOKEN_TO_MODULE:
            parser.error(f"unknown chain module: {t!r} (valid: {', '.join(_CHAIN_MODULES)})")
        chain.append(_CHAIN_TOKEN_TO_MODULE[t])

    from pipeline.pipeline import ChemReXPipeline
    pipe = ChemReXPipeline()
    result = pipe.run(
        text=args.text,
        language=args.language,
        task=args.task,
        image_base=args.image_base,
        chain=chain,
    )

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
        logger.info(f"Saved: {args.output}")
    else:
        print(payload)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Available subcommands: " + ", ".join(list(_FORWARD) + ["chain"]))
        return

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "chain":
        _chain_main(rest)
        return

    if cmd in _FORWARD:
        _forward(cmd, rest)
        return

    print(f"Unknown subcommand: {cmd!r}")
    print("Available subcommands: " + ", ".join(list(_FORWARD) + ["chain"]))
    sys.exit(2)


if __name__ == "__main__":
    main()
