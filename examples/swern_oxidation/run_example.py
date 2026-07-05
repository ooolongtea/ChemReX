"""Run the full ChemReX pipeline on the Swern-oxidation example entry.

Loads entry.json (procedure text + ground truth), then runs all 5 modules in one
process via ChemReXPipeline.run(chain=[...]):
    extraction -> standardize -> image -> coref -> validation

The image module processes examples/swern_oxidation/images/reaction.png (the RDKit
drawing). Results are saved to pipeline_result.json next to this script.

Environment (set before running):
    API_KEY, BASE_URL              — vLLM endpoint (extraction/standardize/coref)
    STANDARDIZE_MODEL, COREF_MODEL — model id served by vLLM
    CUDA_VISIBLE_DEVICES           — GPU for image module
"""

import json
import os
import sys
import time

# Make `import config` and `import pipeline` work when run from repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))  # xyq/ChemReX
sys.path.insert(0, REPO)

from pipeline.pipeline import ChemReXPipeline  # noqa: E402


def main():
    entry = json.load(open(os.path.join(HERE, "entry.json"), encoding="utf-8"))
    text = entry["procedure_text"]
    image_base = os.path.join(HERE)  # contains images/ subdir

    chain = ["extraction", "standardize", "image", "coref"]  # validation runs separately (needs rxnmapper in retroparser env)
    pipe = ChemReXPipeline()
    t0 = time.time()
    result = pipe.run(
        text=text,
        language="en",
        task="both",
        image_base=image_base,
        chain=chain,
    )
    elapsed = time.time() - t0

    out_path = os.path.join(HERE, "pipeline_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save coref-enriched NER (1a/2a SMILES filled from image) for validation.
    coref = result.get("coref", {})
    if isinstance(coref, dict) and "ner" in coref:
        record = {
            "ner": coref["ner"],
            "action": result.get("standardize", {}).get("action", []),
            "language": "en",
            "task": "both",
        }
        coref_ner_path = os.path.join(HERE, "standardized_coref.json")
        with open(coref_ner_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"Coref-enriched NER -> {coref_ner_path}")

    print(f"\n=== Pipeline finished in {elapsed:.1f}s ===")
    print(f"Saved: {out_path}")
    # one-line per-module status
    for mod in chain:
        r = result.get(mod)
        if isinstance(r, dict):
            keys = list(r.keys())
            print(f"  [{mod}] keys={keys[:8]}")
        else:
            print(f"  [{mod}] {type(r).__name__}")


if __name__ == "__main__":
    main()
