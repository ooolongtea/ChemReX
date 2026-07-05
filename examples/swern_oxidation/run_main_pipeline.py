"""主流程驱动（retroparser 环境）：extraction → standardize → coref → validation。

在 retroparser 环境中运行（含 cirpy / pubchempy / rxnmapper），读取 entry.json 和
图像模块预先生成的 dict_result.json，依次跑 4 个文本侧模块并保存中间结果。

用法:
    python run_main_pipeline.py --entry entry.json --dict-result dict.json --output-dir out/
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from pipeline.extraction.extractor import ChemReXExtractor
from pipeline.standardize.standardizer import Standardizer
from pipeline.coref.coref_resolver import CorefResolver
from pipeline.validation.validator import ReactionValidator


def main():
    p = argparse.ArgumentParser(description="ChemReX main pipeline (retroparser env)")
    p.add_argument("--entry", required=True, help="entry.json (procedure_text + language)")
    p.add_argument("--dict-result", required=True, help="image dict_result JSON ({id: smiles})")
    p.add_argument("--output-dir", required=True, help="output directory")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    entry = json.load(open(args.entry, encoding="utf-8"))
    text = entry["procedure_text"]
    lang_raw = str(entry.get("language", "en")).strip().lower()
    if lang_raw in ("zh", "zh-cn", "chinese", "cn"):
        language = "zh"
    elif lang_raw in ("en", "english"):
        language = "en"
    else:
        language = lang_raw

    with open(args.dict_result, encoding="utf-8") as f:
        dict_result = json.load(f)

    def save(name, obj):
        path = os.path.join(args.output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path

    t0 = time.time()

    # 1. extraction (M1)
    print(">>> [1/4] extraction ...")
    ext = ChemReXExtractor()
    extraction = ext.extract(text=text, language=language, task="both")
    save("1_extraction.json", extraction)

    # 2. standardize (M3a — name→SMILES 级联: LLM → CIR → PubChem → OPSIN → RDKit)
    print(">>> [2/4] standardize ...")
    std = Standardizer()
    standardized = std.standardize(extraction)
    save("2_standardized.json", standardized)

    # 3. coref (M3b — ID→SMILES 映射，消费 image dict_result)
    print(">>> [3/4] coref ...")
    coref = CorefResolver()
    payload = dict(standardized)
    payload["dict_result"] = dict_result
    coref_result = coref.resolve(payload)
    save("3_coref.json", coref_result)

    # 4. validation (M4 — 字段约束 + RXNMapper 映射筛查)
    print(">>> [4/4] validation ...")
    val = ReactionValidator()
    validation_result = val.validate_record(coref_result)
    save("4_validated.json", validation_result)

    elapsed = time.time() - t0
    print(f"\n=== Main pipeline done in {elapsed:.1f}s ===")
    print(f"Output dir: {args.output_dir}")

    # 摘要
    ner = coref_result.get("ner", {})
    print("\n--- Coref NER ---")
    for field in ("STARTING_MATERIAL", "REACTION_PRODUCT", "REAGENT_CATALYST", "SOLVENT"):
        print(f"  {field}: {json.dumps(ner.get(field), ensure_ascii=False)}")
    print(f"  TEMPERATURE={ner.get('TEMPERATURE')}")
    print(f"  TIME={ner.get('TIME')}")
    print(f"  YIELD_PERCENT={ner.get('YIELD_PERCENT')}")

    print(f"\n--- Coref summary ---")
    print(f"  {json.dumps(coref_result.get('summary'), ensure_ascii=False)}")
    print(f"  id_smiles_mapping: {json.dumps(coref_result.get('id_smiles_mapping'), ensure_ascii=False)}")

    print(f"\n--- Validation ---")
    rv = validation_result.get("rxn_validation", {})
    print(f"  status={rv.get('status')}  diagnosis={rv.get('diagnosis')}  needs_review={rv.get('needs_human_review')}")


if __name__ == "__main__":
    main()
