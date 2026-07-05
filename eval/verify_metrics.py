#!/usr/bin/env python
"""verify_metrics.py — 用 ChemReX/eval/evaluate.py 的指标逻辑重评已有 predictions.jsonl。

用法:
  python verify_metrics.py <predictions.jsonl> [--data <test.json>]

若不传 --data，则用样本里的 output 字段作为 reference（与 predictions.jsonl 内嵌一致）。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 导入 ChemReX 版 evaluate.py 的指标函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="predictions.jsonl 路径")
    ap.add_argument("--data", default=None, help="原始 test.json（用于取 reference output）")
    ap.add_argument("--out", default=None, help="输出 eval_metrics JSON 路径")
    args = ap.parse_args()

    preds = load_jsonl(args.predictions)
    print(f"加载预测: {len(preds)} 条  from {args.predictions}")

    # 预测里的字段: _prediction, _reference(若有), task_type, output, ...
    # 优先用 _reference；否则用 output；否则从 --data 取
    refs = []
    for r in preds:
        ref = r.get("_reference") or r.get("output") or ""
        refs.append(ref)

    # 若指定 --data，用 input 对齐取 output
    if args.data:
        data = json.load(open(args.data, encoding="utf-8"))
        # 建立 input -> output 映射
        input2out = {}
        for d in data:
            input2out[d.get("input", "")] = d.get("output", "")
        refs = []
        missing = 0
        for r in preds:
            inp = r.get("input", "")
            if inp in input2out:
                refs.append(input2out[inp])
            else:
                refs.append(r.get("output", ""))
                missing += 1
        if missing:
            print(f"⚠️ {missing} 条预测未在 --data 中找到对应 input，回退用内嵌 output")

    task_types = [r.get("task_type", "unknown") for r in preds]
    predictions = [r.get("_prediction", "") for r in preds]

    # 分组
    task_data = defaultdict(lambda: {"predictions": [], "references": []})
    for p, r, t in zip(predictions, refs, task_types):
        task_data[t]["predictions"].append(p)
        task_data[t]["references"].append(r)

    metrics = {
        "total_samples": len(preds),
        "counts": {k: len(v["predictions"]) for k, v in task_data.items()},
        "ner": {},
        "action": {},
    }

    # NER
    if "ner" in task_data and task_data["ner"]["predictions"]:
        fm = ev.compute_field_level_ner_metrics(
            task_data["ner"]["predictions"],
            task_data["ner"]["references"],
        )
        metrics["ner"]["field_metrics"] = fm
        metrics["ner"]["avg"] = fm["AVERAGE"]

    # Action
    if "action" in task_data and task_data["action"]["predictions"]:
        metrics["action"] = ev.compute_action_metrics(
            task_data["action"]["predictions"],
            task_data["action"]["references"],
        )

    # 打印
    print("\n" + "=" * 80)
    print(f"样本数: {metrics['total_samples']}  分布: {metrics['counts']}")
    if "avg" in metrics.get("ner", {}):
        avg = metrics["ner"]["avg"]
        print(f"[NER] Avg P={avg['precision']:.4f} R={avg['recall']:.4f} F1={avg['f1']:.4f}")
        print(f"  {'字段':<22}{'P':>9}{'R':>9}{'F1':>9}")
        for f in ev.NER_FIELDS:
            m = metrics["ner"]["field_metrics"][f]
            print(f"  {f:<22}{m['precision']:>9.4f}{m['recall']:>9.4f}{m['f1']:>9.4f}")
    if metrics.get("action") and "bleu_4" in metrics["action"]:
        a = metrics["action"]
        print(f"[Action] BLEU-4={a['bleu_4']:.4f}  EM={a['em']:.4f}  BERT={a['bertscore_f1']:.4f}")
    print("=" * 80)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(metrics, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
