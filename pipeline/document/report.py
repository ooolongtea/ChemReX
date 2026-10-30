"""Summarize a completed whole-document run without rerunning models."""

import argparse
import json
from collections import Counter
from pathlib import Path


def write_report(output_dir: str) -> Path:
    folder = Path(output_dir)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    rows = {}
    with (folder / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["id"]] = row
    status = Counter(row["status"] for row in rows.values())
    validation = Counter(
        row.get("validation", {}).get("rxn_validation", {}).get("status", "unknown")
        for row in rows.values() if row["status"] == "ok"
    )
    complete = manifest["total_chunks"] == len(rows) and not status.get("error")
    smiles_count = 0
    all_compounds = 0
    image_records = 0
    image_reaction_records = 0
    local_image_maps = 0
    resolved_image_ids = 0
    empty_m1 = []
    missing_smiles = []
    for row in rows.values():
        if row["status"] != "ok":
            continue
        image = row.get("image", {})
        image_records += bool(image.get("image_files"))
        image_reaction_records += bool(image.get("image_result"))
        local_image_maps += bool(image.get("dict_result"))
        resolved_image_ids += row.get("coref", {}).get("summary", {}).get("resolved", 0)
        ner = row["extraction"].get("ner", {})
        if not any(bool(value) for value in ner.values()):
            empty_m1.append(row["id"])
        names = [item for value in row.get("coref", {}).get("ner", {}).values()
                 if isinstance(value, list) for item in value if isinstance(item, dict)]
        all_compounds += len(names)
        smiles_count += sum(bool(item.get("smiles")) for item in names)
        if names and not any(item.get("smiles") for item in names):
            missing_smiles.append(row["id"])
    lines = [
        f"# {Path(manifest['source']).stem.replace('_content_list', '')} 全文处理报告", "",
        f"- 来源：`{manifest['source']}`",
        f"- 分块：{manifest['total_chunks']}（Example {manifest['by_kind'].get('example', 0)}；"
        f"Intermediate {manifest['by_kind'].get('intermediate', 0)}；跨页 {manifest['cross_page']}）",
        f"- 完成：{len(rows)}/{manifest['total_chunks']}；成功 {status.get('ok', 0)}；"
        f"错误 {status.get('error', 0)}；全量完成：{'是' if complete else '否'}", "",
        "## 模型输出概况", "",
        f"- M3 化合物条目：{all_compounds}；有 SMILES：{smiles_count}。",
        f"- M2 关联图片的记录：{image_records}；关联到图像反应结果："
        f"{image_reaction_records}；有逐图 ID 映射：{local_image_maps}；"
        f"实际匹配到文本 ID：{resolved_image_ids}。",
        "- M4 状态：" + "；".join(f"{key} {value}" for key, value in sorted(validation.items())) + "。",
        f"- M1 全字段为空：{len(empty_m1)}。",
        f"- M3 有化合物但无 SMILES：{len(missing_smiles)}。", "",
        "## 需要复核的记录", "",
    ]
    review = [row["id"] for row in rows.values()
              if row["status"] == "error" or row.get("validation", {}).get("rxn_validation", {}).get("needs_human_review")]
    lines.append(", ".join(review[:80]) + (f" …（共 {len(review)} 条）" if len(review) > 80 else "") if review else "无")
    lines += ["", "## 解释", "",
              "分块基于文档中的标题与步骤，不等于已经验证了所有反应或串联出唯一合成路线。"
              "`links.json` 中的连接是文字提及关系，必须结合结构与化学逻辑复核。"
              "M4 的 `needs_review` 是模型校验状态，不应当作提取失败。"
              "本次 `skipped` 记录的原因为缺少可用于映射的有效反应物或产物 SMILES。", ""]
    report = folder / "document_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(write_report(args.output_dir))
