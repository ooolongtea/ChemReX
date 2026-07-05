#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_api.py
使用外部大模型 API 做离线评测 (NER + Action)
"""

# =========================
# 配置区域（直接改这里）
# =========================
CONFIG = {
    "data": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "seed-67", "test.json"),
    "model": "gpt-4o",
    "temperature": 0.0,
    "timeout": 300,
    "limit": 0,  # 0 表示全量
    "save_predictions": True,
    "show_examples": 0,
    "max_workers": 5,  # 并行请求线程数
    "max_retries": 3,  # 重试次数
    "backoff_base": 1.0,  # 指数退避基数（秒）
}

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# 让脚本可直接找到项目根目录下的 GPTAPI.py
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from GPTAPI import GPTAPI

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from bert_score import score as bert_score
    METRICS_AVAILABLE = True
except Exception:
    METRICS_AVAILABLE = False

NER_FIELDS = [
    "REACTION_PRODUCT",
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
    "TEMPERATURE",
    "TIME",
    "YIELD_PERCENT",
]
FIELD_NAME_MAPPING = {field.lower(): field for field in NER_FIELDS}


def setup_logging(output_dir: str, log_file: str = "evaluate.log") -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("Evaluator-API")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(output_dir, log_file), encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_data(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def identify_task_type(sample: Dict[str, Any]) -> str:
    # 优先使用数据里的 task_type
    v = sample.get("task_type")
    if isinstance(v, int):
        return "action" if v == 1 else "ner"
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in ("action", "1"):
            return "action"
        if vv in ("ner", "0"):
            return "ner"
    # 兜底：根据 instruction 判别
    ins = (sample.get("instruction") or "").lower()
    if ("动作" in ins) or ("action" in ins) or ("操作" in ins) or ("atomic action verbs" in ins):
        return "action"
    return "ner"


def parse_json_output(output_str: str) -> Any:
    if not output_str or not isinstance(output_str, str):
        return None
    s = output_str.strip()
    if "```json" in s.lower():
        try:
            start = s.lower().find("```json") + 7
            end = s.find("```", start)
            if end != -1:
                return json.loads(s[start:end].strip())
        except Exception:
            pass
    if "'''json" in s.lower():
        try:
            start = s.lower().find("'''json") + 7
            end = s.find("'''", start)
            if end != -1:
                return json.loads(s[start:end].strip())
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_ner_output(output_str: str) -> Dict[str, str]:
    parsed = parse_json_output(output_str)
    if parsed is None or not isinstance(parsed, dict):
        return {}
    out = {}
    for k, v in parsed.items():
        kk = FIELD_NAME_MAPPING.get(str(k).lower(), str(k).upper())
        out[kk] = "" if v is None else str(v)
    return out


def parse_action_output(output_str: str) -> List[str]:
    if not output_str or not isinstance(output_str, str):
        return []
    s = output_str.strip()
    parsed = parse_json_output(s)
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
    if isinstance(parsed, dict):
        for key in ["actions", "action_sequence", "steps", "operations"]:
            if key in parsed and isinstance(parsed[key], list):
                return [str(x).strip() for x in parsed[key] if x is not None and str(x).strip()]
    if "," in s:
        return [a.strip() for a in s.split(",") if a.strip()]
    if ";" in s or "；" in s:
        return [a.strip() for a in s.replace("；", ";").split(";") if a.strip()]
    return []


def normalize_multi_value(text):
    if text is None or not text:
        return set()
    if not isinstance(text, str):
        text = str(text)
    s = text.replace("；", ";")
    parts = []
    for chunk in s.split(";"):
        parts.extend(chunk.split(","))
    return {item.strip() for item in parts if item.strip()}


def compute_field_level_ner_metrics(preds: List[str], refs: List[str], logger=None) -> Dict[str, Dict[str, float]]:
    field_metrics = {}
    pred_parse_fail = 0
    ref_parse_fail = 0
    for field in NER_FIELDS:
        TP = FP = FN = 0
        for p, r in zip(preds, refs):
            pd = parse_ner_output(p)
            rd = parse_ner_output(r)
            if not pd and p.strip():
                pred_parse_fail += 1
            if not rd and r.strip():
                ref_parse_fail += 1
            p_items = normalize_multi_value(pd.get(field, ""))
            r_items = normalize_multi_value(rd.get(field, ""))
            TP += len(p_items & r_items)
            FP += len(p_items - r_items)
            FN += len(r_items - p_items)
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        field_metrics[field] = {"precision": precision, "recall": recall, "f1": f1, "TP": TP, "FP": FP, "FN": FN}
    field_metrics["AVERAGE"] = {
        "precision": sum(m["precision"] for m in field_metrics.values()) / len(NER_FIELDS),
        "recall": sum(m["recall"] for m in field_metrics.values()) / len(NER_FIELDS),
        "f1": sum(m["f1"] for m in field_metrics.values()) / len(NER_FIELDS),
    }
    if logger and (pred_parse_fail or ref_parse_fail):
        logger.warning(f"NER解析统计: 预测解析失败 {pred_parse_fail}/{len(preds)}, 参考解析失败 {ref_parse_fail}/{len(refs)}")
    return field_metrics


def compute_action_metrics(preds: List[str], refs: List[str], logger=None) -> Dict[str, float]:
    if not METRICS_AVAILABLE:
        return {"bleu_4": 0.0, "em": 0.0, "bertscore_f1": 0.0}
    bleu_scores, exact_matches, pred_texts, ref_texts = [], [], [], []
    smoother = SmoothingFunction().method1
    pred_parse_fail = 0
    ref_parse_fail = 0
    for p, r in zip(preds, refs):
        pa = parse_action_output(p)
        ra = parse_action_output(r)
        if not pa and p.strip():
            pred_parse_fail += 1
        if not ra and r.strip():
            ref_parse_fail += 1
        if not ra:
            continue
        try:
            bleu_scores.append(sentence_bleu([ra], pa, weights=(0.25,)*4, smoothing_function=smoother))
        except Exception:
            bleu_scores.append(0.0)
        exact_matches.append(1.0 if pa == ra else 0.0)
        pred_texts.append(" ".join(pa))
        ref_texts.append(" ".join(ra))
    bertscore_f1 = 0.0
    if pred_texts and ref_texts:
        try:
            _, _, F1 = bert_score(pred_texts, ref_texts, model_type="FacebookAI/xlm-roberta-large", num_layers=17, device="cuda", verbose=False)
            bertscore_f1 = F1.mean().item()
        except Exception:
            pass
    if logger and (pred_parse_fail or ref_parse_fail):
        logger.warning(f"Action解析统计: 预测解析失败 {pred_parse_fail}/{len(preds)}, 参考解析失败 {ref_parse_fail}/{len(refs)}")
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"bleu_4": avg(bleu_scores), "em": avg(exact_matches), "bertscore_f1": bertscore_f1}


def _call_api(prompt: str, cfg: Dict[str, Any]) -> str:
    # 每个线程独立创建客户端，避免线程安全问题
    api = GPTAPI(model=cfg["model"], temperature=cfg["temperature"], timeout=cfg["timeout"])
    retries = int(cfg.get("max_retries", 0))
    base = float(cfg.get("backoff_base", 1.0))
    last_err = None
    for attempt in range(retries + 1):
        try:
            return api.answer_wo_vision(prompt=prompt)
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(base * (2 ** attempt))
    raise last_err


def main():
    cfg = CONFIG
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"external_{cfg['model']}_{timestamp}"
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _results_dir = os.path.join(os.path.dirname(_script_dir), "results")
    output_dir = os.path.join(_results_dir, exp_name)
    logger = setup_logging(output_dir)

    logger.info(f"开始评测: {timestamp}")
    logger.info(f"输出目录: {output_dir}")

    samples = load_data(cfg["data"])
    if cfg["limit"] and cfg["limit"] > 0:
        samples = samples[:cfg["limit"]]
    logger.info(f"样本数: {len(samples)}")

    predictions = []
    references = []
    task_types = []
    t0 = time.time()

    # 先准备输入，避免线程内做额外逻辑
    instructions = [(s.get("instruction") or "").strip() for s in samples]
    task_types = [identify_task_type(s) for s in samples]
    references = [(s.get("output") or "").strip() for s in samples]

    # 并行请求大模型
    predictions = [""] * len(samples)
    with ThreadPoolExecutor(max_workers=cfg.get("max_workers", 8)) as ex:
        future_map = {ex.submit(_call_api, ins, cfg): idx for idx, ins in enumerate(instructions)}
        done = 0
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                predictions[idx] = fut.result()
            except Exception as e:
                logger.warning(f"API 调用失败 (样本 {idx+1}): {e}")
                predictions[idx] = ""
            done += 1
            if done % 10 == 0 or done == len(samples):
                pct = done / len(samples) * 100
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0.0
                eta = (len(samples) - done) / speed if speed > 0 else 0.0
                logger.info(
                    f"进度: {done}/{len(samples)} ({pct:.1f}%) | "
                    f"耗时 {elapsed:.1f}s | 速度 {speed:.2f}条/s | ETA {eta/60:.1f}min"
                )

    dt = time.time() - t0
    logger.info(f"生成完成: {len(predictions)} 条, 用时 {dt:.1f}s, 平均 {dt/max(len(predictions),1):.3f}s/条")

    task_data = {"ner": {"predictions": [], "references": []}, "action": {"predictions": [], "references": []}}
    for pred, ref, tt in zip(predictions, references, task_types):
        task_data[tt]["predictions"].append(pred)
        task_data[tt]["references"].append(ref)

    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": cfg["data"],
        "total_samples": len(samples),
        "counts": {k: len(v["predictions"]) for k, v in task_data.items()},
        "ner": {},
        "action": {},
    }

    if task_data["ner"]["predictions"]:
        logger.info("计算 NER 指标...")
        ner_field_metrics = compute_field_level_ner_metrics(
            task_data["ner"]["predictions"], task_data["ner"]["references"], logger=logger
        )
        metrics["ner"]["field_metrics"] = ner_field_metrics
        metrics["ner"]["avg"] = ner_field_metrics["AVERAGE"]

    if task_data["action"]["predictions"]:
        logger.info("计算 Action 指标...")
        metrics["action"] = compute_action_metrics(
            task_data["action"]["predictions"], task_data["action"]["references"], logger=logger
        )

    config = {
        "experiment_name": exp_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": {
            "model": cfg["model"],
            "temperature": cfg["temperature"],
            "timeout": cfg["timeout"],
        },
        "data": cfg["data"],
        "metrics_available": METRICS_AVAILABLE,
    }
    save_json(os.path.join(output_dir, "config.json"), config)
    save_json(os.path.join(output_dir, "eval_metrics.json"), metrics)
    logger.info(f"已保存指标: {os.path.join(output_dir, 'eval_metrics.json')}")

    if cfg["save_predictions"]:
        rows = []
        for s, pred, tt in zip(samples, predictions, task_types):
            row = dict(s)
            row["_task_type_eval"] = tt
            row["_prediction"] = pred
            if tt == "ner":
                row["_pred_parsed"] = parse_ner_output(pred)
                row["_ref_parsed"] = parse_ner_output((s.get("output") or "").strip())
            else:
                row["_pred_parsed"] = parse_action_output(pred)
                row["_ref_parsed"] = parse_action_output((s.get("output") or "").strip())
            rows.append(row)
        pred_path = os.path.join(output_dir, "predictions.jsonl")
        save_jsonl(pred_path, rows)
        logger.info(f"已保存预测: {pred_path}")

    if cfg["show_examples"] and cfg["show_examples"] > 0:
        n = min(cfg["show_examples"], len(samples))
        for i in range(n):
            logger.info("-" * 80)
            logger.info(f"#{i} [{task_types[i]}]")
            logger.info(f"PRED: {predictions[i][:500]}")
            logger.info(f"REF : {references[i][:500]}")


if __name__ == "__main__":
    main()