#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
离线评估脚本 - 从 predictions.jsonl 重新解析并计算指标

功能:
1. 从 jsonl 文件读取预测结果
2. 重新解析预测和参考答案（支持多种格式）
3. 计算 NER 和 Action 任务的评估指标
"""

import os
# Use HF mirror for BERTScore model downloads when HuggingFace is not reachable.
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path

# =========================
# 配置区域（直接改这里）
# =========================
CONFIG = {
    # 输入：可配置多个文件，或用 data_dir 批量遍历
    "data": [
    ],
    "data_dir": str(Path(__file__).resolve().parent.parent / "results"),  # 遍历 ../results/ 下所有子目录

    # 字段名
    "pred_key": "_prediction",
    "ref_key": "output",

    # 输出
    "out_dir": None,  # None 表示“与 data 同级目录”
    "out_name": "eval_metrics_language.json",

    # 过滤
    "task_filter": "all",  # all / ner / action
    "language_filter": "all",  # all / zh / en

    # 语言分组评估
    "split_by_language": True,
}
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# =============== Optional metrics libs (Action) ===============
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except Exception:
    BERTSCORE_AVAILABLE = False

METRICS_AVAILABLE = NLTK_AVAILABLE and BERTSCORE_AVAILABLE

# =============== NER fields ===============
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


# =========================
# IO
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] skip line {line_no}: {e}")
    return rows


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# Task type
# =========================
def identify_task_type(sample: Dict[str, Any]) -> str:
    """
    识别任务类型（优先使用 _task_type_eval 字段）
    
    支持多种格式:
    - _task_type_eval: "ner" / "action" (最优先，评测脚本添加的标记)
    - task_type: "NER" / "Action" 或 0 / 1
    - 根据 instruction 内容判断
    
    Returns:
        "ner" 或 "action" (小写)
    """
    # 1. 优先使用 _task_type_eval（评测脚本生成的标记）
    if "_task_type_eval" in sample and sample["_task_type_eval"]:
        v = str(sample["_task_type_eval"]).strip().lower()
        if v in ("ner", "action"):
            return v
    
    # 2. 使用 task_type 字段
    if "task_type" in sample:
        v = sample["task_type"]
        if isinstance(v, int):
            if v == 0:
                return "ner"
            if v == 1:
                return "action"
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("0", "ner"):
                return "ner"
            if vv in ("1", "action"):
                return "action"
    
    # 3. 根据 instruction 内容判断
    ins = (sample.get("instruction") or "").lower()
    if ("动作" in ins) or ("action" in ins) or ("操作" in ins) or ("atomic action verbs" in ins):
        return "action"
    
    return "ner"


# =========================
# Robust JSON parsing
# =========================
def _try_load_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_fenced_block(text: str, start_markers: List[str], end_marker: str) -> Optional[str]:
    low = text.lower()
    start_idx = -1
    for m in start_markers:
        idx = low.find(m.lower())
        if idx != -1:
            start_idx = idx + len(m)
            break
    if start_idx == -1:
        return None
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        return None
    return text[start_idx:end_idx].strip()


def _extract_fenced_block_allow_eof(text: str, start_markers: List[str], end_marker: str) -> Optional[str]:
    block = _extract_fenced_block(text, start_markers, end_marker)
    if block is not None:
        return block
    low = text.lower()
    for m in start_markers:
        idx = low.find(m.lower())
        if idx != -1:
            return text[idx + len(m):].strip()
    return None


def parse_json_output(output_str: str) -> Any:
    """
    Supports:
      - ```json ... ```
      - '''json ... '''
      - '''markdown ... ''' (新增支持)
      - ```markdown ... ``` (新增支持)
      - '''json[{}]''' / ```json\n[{}]\n```  (no newline / list wrapper)
      - extra text before/after JSON: extracts outermost JSON object/array
    """
    if not output_str or not isinstance(output_str, str):
        return None
    s = output_str.strip()

    # 1) ```json ... ```
    block = _extract_fenced_block(s, ["```json", "```JSON"], "```")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed

    # 2) '''json ... '''
    block = _extract_fenced_block(s, ["'''json", "'''JSON"], "'''")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed

    # 3) ```markdown ... ``` (新增)
    block = _extract_fenced_block(s, ["```markdown", "```MARKDOWN", "```md", "```MD"], "```")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed

    # 4) '''markdown ... ''' (新增)
    block = _extract_fenced_block(s, ["'''markdown", "'''MARKDOWN", "'''md", "'''MD"], "'''")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed

    # 5) ``` ... ``` (no language)
    block = _extract_fenced_block_allow_eof(s, ["```"], "```")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed

    # 6) ''' ... ''' (no language)
    block = _extract_fenced_block_allow_eof(s, ["'''"], "'''")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed

    # 7) raw json
    parsed = _try_load_json(s)
    if parsed is not None:
        return parsed

    # 8) fallback: extract outermost {...} or [...]
    first_obj = s.find("{")
    first_arr = s.find("[")
    if first_obj == -1 and first_arr == -1:
        return None

    start = first_obj if (first_obj != -1 and (first_arr == -1 or first_obj < first_arr)) else first_arr
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1].strip()
                    return _try_load_json(candidate)

    return None


# =========================
# NER parsing + metrics
# =========================
import re

_WS_RE = re.compile(r"\s+")

def strip_all_whitespace(s: Any) -> str:
    """删除所有空白字符（空格/换行/tab等）"""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return _WS_RE.sub("", s)

_YIELD_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$")

def normalize_yield_value(v: Any) -> str:
    """
    产率归一化：
    - None / "" / 空白 => ""
    - "99" / "99%" / " 99 % " => "99"
    - "99.0%" => "99.0"
    - 其他复杂文本（比如 "90%(crude)"）会先去空白再尽量抽取开头数字
    """
    s = strip_all_whitespace(v)
    if not s:
        return ""

    m = _YIELD_NUM_RE.match(s)
    if m:
        return m.group(1)

    # 兜底：抽取第一个数字（避免 "90%(crude)" 这种）
    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    if m2:
        return m2.group(1)

    # 实在不行就返回去空白后的原串
    return s

from typing import Any, Optional

_NULL_LIKE = {"", "null", "none"}  # 只按你需求最小集合即可

def normalize_multi_value(text: Any, field: Optional[str] = None) -> set:
    """
    目标：标签为 null 时，预测的 null / "" / None 都算匹配（都映射为“空集合”）
    其它逻辑保持：去空白 + split(",")
    """
    # ✅ 统一空值：None / "" / "null" / "None" 都当作空
    if text is None:
        return set()
    if not isinstance(text, str):
        text = str(text)
    s0 = text.strip()
    if s0.lower() in _NULL_LIKE:
        return set()
    if s0.lower() == "n/a" or s0.lower() == "notspecified":
        return set()

    # ✅ 保持你原来的逻辑
    if field == "YIELD_PERCENT":
        y = normalize_yield_value(text)   # 你原来已有的
        return set() if not y else {y}

    # 先按分号分隔，再按逗号分隔，最后去空白
    s = str(text).replace("；", ";")
    parts = []
    for chunk in s.split(";"):
        parts.extend(chunk.split(","))
    cleaned = {strip_all_whitespace(p) for p in parts if strip_all_whitespace(p)}
    return cleaned



def normalize_field_name(field_name: str) -> str:
    if not field_name:
        return ""
    mapped = FIELD_NAME_MAPPING.get(field_name.strip().lower())
    return mapped if mapped else field_name.strip().upper()


def parse_ner_output(output_str: str) -> Dict[str, str]:
    parsed = parse_json_output(output_str)

    # Support list[dict] e.g. ```json [ {..} ] ```
    if isinstance(parsed, list):
        if len(parsed) == 0:
            return {}
        if isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            return {}

    if parsed is None or not isinstance(parsed, dict):
        return {}

    out: Dict[str, str] = {}
    for k, v in parsed.items():
        kk = normalize_field_name(str(k))
        out[kk] = "" if v is None else str(v)
    return out


def _record_parse_failures(
    rows: List[Dict[str, Any]],
    task_types: List[str],
    preds: List[str],
    refs: List[str],
    out_dir: str,
) -> None:
    failures: List[Dict[str, Any]] = []
    for i, (s, tt, pred, ref) in enumerate(zip(rows, task_types, preds, refs)):
        if tt == "ner":
            pred_parsed = parse_ner_output(pred)
            ref_parsed = parse_ner_output(ref)
            if (not pred_parsed and pred.strip()) or (not ref_parsed and ref.strip()):
                failures.append({
                    "index": i,
                    "task_type": tt,
                    "instruction": s.get("instruction"),
                    "input": s.get("input"),
                    "pred": pred,
                    "ref": ref,
                    "pred_parsed": pred_parsed,
                    "ref_parsed": ref_parsed,
                })
        else:
            pred_parsed = parse_action_output(pred)
            ref_parsed = parse_action_output(ref)
            if (not pred_parsed and pred.strip()) or (not ref_parsed and ref.strip()):
                failures.append({
                    "index": i,
                    "task_type": tt,
                    "instruction": s.get("instruction"),
                    "input": s.get("input"),
                    "pred": pred,
                    "ref": ref,
                    "pred_parsed": pred_parsed,
                    "ref_parsed": ref_parsed,
                })

    if failures:
        fail_path = os.path.join(out_dir, "parse_failures.json")
        save_json(fail_path, failures)
        print(f"[INFO] 解析失败样本已保存: {fail_path} ({len(failures)})")


def compute_field_level_ner_metrics(preds: List[str], refs: List[str], logger=None) -> Dict[str, Dict[str, float]]:
    field_metrics: Dict[str, Dict[str, float]] = {}

    # 统计解析失败情况
    pred_parse_fail = 0
    ref_parse_fail = 0

    for field in NER_FIELDS:
        TP = FP = FN = 0
        for p, r in zip(preds, refs):
            pd = parse_ner_output(p)
            rd = parse_ner_output(r)

            # 统计解析失败（只在第一个字段时统计，避免重复）
            if field == NER_FIELDS[0]:
                if not pd and p.strip():
                    pred_parse_fail += 1
                if not rd and r.strip():
                    ref_parse_fail += 1

            p_items = normalize_multi_value(pd.get(field, ""), field=field)
            r_items = normalize_multi_value(rd.get(field, ""), field=field)

            TP += len(p_items & r_items)
            FP += len(p_items - r_items)
            FN += len(r_items - p_items)

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        field_metrics[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP,
            "FP": FP,
            "FN": FN,
        }

    avg_precision = sum(field_metrics[f]["precision"] for f in NER_FIELDS) / len(NER_FIELDS)
    avg_recall = sum(field_metrics[f]["recall"] for f in NER_FIELDS) / len(NER_FIELDS)
    avg_f1 = sum(field_metrics[f]["f1"] for f in NER_FIELDS) / len(NER_FIELDS)

    field_metrics["AVERAGE"] = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }

    # 打印解析统计
    total_samples = len(preds)
    if pred_parse_fail > 0 or ref_parse_fail > 0:
        msg = f"NER 解析失败统计: 预测 {pred_parse_fail}/{total_samples}, 参考解析失败 {ref_parse_fail}/{total_samples}"
        if logger:
            logger.warning(msg)
        else:
            print(f"[WARN] {msg}")

    return field_metrics


# =========================
# Action parsing + metrics
# =========================
def parse_action_output(output_str: str) -> List[str]:
    """
    解析 Action 任务的输出
    
    支持多种格式:
    1. 逗号分隔字符串: "dissolve, cool, add" (新格式，主流)
    2. 分号分隔字符串: "dissolve; cool; add" (旧格式)
    3. JSON 数组: ["dissolve", "cool", "add"]
    4. JSON 对象: {"actions": ["dissolve", "cool", "add"]}
    
    Returns:
        动作列表
    """
    if not output_str or not isinstance(output_str, str):
        return []
    
    output_str = output_str.strip()
    
    # 1. 尝试解析 JSON 格式
    parsed = parse_json_output(output_str)
    if parsed is not None:
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
        if isinstance(parsed, dict):
            for key in ["actions", "action_sequence", "steps", "operations"]:
                if key in parsed and isinstance(parsed[key], list):
                    return [str(x).strip() for x in parsed[key] if x is not None and str(x).strip()]
    
    # 2. 如果不是 JSON，尝试作为分隔字符串解析
    # 优先使用逗号分隔（新格式，主流）
    if ',' in output_str:
        actions = [a.strip() for a in output_str.split(',') if a.strip()]
        return actions
    elif ';' in output_str or '；' in output_str:
        # 兼容旧格式：分号分隔
        output_str = output_str.replace('；', ';')
        actions = [a.strip() for a in output_str.split(';') if a.strip()]
        return actions
    elif '、' in output_str:
        actions = [a.strip() for a in output_str.split('、') if a.strip()]
        return actions
    
    # 3. 如果都不是，返回空列表
    return []


def compute_action_metrics(preds: List[str], refs: List[str], logger=None) -> Dict[str, float]:
    """
    计算 Action 任务的评估指标

    指标包括:
    1. BLEU-4: 4-gram BLEU 分数
    2. EM (Exact Match): 精确匹配率
    3. BERTScore: 基于 BERT 的语义相似度 (F1)

    Args:
        preds: 预测的动作序列列表
        refs: 参考的动作序列列表
        logger: 可选日志对象

    Returns:
        包含各项指标的字典
    """
    if not NLTK_AVAILABLE:
        return {"bleu_4": 0.0, "em": 0.0, "bertscore_f1": None}

    bleu_scores = []
    exact_matches = []
    pred_texts_for_bert = []
    ref_texts_for_bert = []

    smoother = SmoothingFunction().method1

    # 统计解析失败情况
    pred_parse_fail = 0
    ref_parse_fail = 0

    for p, r in zip(preds, refs):
        pa = parse_action_output(p)
        ra = parse_action_output(r)

        # 统计解析失败
        if not pa and p.strip():
            pred_parse_fail += 1
        if not ra and r.strip():
            ref_parse_fail += 1

        if not ra:
            continue

        # 1. BLEU-4 计算
        try:
            # BLEU-4: 使用 weights=(0.25, 0.25, 0.25, 0.25) 来计算 4-gram BLEU
            bleu_4 = sentence_bleu(
                [ra],
                pa,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoother
            )
            bleu_scores.append(bleu_4)
        except Exception:
            bleu_scores.append(0.0)

        # 2. EM (Exact Match) 计算
        # 精确匹配：预测序列和参考序列完全一致
        is_exact_match = (pa == ra)
        exact_matches.append(1.0 if is_exact_match else 0.0)

        # 3. 准备 BERTScore 的文本（用空格连接动作）
        pred_text = " ".join(pa) if pa else ""
        ref_text = " ".join(ra)
        pred_texts_for_bert.append(pred_text)
        ref_texts_for_bert.append(ref_text)

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # 计算 BERTScore
    bertscore_f1 = None
    if BERTSCORE_AVAILABLE:
        if pred_texts_for_bert and ref_texts_for_bert:
            try:
                # 使用 BERTScore 计算语义相似度
                # 采用多语言模型 FacebookAI/xlm-roberta-large 统一处理中英文
                P, R, F1 = bert_score(
                    pred_texts_for_bert,
                    ref_texts_for_bert,
                    model_type="FacebookAI/xlm-roberta-large",
                    num_layers=17,
                    device="cuda",
                    verbose=False
                )
                bertscore_f1 = F1.mean().item()
            except Exception as e:
                msg = f"BERTScore 计算失败: {e}"
                if logger:
                    logger.warning(msg)
                else:
                    print(f"[WARN] {msg}")
                bertscore_f1 = None
    else:
        msg = "BERTScore 不可用: 未安装 bert-score / torch"
        if logger:
            logger.warning(msg)
        else:
            print(f"[WARN] {msg}")

    # 打印解析统计
    total_samples = len(preds)
    if pred_parse_fail > 0 or ref_parse_fail > 0:
        msg = f"Action 解析失败统计: 预测 {pred_parse_fail}/{total_samples}, 参考解析失败 {ref_parse_fail}/{total_samples}"
        if logger:
            logger.warning(msg)
        else:
            print(f"[WARN] {msg}")

    return {
        "bleu_4": avg(bleu_scores),
        "em": avg(exact_matches),
        "bertscore_f1": bertscore_f1
    }


# =========================
# Main
# =========================
def main():
    cfg = CONFIG
    data_paths: List[str] = []
    if cfg["data_dir"]:
        for name in sorted(os.listdir(cfg["data_dir"])):
            sub = os.path.join(cfg["data_dir"], name)
            if os.path.isdir(sub):
                cand = os.path.join(sub, "predictions.jsonl")
                if os.path.exists(cand):
                    data_paths.append(cand)
    data_paths.extend([p for p in cfg["data"] if p])

    if not data_paths:
        print("[WARN] 未找到任何 predictions.jsonl")
        return

    for data_path in data_paths:
        print(f"[INFO] 加载数据: {data_path}")
        rows = load_jsonl(data_path)
        print(f"[INFO] 总样本数: {len(rows)}")

        print(f"[INFO] 从字段读取: 预测='{cfg['pred_key']}', 参考='{cfg['ref_key']}'")
        
        task_data = defaultdict(lambda: {"predictions": [], "references": []})
        task_types: List[str] = []
        preds: List[str] = []
        refs: List[str] = []
        langs: List[str] = []
        filtered_rows: List[Dict[str, Any]] = []
        for s in rows:
            tt = identify_task_type(s)
            pred = (s.get(cfg["pred_key"]) or "").strip()
            ref = (s.get(cfg["ref_key"]) or "").strip()
            lang = (s.get("language") or "").strip().lower()

            if cfg["task_filter"] in ("ner", "action") and tt != cfg["task_filter"]:
                continue
            if cfg["language_filter"] in ("zh", "en") and lang != cfg["language_filter"]:
                continue

            filtered_rows.append(s)
            task_types.append(tt)
            preds.append(pred)
            refs.append(ref)
            langs.append(lang)
            task_data[tt]["predictions"].append(pred)
            task_data[tt]["references"].append(ref)

        # 打印任务分布
        print(f"[INFO] 任务分布:")
        for task_type, data in task_data.items():
            print(f"  - {task_type.upper()}: {len(data['predictions'])} 样本")

        def _compute_metrics(p_list, r_list, tt_list) -> Dict[str, Any]:
            data = defaultdict(lambda: {"predictions": [], "references": []})
            for p, r, tt in zip(p_list, r_list, tt_list):
                data[tt]["predictions"].append(p)
                data[tt]["references"].append(r)
            out: Dict[str, Any] = {"ner": {}, "action": {}}
            if "ner" in data and data["ner"]["predictions"]:
                field_metrics = compute_field_level_ner_metrics(
                    data["ner"]["predictions"], data["ner"]["references"]
                )
                out["ner"]["field_metrics"] = field_metrics
                out["ner"]["avg"] = field_metrics["AVERAGE"]
            if "action" in data and data["action"]["predictions"]:
                out["action"] = compute_action_metrics(
                    data["action"]["predictions"], data["action"]["references"]
                )
            return out

        metrics: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": data_path,
            "total_samples": len(filtered_rows),
            "counts": {k: len(v["predictions"]) for k, v in task_data.items()},
            "ner": {},
            "action": {},
            "filters": {
                "task_filter": cfg["task_filter"],
                "language_filter": cfg["language_filter"],
            },
        }

        metrics.update(_compute_metrics(preds, refs, task_types))

        if cfg["split_by_language"]:
            by_language = {}
            for lang in sorted({l for l in langs if l}):
                idxs = [i for i, l in enumerate(langs) if l == lang]
                p_list = [preds[i] for i in idxs]
                r_list = [refs[i] for i in idxs]
                t_list = [task_types[i] for i in idxs]
                by_language[lang.upper()] = _compute_metrics(p_list, r_list, t_list)
            metrics["by_language"] = by_language

        # ✅ 改动：输出目录与 data 同级
        out_dir = cfg["out_dir"] if cfg["out_dir"] else os.path.dirname(os.path.abspath(data_path))
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, cfg["out_name"])

        # ✅ 覆盖同名文件：save_json 用 "w" 写入，本身会覆盖
        if os.path.exists(out_path):
            print(f"[INFO] overwrite existing file: {out_path}")

        save_json(out_path, metrics)

        print(f"[OK] saved: {out_path}")
        _record_parse_failures(filtered_rows, task_types, preds, refs, out_dir)
        print(json.dumps({
            "total_samples": metrics["total_samples"],
            "counts": metrics["counts"],
            "ner_avg_f1": metrics.get("ner", {}).get("avg", {}).get("f1", None),
            "action_bleu_4": metrics.get("action", {}).get("bleu_4", None),
            "action_em": metrics.get("action", {}).get("em", None),
            "action_bertscore_f1": metrics.get("action", {}).get("bertscore_f1", None),
        }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
