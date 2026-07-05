#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_chemllm.py
专用于 ChemLLM/InternLM2 系 remote-code 模型的离线评测脚本 (NER + Action)

关键修复：
- 生成时使用 use_cache=True（依赖本地模型代码已修复 past_key_values 判空）
- 生成参数使用 max_new_tokens（只限制新生成长度）

支持模型类型：
1) base: 本地基座模型
2) lora: 本地 LoRA 模型（base + adapter）

评测指标：
- NER: 字段级别 Precision/Recall/F1 + Average
- Action: BLEU-4 / EM (Exact Match) / BERTScore-F1

输出目录：
experiments/<experiment_name>/
  ├── config.json
  ├── eval_metrics.json
  ├── predictions.jsonl   (可选)
  └── evaluate.log
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# 可选指标库
# =========================
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from bert_score import score as bert_score
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("⚠️ 警告: 评估指标库未安装，Action 指标将为 0")
    print("   安装: pip install nltk bert-score")


# =========================
# NER 字段定义
# =========================
NER_FIELDS = [
    "REACTION_PRODUCT",
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
    "TEMPERATURE",
    "TIME",
    "YIELD_PERCENT",
]
FIELD_NAME_MAPPING = {f.lower(): f for f in NER_FIELDS}


# =========================
# Logging
# =========================
def setup_logging(output_dir: str, log_file: str = "evaluate.log") -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file)

    logger = logging.getLogger("ChemLLM-Evaluator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# =========================
# JSON utils
# =========================
def load_data(path: str) -> List[Dict[str, Any]]:
    """
    加载数据文件，自动识别 JSON 或 JSONL 格式
    
    Args:
        path: 数据文件路径（.json 或 .jsonl）
    
    Returns:
        数据列表
    """
    data: List[Dict[str, Any]] = []
    
    # 根据文件扩展名判断格式
    if path.endswith('.jsonl'):
        # JSONL 格式：每行一个 JSON 对象
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: 跳过 {path} 第 {line_no} 行,JSON 解析错误: {e}")
    else:
        # JSON 格式：整个文件是一个 JSON 数组
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
                elif isinstance(loaded, dict):
                    # 如果是单个对象，包装成列表
                    data = [loaded]
                else:
                    print(f"Warning: {path} 不是有效的数据格式")
        except json.JSONDecodeError as e:
            print(f"Warning: 无法解析 {path}: {e}")
    
    return data


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """兼容旧版本的函数名"""
    return load_data(path)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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
# Prompt build
# =========================
def build_user_content(sample: Dict[str, Any]) -> str:
    instruction = (sample.get("instruction") or "").strip()
    user_input = (sample.get("input") or "").strip()
    if instruction and user_input:
        return f"{instruction}\n\n{user_input}"
    return instruction or user_input


def apply_chat_template_compat(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# =========================
# JSON parse helpers
# =========================
def parse_json_output(output_str: str) -> Any:
    if not output_str or not isinstance(output_str, str):
        return None
    s = output_str.strip()

    # ```json ... ```
    if "```json" in s.lower():
        try:
            start_markers = ["```json", "```JSON"]
            end_marker = "```"
            start_idx = -1
            for m in start_markers:
                idx = s.find(m)
                if idx != -1:
                    start_idx = idx + len(m)
                    break
            if start_idx != -1:
                end_idx = s.find(end_marker, start_idx)
                if end_idx != -1:
                    json_str = s[start_idx:end_idx].strip()
                    return json.loads(json_str)
        except Exception:
            pass

    # '''json ... '''
    if "'''json" in s.lower():
        try:
            start_markers = ["'''json", "'''JSON"]
            end_marker = "'''"
            start_idx = -1
            for m in start_markers:
                idx = s.find(m)
                if idx != -1:
                    start_idx = idx + len(m)
                    break
            if start_idx != -1:
                end_idx = s.find(end_marker, start_idx)
                if end_idx != -1:
                    json_str = s[start_idx:end_idx].strip()
                    return json.loads(json_str)
        except Exception:
            pass

    try:
        return json.loads(s)
    except Exception:
        return None


# =========================
# NER metrics
# =========================
def normalize_multi_value(text):
    if text is None or not text:
        return set()
    if not isinstance(text, str):
        text = str(text)
    return {item.strip() for item in text.split(",") if item.strip()}


def normalize_field_name(field_name: str) -> str:
    norm = FIELD_NAME_MAPPING.get(field_name.lower())
    return norm if norm else field_name.upper()


def parse_ner_output(output_str: str) -> Dict[str, str]:
    parsed = parse_json_output(output_str)

    # 兼容模型输出 list 的情况: [ {...} ]
    if isinstance(parsed, list):
        if len(parsed) == 0:
            return {}
        if isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            return {}

    if parsed is None or not isinstance(parsed, dict):
        return {}

    out = {}
    for k, v in parsed.items():
        kk = normalize_field_name(k)
        out[kk] = "" if v is None else str(v)
    return out


def compute_field_level_ner_metrics(predictions: List[str], references: List[str], logger=None) -> Dict[str, Dict[str, float]]:
    field_metrics: Dict[str, Dict[str, float]] = {}
    pred_parse_fail = 0
    ref_parse_fail = 0

    for field in NER_FIELDS:
        TP = FP = FN = 0

        for pred_str, ref_str in zip(predictions, references):
            pred_dict = parse_ner_output(pred_str)
            ref_dict = parse_ner_output(ref_str)

            if not pred_dict and pred_str.strip():
                pred_parse_fail += 1
            if not ref_dict and ref_str.strip():
                ref_parse_fail += 1

            pred_items = normalize_multi_value(pred_dict.get(field, ""))
            ref_items = normalize_multi_value(ref_dict.get(field, ""))

            TP += len(pred_items & ref_items)
            FP += len(pred_items - ref_items)
            FN += len(ref_items - pred_items)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        field_metrics[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP,
            "FP": FP,
            "FN": FN,
        }

    avg_precision = sum(m["precision"] for m in field_metrics.values()) / len(NER_FIELDS)
    avg_recall = sum(m["recall"] for m in field_metrics.values()) / len(NER_FIELDS)
    avg_f1 = sum(m["f1"] for m in field_metrics.values()) / len(NER_FIELDS)
    field_metrics["AVERAGE"] = {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}

    if logger:
        total = len(predictions)
        if pred_parse_fail or ref_parse_fail:
            logger.warning(f"NER解析统计: 预测解析失败 {pred_parse_fail}/{total}, 参考解析失败 {ref_parse_fail}/{total}")

    return field_metrics


# =========================
# Action metrics
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
    
    # 3. 如果都不是，返回空列表
    return []


def compute_action_metrics(predictions: List[str], references: List[str], logger=None) -> Dict[str, float]:
    """
    计算 Action 任务的评估指标
    
    指标包括:
    1. BLEU-4: 4-gram BLEU 分数
    2. EM (Exact Match): 精确匹配率
    3. BERTScore: 基于 BERT 的语义相似度 (F1)
    
    Args:
        predictions: 预测的动作序列列表
        references: 参考的动作序列列表
        logger: 日志记录器
    
    Returns:
        包含各项指标的字典
    """
    if not METRICS_AVAILABLE:
        return {"bleu_4": 0.0, "em": 0.0, "bertscore_f1": 0.0}
    
    bleu_scores = []
    exact_matches = []
    pred_texts_for_bert = []
    ref_texts_for_bert = []
    
    smoother = SmoothingFunction().method1
    
    pred_parse_fail = 0
    ref_parse_fail = 0
    
    for pred_str, ref_str in zip(predictions, references):
        pred_actions = parse_action_output(pred_str)
        ref_actions = parse_action_output(ref_str)
        
        if not pred_actions and pred_str.strip():
            pred_parse_fail += 1
        if not ref_actions and ref_str.strip():
            ref_parse_fail += 1
        
        if not ref_actions:
            continue
        
        # 1. BLEU-4 计算
        try:
            # BLEU-4: 使用 weights=(0.25, 0.25, 0.25, 0.25) 来计算 4-gram BLEU
            bleu_4 = sentence_bleu(
                [ref_actions],
                pred_actions,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoother
            )
            bleu_scores.append(bleu_4)
        except Exception:
            bleu_scores.append(0.0)
        
        # 2. EM (Exact Match) 计算
        # 精确匹配：预测序列和参考序列完全一致
        is_exact_match = (pred_actions == ref_actions)
        exact_matches.append(1.0 if is_exact_match else 0.0)
        
        # 3. 准备 BERTScore 的文本（用空格连接动作）
        pred_text = " ".join(pred_actions) if pred_actions else ""
        ref_text = " ".join(ref_actions)
        pred_texts_for_bert.append(pred_text)
        ref_texts_for_bert.append(ref_text)
    
    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0
    
    # 计算 BERTScore
    bertscore_f1 = 0.0
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
            if logger:
                logger.warning(f"BERTScore 计算失败: {e}")
            bertscore_f1 = 0.0
    
    if logger:
        total = len(predictions)
        if pred_parse_fail or ref_parse_fail:
            logger.warning(f"Action解析统计: 预测解析失败 {pred_parse_fail}/{total}, 参考解析失败 {ref_parse_fail}/{total}")
    
    return {
        "bleu_4": avg(bleu_scores),
        "em": avg(exact_matches),
        "bertscore_f1": bertscore_f1
    }


# =========================
# GPU 控制（重点新增）
# =========================
def apply_gpu_setting(gpu: Optional[str], logger: Optional[logging.Logger] = None) -> None:
    """
    通过环境变量 CUDA_VISIBLE_DEVICES 控制可见 GPU。
    - gpu is None / "" / "auto": 不改环境变量（让 HF/accelerate 自己选）
    - gpu == "-1" / "cpu": 强制 CPU（禁用 CUDA）
    - gpu like "2": 只暴露物理 GPU 2（进程内会变成 cuda:0）
    - gpu like "0,1": 只暴露物理 GPU 0 和 1（进程内是 cuda:0/cuda:1）
    """
    if gpu is None:
        return
    g = str(gpu).strip()
    if g == "":
        return
    low = g.lower()

    if low in ["auto", "none", "null"]:
        # 不改 CUDA_VISIBLE_DEVICES
        return

    if low in ["-1", "cpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if logger:
            logger.info("GPU 设置: 强制 CPU（CUDA_VISIBLE_DEVICES=''）")
        return

    # 形如 "2" 或 "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = g
    if logger:
        logger.info(f"GPU 设置: CUDA_VISIBLE_DEVICES={g}（进程内将重映射为 cuda:0..）")


def resolve_device_map(gpu: Optional[str]) -> str:
    """
    返回给 from_pretrained 的 device_map：
    - 如果 CUDA_VISIBLE_DEVICES 被设置为 ""，那就是 CPU
    - 否则：用 auto 最稳（只会在“可见 GPU”里切）
    """
    # 如果用户强制 CPU
    if gpu is not None and str(gpu).strip().lower() in ["-1", "cpu"]:
        return "cpu"
    # 如果 CUDA 被禁用
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return "cpu"
    return "auto"


# =========================
# Local model generate (ChemLLM-safe)
# =========================
@torch.no_grad()
def generate_with_hf_model_chemllm(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 0.7,
    logger=None,
) -> List[str]:
    """
    ChemLLM 安全的生成函数
    
    参数说明:
    - max_new_tokens: 新生成的最大 token 数（不包括输入）
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    outputs: List[str] = []
    total = len(prompts)
    total_batches = (total + batch_size - 1) // batch_size

    if logger:
        logger.info(f"总共有 {total} 条样本，batch_size={batch_size}，共 {total_batches} 个 batch")
        logger.info(
            f"generate: max_new_tokens={max_new_tokens}, do_sample={do_sample}, top_k={top_k}, top_p={top_p}, "
            f"temperature={temperature}, use_cache=True"
        )

    for b, start in enumerate(range(0, total, batch_size), 1):
        batch_prompts = prompts[start:start + batch_size]

        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        # 每条样本自己的有效输入长度
        if "attention_mask" in enc:
            input_lens = enc["attention_mask"].sum(dim=1).tolist()
        else:
            input_lens = [enc["input_ids"].shape[1]] * enc["input_ids"].shape[0]

        gen_kwargs = dict(
            **enc,
            max_new_tokens=max_new_tokens,  # ✅ 改用 max_new_tokens，表示新生成的 token 数
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        gen_ids = model.generate(**gen_kwargs)

        for row_ids, in_len in zip(gen_ids, input_lens):
            row_ids = row_ids.tolist()
            gen_only = row_ids[in_len:]
            text = tokenizer.decode(gen_only, skip_special_tokens=True)
            outputs.append(text)

        if logger:
            done = len(outputs)
            logger.info(f"Batch {b}/{total_batches} 完成: {done}/{total} ({done/total*100:.1f}%)")

    tokenizer.padding_side = "right"
    return outputs


# =========================
# Load model
# =========================
def load_model_and_tokenizer(args, logger):
    if not args.base_model_path:
        raise ValueError("base/lora 模式需要 --base_model_path")

    device_map = resolve_device_map(args.gpu)
    logger.info(f"device_map = {device_map}")

    logger.info(f"加载 tokenizer: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    logger.info(f"加载 base model: {args.base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=device_map,
        trust_remote_code=True,
    )

    if args.model_type == "lora":
        if not args.adapter_path:
            raise ValueError("lora 模式需要 --adapter_path")
        logger.info(f"加载 LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 开启 KV cache（依赖本地模型代码已修复 past_key_values 判空）
    model.config.use_cache = True
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = True

    model.eval()

    # 打印一些设备信息，方便你看是不是 offload / 多卡
    try:
        logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', None)}")
        logger.info(f"torch.cuda.is_available={torch.cuda.is_available()}")
        logger.info(f"torch.cuda.device_count={torch.cuda.device_count()}")
        if hasattr(model, "hf_device_map"):
            logger.info(f"model.hf_device_map={getattr(model, 'hf_device_map')}")
    except Exception as e:
        logger.warning(f"设备信息打印失败: {e}")

    return model, tokenizer


# =========================
# Main
# =========================
ARGS_CONFIG = {
    "data": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "seed-67", "test.json"),
    "base_model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Model", "ChemLLM-7B-SFT"),
    "adapter_path": "",

    # 推理参数
    "batch_size": 1,
    "max_new_tokens": 512,  # 改为 max_new_tokens：新生成的最大 token 数（不包括输入）
    "do_sample": False,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.0,

    "enable_thinking": False,
    "limit": 0,
    "save_predictions": True,
    "show_examples": 0,

    # GPU 控制（字符串更通用）
    # - "2" 或 "0,1" 或 "auto" 或 "-1"
    "gpu": "2,3",
}

def main():
    parser = argparse.ArgumentParser(description="ChemLLM 多任务评测 (NER + Action)")

    parser.add_argument("--model_type", type=str, required=False, default="base", choices=["base", "lora"])
    parser.add_argument("--data", type=str, default=ARGS_CONFIG["data"])

    parser.add_argument("--base_model_path", type=str, default=ARGS_CONFIG["base_model_path"])
    parser.add_argument("--adapter_path", type=str, default=ARGS_CONFIG["adapter_path"])

    # ✅ 新增：GPU 参数
    parser.add_argument(
        "--gpu",
        type=str,
        default=ARGS_CONFIG["gpu"],
        help="GPU 控制：'2' 或 '0,1' 或 'auto' 或 '-1'(CPU). 内部通过 CUDA_VISIBLE_DEVICES 实现。",
    )

    # 推理参数
    parser.add_argument("--batch_size", type=int, default=ARGS_CONFIG["batch_size"])
    parser.add_argument("--max_new_tokens", type=int, default=ARGS_CONFIG["max_new_tokens"], 
                       help="新生成的最大 token 数（不包括输入）")
    parser.add_argument("--do_sample", action="store_true", help="默认关闭的话可改 ARGS_CONFIG 或命令行显式开")
    parser.add_argument("--top_k", type=int, default=ARGS_CONFIG["top_k"])
    parser.add_argument("--top_p", type=float, default=ARGS_CONFIG["top_p"])
    parser.add_argument("--temperature", type=float, default=ARGS_CONFIG["temperature"])

    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=ARGS_CONFIG["limit"])
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--show_examples", type=int, default=ARGS_CONFIG["show_examples"])

    # 默认值设置
    parser.set_defaults(
        do_sample=ARGS_CONFIG["do_sample"],
        enable_thinking=ARGS_CONFIG["enable_thinking"],
        save_predictions=ARGS_CONFIG["save_predictions"],
    )

    args = parser.parse_args()

    # ========== experiment dir ==========
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = os.path.basename((args.base_model_path or "").rstrip("/")) or "model"
    exp_name = f"{args.model_type}_{model_name}_{timestamp}"

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", exp_name
    )
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)

    # ✅ 在加载模型之前应用 GPU 设置（必须在 from_pretrained 前）
    apply_gpu_setting(args.gpu, logger=logger)

    logger.info(f"开始评测: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"实验名称: {exp_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"GPU 参数: {args.gpu}")

    # ========== save config ==========
    config = {
        "experiment_name": exp_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": args.model_type,
        "data": args.data,
        "model_config": {
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path,
            "gpu": args.gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        },
        "inference_config": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "enable_thinking": args.enable_thinking,
            "use_cache": True,
        },
        "metrics_available": METRICS_AVAILABLE,
    }
    save_json(os.path.join(output_dir, "config.json"), config)

    # ========== load data ==========
    logger.info(f"加载数据: {args.data}")
    samples = load_data(args.data)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
    logger.info(f"样本数: {len(samples)}")

    # ========== load model ==========
    model, tokenizer = load_model_and_tokenizer(args, logger)

    # ========== build prompts / references / task_types ==========
    references = [(s.get("output") or "").strip() for s in samples]
    task_types = [identify_task_type(s) for s in samples]

    logger.info("构建 prompts...")
    prompts: List[str] = []
    for s in samples:
        user_content = build_user_content(s)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""},
        ]
        prompt = apply_chat_template_compat(
            tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prompts.append(prompt)

    # ========== generate ==========
    logger.info("开始生成预测...")
    t0 = time.time()

    predictions = generate_with_hf_model_chemllm(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        logger=logger,
    )

    dt = time.time() - t0
    logger.info(f"生成完成: {len(predictions)} 条, 用时 {dt:.1f}s, 平均 {dt/max(len(predictions),1):.3f}s/条")

    # ========== group by task ==========
    task_data = defaultdict(lambda: {"predictions": [], "references": []})
    for pred, ref, tt in zip(predictions, references, task_types):
        task_data[tt]["predictions"].append(pred)
        task_data[tt]["references"].append(ref)

    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": args.data,
        "total_samples": len(samples),
        "counts": {k: len(v["predictions"]) for k, v in task_data.items()},
        "ner": {},
        "action": {},
    }

    # ========== metrics ==========
    if "ner" in task_data and task_data["ner"]["predictions"]:
        logger.info("计算 NER 指标...")
        ner_field_metrics = compute_field_level_ner_metrics(
            task_data["ner"]["predictions"],
            task_data["ner"]["references"],
            logger=logger,
        )
        metrics["ner"]["field_metrics"] = ner_field_metrics
        metrics["ner"]["avg"] = ner_field_metrics["AVERAGE"]

    if "action" in task_data and task_data["action"]["predictions"]:
        logger.info("计算 Action 指标...")
        metrics["action"] = compute_action_metrics(
            task_data["action"]["predictions"],
            task_data["action"]["references"],
            logger=logger,
        )

    # ========== print summary ==========
    logger.info("\n" + "=" * 90)
    logger.info("评测结果摘要")
    logger.info("=" * 90)

    if "avg" in metrics.get("ner", {}):
        avg = metrics["ner"]["avg"]
        logger.info(f"[NER] Avg Precision={avg['precision']:.4f} | Recall={avg['recall']:.4f} | F1={avg['f1']:.4f}")
        logger.info("       字段级别：")
        logger.info(f"{'字段':<25s} {'Precision':>12s} {'Recall':>12s} {'F1':>12s}")
        logger.info("-" * 65)
        for f in NER_FIELDS:
            m = metrics["ner"]["field_metrics"][f]
            logger.info(f"{f:<25s} {m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1']:>12.4f}")
    else:
        logger.info("[NER] 无 NER 样本")

    if metrics.get("action") and "bleu_4" in metrics["action"]:
        logger.info(
            f"[Action] BLEU-4={metrics['action']['bleu_4']:.4f} | EM={metrics['action']['em']:.4f} | "
            f"BERTScore-F1={metrics['action']['bertscore_f1']:.4f}"
        )
    else:
        logger.info("[Action] 无 Action 样本")

    logger.info("=" * 90 + "\n")

    # ========== show examples ==========
    if args.show_examples and args.show_examples > 0:
        n = min(args.show_examples, len(samples))
        logger.info(f"示例输出（前 {n} 条）:")
        for i in range(n):
            tt = task_types[i]
            logger.info("-" * 90)
            logger.info(f"#{i} [{tt}]")
            logger.info(f"PROMPT(head): {prompts[i][:200].replace(chr(10),' ')} ...")
            logger.info(f"PRED: {predictions[i][:500]}")
            logger.info(f"REF : {references[i][:500]}")

    # ========== save ==========
    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    save_json(metrics_path, metrics)
    logger.info(f"已保存指标: {metrics_path}")

    if args.save_predictions:
        pred_rows = []
        for s, pred, tt in zip(samples, predictions, task_types):
            row = dict(s)
            row["_task_type_eval"] = tt
            row["_prediction"] = pred
            if tt == "ner":
                row["_pred_parsed"] = parse_ner_output(pred)
                row["_ref_parsed"] = parse_ner_output((s.get("output") or "").strip())
            elif tt == "action":
                row["_pred_parsed"] = parse_action_output(pred)
                row["_ref_parsed"] = parse_action_output((s.get("output") or "").strip())
            pred_rows.append(row)

        pred_path = os.path.join(output_dir, "predictions.jsonl")
        save_jsonl(pred_path, pred_rows)
        logger.info(f"已保存预测: {pred_path}")

    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
