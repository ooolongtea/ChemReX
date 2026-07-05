#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# 配置区域
# =========================
CONFIG = {
    "data_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "seed-67", "test.json"),
    "base_model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Model", "Qwen3-8B"),
    "adapter_paths": [
        # 列出要评测的 LoRA adapter（相对于 experiments/models/ 或绝对路径）
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3-8B-LoRA-0.7"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3-8B-LoRA-0.6"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3-8B-LoRA-0.9"),
        # 可以添加更多 adapter 路径
    ],
    "output_root": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"),
    "gpu_id": "1",
    "use_bf16": True,
    "use_fp16": False,
    "max_new_tokens": 512,
    "batch_size": 4,
    "do_sample": False,
    "temperature": 0.0,
    "enable_thinking": False,
    "limit": 0,
    "save_predictions": True,
    "show_examples": 3,
}


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


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "evaluate.log")

    logger = logging.getLogger("NER-LoRA-Evaluator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_data(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"Warning: skip {path}:{line_no}, invalid json: {exc}")
        return rows

    with open(path, "r", encoding="utf-8") as file:
        loaded = json.load(file)
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        return [loaded]
    return []


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_ner_sample(sample: Dict[str, Any]) -> bool:
    task_type = sample.get("task_type")
    if isinstance(task_type, int):
        return task_type == 0
    if isinstance(task_type, str):
        return task_type.strip().lower() == "ner"

    instruction = (sample.get("instruction") or "").lower()
    return "reaction_product" in instruction and "yield_percent" in instruction


def build_user_content(sample: Dict[str, Any]) -> str:
    instruction = (sample.get("instruction") or "").strip()
    user_input = (sample.get("input") or "").strip()

    if instruction and ("###实验步骤：" in instruction or "###Procedure:" in instruction):
        return instruction
    if instruction and user_input:
        return f"{instruction}\n\n{user_input}"
    return instruction or user_input


def apply_chat_template_compat(tokenizer, messages: List[Dict[str, str]], enable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_json_output(output_str: str) -> Any:
    if not output_str or not isinstance(output_str, str):
        return None
    text = output_str.strip()

    for start_marker, end_marker in (("```json", "```"), ("```JSON", "```"), ("'''json", "'''"), ("'''JSON", "'''")):
        start_idx = text.find(start_marker)
        if start_idx == -1:
            continue
        start_idx += len(start_marker)
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            continue
        try:
            return json.loads(text[start_idx:end_idx].strip())
        except Exception:
            pass

    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_multi_value(text: Optional[str]) -> set:
    if text is None or text == "":
        return set()
    if not isinstance(text, str):
        text = str(text)
    normalized = text.replace("；", ";")
    items: List[str] = []
    for chunk in normalized.split(";"):
        for item in chunk.split(","):
            cleaned = item.strip()
            if cleaned:
                items.append(cleaned)
    return set(items)


def normalize_field_name(field_name: str) -> str:
    normalized = FIELD_NAME_MAPPING.get(field_name.lower())
    return normalized if normalized else field_name.upper()


def parse_ner_output(output_str: str) -> Dict[str, str]:
    parsed = parse_json_output(output_str)
    if not isinstance(parsed, dict):
        return {}

    result: Dict[str, str] = {}
    for key, value in parsed.items():
        field_name = normalize_field_name(str(key))
        result[field_name] = "" if value is None else str(value)
    return result


def compute_field_level_ner_metrics(predictions: List[str], references: List[str], logger: logging.Logger) -> Dict[str, Dict[str, float]]:
    field_metrics: Dict[str, Dict[str, float]] = {}
    pred_parse_fail = 0
    ref_parse_fail = 0

    for field in NER_FIELDS:
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for pred_str, ref_str in zip(predictions, references):
            pred_dict = parse_ner_output(pred_str)
            ref_dict = parse_ner_output(ref_str)

            if not pred_dict and pred_str.strip():
                pred_parse_fail += 1
            if not ref_dict and ref_str.strip():
                ref_parse_fail += 1

            pred_items = normalize_multi_value(pred_dict.get(field, ""))
            ref_items = normalize_multi_value(ref_dict.get(field, ""))

            true_positive += len(pred_items & ref_items)
            false_positive += len(pred_items - ref_items)
            false_negative += len(ref_items - pred_items)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        field_metrics[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": true_positive,
            "FP": false_positive,
            "FN": false_negative,
        }

    field_metrics["AVERAGE"] = {
        "precision": sum(metric["precision"] for metric in field_metrics.values()) / len(NER_FIELDS),
        "recall": sum(metric["recall"] for metric in field_metrics.values()) / len(NER_FIELDS),
        "f1": sum(metric["f1"] for metric in field_metrics.values()) / len(NER_FIELDS),
    }

    total_samples = len(predictions)
    if pred_parse_fail or ref_parse_fail:
        logger.warning(
            "NER解析统计: 预测解析失败 %s/%s, 参考解析失败 %s/%s",
            pred_parse_fail,
            total_samples,
            ref_parse_fail,
            total_samples,
        )

    return field_metrics


def parse_gpu_config(gpu_id: Any, logger: logging.Logger) -> str:
    if gpu_id is None or gpu_id == -1:
        logger.info("使用 CPU")
        return "cpu"

    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        return "cpu"

    if isinstance(gpu_id, str):
        gpu_id = gpu_id.strip()
        if gpu_id.lower() == "auto":
            logger.info("使用 device_map='auto'")
            return "auto"
        if "," in gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            logger.info("设置 CUDA_VISIBLE_DEVICES=%s，并使用 device_map='auto'", gpu_id)
            return "auto"
        try:
            gpu_index = int(gpu_id)
        except ValueError:
            logger.warning("无效的 gpu_id=%s，回退到 CPU", gpu_id)
            return "cpu"
        if gpu_index < 0:
            logger.info("使用 CPU")
            return "cpu"
        return f"cuda:{gpu_index}"

    if isinstance(gpu_id, int):
        if gpu_id < 0:
            logger.info("使用 CPU")
            return "cpu"
        return f"cuda:{gpu_id}"

    logger.warning("未知的 gpu_id 类型=%s，回退到 CPU", type(gpu_id))
    return "cpu"


def load_model_and_tokenizer(config: Dict[str, Any], adapter_path: str, logger: logging.Logger):
    device_map = parse_gpu_config(config["gpu_id"], logger)

    compute_dtype = torch.float32
    if config["use_bf16"]:
        compute_dtype = torch.bfloat16
    elif config["use_fp16"]:
        compute_dtype = torch.float16

    logger.info("加载基座模型: %s", config["base_model_path"])
    logger.info("加载 LoRA adapter: %s", adapter_path)
    logger.info("推理精度: %s", compute_dtype)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_path"],
        trust_remote_code=True,
        torch_dtype=compute_dtype if compute_dtype != torch.float32 else None,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    logger: logging.Logger,
) -> List[str]:
    outputs: List[str] = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 多卡 device_map="auto" 时 model.device 可能不准确，取第一个参数所在的实际设备
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = torch.device("cpu")

    for batch_index, start in enumerate(range(0, len(prompts), batch_size), 1):
        batch_prompts = prompts[start:start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        prompt_length = encoded["input_ids"].shape[1]
        generated_only = generated[:, prompt_length:]

        for token_ids in generated_only:
            outputs.append(tokenizer.decode(token_ids, skip_special_tokens=True))

        logger.info("Batch %s/%s 完成: %s/%s", batch_index, total_batches, len(outputs), len(prompts))

    tokenizer.padding_side = "right"
    return outputs


def build_output_dir(config: Dict[str, Any], adapter_path: str) -> str:
    """根据 adapter 路径构建输出目录，目录名为 adapter 路径的最后一级"""
    adapter_name = os.path.basename(adapter_path.rstrip("/"))
    return os.path.join(config["output_root"], adapter_name)


def evaluate_single_adapter(config: Dict[str, Any], adapter_path: str) -> Dict[str, Any]:
    """评测单个 LoRA adapter"""
    output_dir = build_output_dir(config, adapter_path)
    logger = setup_logging(output_dir)
    
    adapter_name = os.path.basename(adapter_path.rstrip("/"))
    logger.info("=" * 90)
    logger.info("开始评测 LoRA adapter: %s", adapter_name)
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 90)

    # 保存配置
    eval_config = dict(config)
    eval_config["current_adapter_path"] = adapter_path
    save_json(os.path.join(output_dir, "config.json"), eval_config)

    # 加载数据
    logger.info("加载数据: %s", config["data_path"])
    samples = load_data(config["data_path"])
    ner_samples = [sample for sample in samples if is_ner_sample(sample)]
    if config["limit"] and config["limit"] > 0:
        ner_samples = ner_samples[:config["limit"]]

    logger.info("原始样本数: %s", len(samples))
    logger.info("NER 样本数: %s", len(ner_samples))
    if not ner_samples:
        raise ValueError("没有筛到 NER 样本，请检查 data_path 或 task_type 字段")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(config, adapter_path, logger)

    # 构建prompts
    prompts: List[str] = []
    references: List[str] = []
    for sample in ner_samples:
        messages = [{"role": "user", "content": build_user_content(sample)}]
        prompts.append(apply_chat_template_compat(tokenizer, messages, config["enable_thinking"]))
        references.append((sample.get("output") or "").strip())

    # 生成预测
    logger.info("开始生成预测...")
    start_time = time.time()
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=config["batch_size"],
        max_new_tokens=config["max_new_tokens"],
        do_sample=config["do_sample"],
        temperature=config["temperature"],
        logger=logger,
    )
    elapsed = time.time() - start_time
    logger.info("生成完成: %s 条, 用时 %.1fs, 平均 %.3fs/条", len(predictions), elapsed, elapsed / max(len(predictions), 1))

    # 计算指标
    logger.info("计算 NER 指标...")
    field_metrics = compute_field_level_ner_metrics(predictions, references, logger)
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "adapter_name": adapter_name,
        "adapter_path": adapter_path,
        "data_path": config["data_path"],
        "total_samples": len(samples),
        "ner_samples": len(ner_samples),
        "avg": field_metrics["AVERAGE"],
        "field_metrics": field_metrics,
    }

    # 输出结果
    logger.info("=" * 90)
    logger.info("NER 评测结果 - %s", adapter_name)
    logger.info("=" * 90)
    logger.info(
        "Avg Precision=%.4f | Recall=%.4f | F1=%.4f",
        metrics["avg"]["precision"],
        metrics["avg"]["recall"],
        metrics["avg"]["f1"],
    )
    logger.info(f"{'字段':<25s} {'Precision':>12s} {'Recall':>12s} {'F1':>12s}")
    logger.info("-" * 65)
    for field in NER_FIELDS:
        score = field_metrics[field]
        logger.info("%-25s %12.4f %12.4f %12.4f", field, score["precision"], score["recall"], score["f1"])
    logger.info("=" * 90)

    # 显示示例
    if config["show_examples"] and config["show_examples"] > 0:
        example_count = min(config["show_examples"], len(ner_samples))
        logger.info("示例输出（前 %s 条）", example_count)
        for index in range(example_count):
            logger.info("-" * 90)
            logger.info("#%s", index)
            logger.info("PROMPT(head): %s ...", prompts[index][:200].replace("\n", " "))
            logger.info("PRED: %s", predictions[index][:500])
            logger.info("REF : %s", references[index][:500])

    # 保存结果
    save_json(os.path.join(output_dir, "eval_metrics.json"), metrics)
    logger.info("已保存指标: %s", os.path.join(output_dir, "eval_metrics.json"))

    if config["save_predictions"]:
        rows: List[Dict[str, Any]] = []
        for sample, pred, ref in zip(ner_samples, predictions, references):
            row = dict(sample)
            row["_prediction"] = pred
            row["_pred_parsed"] = parse_ner_output(pred)
            row["_ref_parsed"] = parse_ner_output(ref)
            rows.append(row)
        save_jsonl(os.path.join(output_dir, "predictions.jsonl"), rows)
        logger.info("已保存预测: %s", os.path.join(output_dir, "predictions.jsonl"))

    logger.info("评测完成: %s\n", adapter_name)
    
    # 释放GPU内存
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return metrics


def main() -> None:
    adapter_paths = CONFIG.get("adapter_paths", [])
    if not adapter_paths:
        raise ValueError("请在 CONFIG 中配置 adapter_paths 列表")
    
    print(f"\n{'=' * 90}")
    print(f"批量 NER LoRA 评测")
    print(f"待评测 adapter 数量: {len(adapter_paths)}")
    print(f"{'=' * 90}\n")
    
    all_metrics: List[Dict[str, Any]] = []
    
    for idx, adapter_path in enumerate(adapter_paths, 1):
        print(f"\n>>> [{idx}/{len(adapter_paths)}] 正在评测: {adapter_path}")
        try:
            metrics = evaluate_single_adapter(CONFIG, adapter_path)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"评测失败: {adapter_path}")
            print(f"错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 输出汇总
    print(f"\n{'=' * 90}")
    print("评测汇总")
    print(f"{'=' * 90}")
    print(f"{'Adapter':<50s} {'Precision':>12s} {'Recall':>12s} {'F1':>12s}")
    print("-" * 90)
    for metrics in all_metrics:
        print(f"{metrics['adapter_name']:<50s} {metrics['avg']['precision']:>12.4f} {metrics['avg']['recall']:>12.4f} {metrics['avg']['f1']:>12.4f}")
    print(f"{'=' * 90}\n")
    
    print(f"全部评测完成! 共评测 {len(all_metrics)}/{len(adapter_paths)} 个 adapter")


if __name__ == "__main__":
    main()