#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate.py
多任务模型离线评测脚本 (NER + Action)

支持三种模型类型：
1. base: 未微调的基座模型
2. lora: LoRA 微调后的模型 (需要指定 adapter 和 base model)
3. external: 外部 API 模型 (OpenAI 兼容接口)

评测指标：
- NER: 字段级别 Precision/Recall/F1 + Average
- Action: BLEU-4 / EM (Exact Match) / BERTScore-F1

输出：
- experiments/<experiment_name>/
  ├── config.json           # 评测配置
  ├── eval_metrics.json     # 详细指标
  ├── predictions.jsonl     # 逐样本预测（可选）
  └── evaluate.log          # 日志文件

依赖：
pip install transformers torch scikit-learn
评估指标库：
pip install nltk bert-score
外部 API：
pip install openai python-dotenv
"""

import os
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =============== 权威评估逻辑统一引用 eval.py ===============
import eval as eval_module

# =============== 进度条 ===============
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ 警告: tqdm 未安装，将使用日志打印进度。安装: pip install tqdm")

# =============== 可选指标库 ===============
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from bert_score import score as bert_score
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("⚠️ 警告: 评估指标库未安装，Action 指标将为 0")
    print("   安装: pip install nltk bert-score")

# =============== NER 字段定义统一从 eval.py 引用 ===============
NER_FIELDS = eval_module.NER_FIELDS
FIELD_NAME_MAPPING = eval_module.FIELD_NAME_MAPPING

EXTERNAL_MODEL_DEFAULTS = {
    "deepseek-v4-pro": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}


def get_external_model_defaults(model_name: str) -> Dict[str, Any]:
    return EXTERNAL_MODEL_DEFAULTS.get(model_name.lower(), {})

# =========================
# Logging
# =========================
def setup_logging(output_dir: str, log_file: str = "evaluate.log") -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file)

    logger = logging.getLogger("Evaluator")
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
# Task type 识别
# =========================
def identify_task_type(sample: Dict[str, Any]) -> str:
    """
    识别任务类型
    
    支持多种格式:
    - task_type: "NER" / "Action" (字符串，新格式)
    - task_type: 0 / 1 (整数，旧格式)
    - task_type: "ner" / "action" (小写字符串)
    
    Returns:
        "ner" 或 "action" (小写)
    """
    if "task_type" in sample:
        v = sample["task_type"]
        if isinstance(v, int):
            if v == 0:
                return "ner"
            if v == 1:
                return "action"
        elif isinstance(v, str):
            vv = v.lower()
            if vv in ["ner", "0"]:
                return "ner"
            if vv in ["action", "1"]:
                return "action"
    return "ner"


# =========================
# Prompt 构建
# =========================
def build_user_content(sample: Dict[str, Any]) -> str:
    """
    构建用户输入内容
    
    新格式（generate_dataset.py 生成）:
    - instruction 字段已包含完整提示（模板中的 {{TEXT}} 已被替换）
    - 直接使用 instruction 字段
    
    旧格式（兼容）:
    - instruction 和 input 分离
    - 需要拼接两者
    
    Args:
        sample: 数据样本
    
    Returns:
        完整的用户输入内容
    """
    instruction = (sample.get("instruction") or "").strip()
    user_input = (sample.get("input") or "").strip()
    
    # 判断是否为新格式：instruction 中包含 "###实验步骤：" 或 "###Procedure:"
    # 说明已经拼接过了，直接使用 instruction
    if instruction and ("###实验步骤：" in instruction or "###Procedure:" in instruction):
        return instruction
    
    # 旧格式：需要拼接
    if instruction and user_input:
        return f"{instruction}\n\n{user_input}"
    
    return instruction or user_input


def apply_chat_template_compat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool, enable_thinking: bool = False) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# =========================
# 评估逻辑统一委托给 eval.py
# =========================
parse_json_output = eval_module.parse_json_output
parse_ner_output = eval_module.parse_ner_output
parse_action_output = eval_module.parse_action_output
compute_field_level_ner_metrics = eval_module.compute_field_level_ner_metrics
compute_action_metrics = eval_module.compute_action_metrics


# =========================
# 本地模型推理 (HF)
# =========================
@torch.no_grad()
def generate_with_hf_model(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    batch_size: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
    logger = None,
) -> List[str]:
    tokenizer.padding_side = "left"
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    if logger:
        logger.info(f"总共有 {len(prompts)} 条样本，batch_size={batch_size}，共 {total_batches} 个 batch")

    outputs: List[str] = []
    batch_iter = range(0, len(prompts), batch_size)
    if TQDM_AVAILABLE:
        batch_iter = tqdm(batch_iter, total=total_batches, desc="本地推理", unit="batch")
    for batch_idx, i in enumerate(batch_iter, 1):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 截取生成部分
        input_len = enc["input_ids"].shape[1]
        gen_only = gen_ids[:, input_len:]

        # 解码
        vocab_size = tokenizer.vocab_size
        for row in gen_only:
            row = row.tolist()
            valid_ids = [int(t) for t in row if 0 <= int(t) < vocab_size]
            text = tokenizer.decode(valid_ids, skip_special_tokens=True)
            outputs.append(text)

        # 进度条已显示进度；同时保留日志记录
        if logger and (batch_idx % 10 == 0 or batch_idx == total_batches):
            current_count = len(outputs)
            progress_pct = (current_count / len(prompts)) * 100
            logger.info(f"Batch {batch_idx}/{total_batches} 完成: {current_count}/{len(prompts)} ({progress_pct:.1f}%)")

    tokenizer.padding_side = "right"
    return outputs


# =========================
# 外部 API 推理
# =========================
def generate_with_external_api(
    samples: List[Dict[str, Any]],
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    enable_thinking: bool = False,
    timeout: int = 300,
    max_workers: int = 32,
    logger = None,
) -> List[str]:
    """使用 OpenAI 兼容 API 进行推理（并发）
    
    直接从样本构建 messages，不需要 tokenizer。
    使用 ThreadPoolExecutor 并发发送请求，max_workers 控制并发数。
    """
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        import os as os_module
    except ImportError:
        raise ImportError("需要安装: pip install openai python-dotenv")
    
    # 加载环境变量
    load_dotenv()
    
    # 获取 API 配置
    model_defaults = get_external_model_defaults(model_name)
    if api_key is None:
        api_key_env = model_defaults.get("api_key_env")
        if api_key_env:
            api_key = os_module.getenv(api_key_env)
        if api_key is None:
            api_key = os_module.getenv('API_KEY')
    if base_url is None:
        base_url = os_module.getenv('BASE_URL') or model_defaults.get("base_url")
    
    if not api_key:
        raise ValueError("需要提供 API Key（deepseek-v4-pro 可通过 DEEPSEEK_API_KEY，其它模型可通过 API_KEY）")
    
    # 初始化客户端
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    else:
        client = OpenAI(api_key=api_key, timeout=timeout)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def _call_one(idx: int) -> tuple[int, str]:
        sample = samples[idx]
        try:
            user_content = build_user_content(sample)
            messages = [{"role": "user", "content": user_content}]
            extra_body = model_defaults.get("extra_body")
            if extra_body is None:
                extra_body = {
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                    }
                }

            request_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "extra_body": extra_body,
            }
            response = client.chat.completions.create(**request_kwargs)
            msg = response.choices[0].message
            answer = msg.content or getattr(msg, "reasoning_content", "") or ""
            return idx, answer
        except Exception as e:
            if logger:
                logger.warning(f"API 调用失败 (样本 {idx}): {e}")
            else:
                print(f"Warning: API 调用失败 (样本 {idx}): {e}")
            return idx, ""
    
    outputs: List[str] = [""] * len(samples)
    completed = 0
    total = len(samples)
    
    if logger:
        logger.info(f"并发调用 API: max_workers={max_workers}, 样本数={total}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_one, i): i for i in range(total)}
        fut_iter = as_completed(futures)
        if TQDM_AVAILABLE:
            fut_iter = tqdm(fut_iter, total=total, desc=f"API推理[{model_name}]", unit="样本")
        for fut in fut_iter:
            idx, answer = fut.result()
            outputs[idx] = answer
            completed += 1
            if logger and completed % 50 == 0:
                logger.info(f"API 调用进度: {completed}/{total} ({completed/total*100:.1f}%)")
    
    if logger:
        logger.info(f"API 调用完成: {completed}/{total}")
    
    return outputs


# =========================
# 加载模型
# =========================
def parse_gpu_config(gpu_id_arg, logger):
    """
    解析 GPU 配置，支持多种格式：
    - None 或 -1: CPU
    - "auto": 自动分配
    - 单个数字 (0, 1, 2): 指定单张卡
    - 逗号分隔 ("2,3" 或 "2, 3"): 多张卡（设置环境变量 + auto）
    
    Returns:
        device_map: "cpu", "auto", 或 "cuda:X"
    """
    if gpu_id_arg is None or gpu_id_arg == -1:
        logger.info("使用 CPU")
        return "cpu"
    
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，使用 CPU")
        return "cpu"
    
    # 处理字符串类型的 gpu_id
    if isinstance(gpu_id_arg, str):
        gpu_id_arg = gpu_id_arg.strip()
        
        # auto 模式
        if gpu_id_arg.lower() == "auto":
            logger.info("使用 device_map='auto' 自动分配 GPU")
            return "auto"
        
        # 多卡模式：逗号分隔
        if "," in gpu_id_arg:
            # 去除空格并验证格式
            gpu_ids = [g.strip() for g in gpu_id_arg.split(",")]
            try:
                gpu_ids_int = [int(g) for g in gpu_ids]
                
                # 验证 GPU 是否存在
                available_gpus = torch.cuda.device_count()
                invalid_gpus = [g for g in gpu_ids_int if g >= available_gpus]
                if invalid_gpus:
                    logger.warning(f"以下 GPU 不存在: {invalid_gpus}，可用 GPU: 0-{available_gpus-1}")
                
                # 设置环境变量限制可见 GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_arg
                logger.info(f"多卡模式: 设置 CUDA_VISIBLE_DEVICES={gpu_id_arg}")
                logger.info(f"使用 device_map='auto' 在指定的 GPU 上自动分配")
                return "auto"
                
            except ValueError:
                logger.warning(f"无效的 GPU ID 格式: {gpu_id_arg}，使用 CPU")
                return "cpu"
        
        # 单卡模式：单个数字字符串
        try:
            gpu_id_int = int(gpu_id_arg)
            if gpu_id_int < 0:
                logger.info("GPU ID < 0，使用 CPU")
                return "cpu"
            
            available_gpus = torch.cuda.device_count()
            if gpu_id_int >= available_gpus:
                logger.warning(f"GPU {gpu_id_int} 不存在（可用: 0-{available_gpus-1}），使用 GPU 0")
                return "cuda:0"
            
            logger.info(f"使用单张 GPU: cuda:{gpu_id_int}")
            return f"cuda:{gpu_id_int}"
            
        except ValueError:
            logger.warning(f"无效的 GPU ID: {gpu_id_arg}，使用 CPU")
            return "cpu"
    
    # 处理整数类型的 gpu_id
    if isinstance(gpu_id_arg, int):
        if gpu_id_arg < 0:
            logger.info("GPU ID < 0，使用 CPU")
            return "cpu"
        
        available_gpus = torch.cuda.device_count()
        if gpu_id_arg >= available_gpus:
            logger.warning(f"GPU {gpu_id_arg} 不存在（可用: 0-{available_gpus-1}），使用 GPU 0")
            return "cuda:0"
        
        logger.info(f"使用单张 GPU: cuda:{gpu_id_arg}")
        return f"cuda:{gpu_id_arg}"
    
    # 默认 CPU
    logger.warning(f"未知的 GPU 配置类型: {type(gpu_id_arg)}，使用 CPU")
    return "cpu"


def load_model_and_tokenizer(args, logger):
    """根据模型类型加载对应的模型和 tokenizer"""
    
    if args.model_type == "external":
        # 外部 API 不需要本地模型和 tokenizer
        logger.info(f"使用外部 API: {args.external_model}")
        return None, None
    
    # 确定设备（使用新的解析函数）
    device_map = parse_gpu_config(args.gpu_id, logger)
    
    # 本地模型（base 或 lora）
    compute_dtype = torch.bfloat16 if args.use_bf16 else torch.float16 if args.use_fp16 else torch.float32
    
    if args.model_type == "base":
        # 基座模型
        if not args.base_model_path:
            raise ValueError("--model_type=base 需要提供 --base_model_path")
        
        logger.info(f"加载基座模型: {args.base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
        
        if args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map=device_map,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                trust_remote_code=True,
                torch_dtype=compute_dtype if compute_dtype != torch.float32 else None,
                device_map=device_map,
            )
    
    elif args.model_type == "lora":
        # LoRA 模型
        if not args.base_model_path or not args.adapter_path:
            raise ValueError("--model_type=lora 需要同时提供 --base_model_path 和 --adapter_path")
        
        logger.info(f"加载基座模型: {args.base_model_path}")
        logger.info(f"加载 LoRA adapter: {args.adapter_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
        
        if args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map=device_map,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                trust_remote_code=True,
                torch_dtype=compute_dtype if compute_dtype != torch.float32 else None,
                device_map=device_map,
            )
        
        # 加载并合并 adapter
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    
    else:
        raise ValueError(f"未知的模型类型: {args.model_type}")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    return model, tokenizer


# =========================
# 主评测流程
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="多任务模型评测 (NER + Action)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 评测基座模型（单卡）:
   python evaluate.py --model_type base \\
       --base_model_path /path/to/base_model \\
       --gpu_id 0
   # 自动创建目录: experiments/base_<模型名>_<时间戳>/

2. 评测基座模型（多卡）:
   python evaluate.py --model_type base \\
       --base_model_path /path/to/base_model \\
       --gpu_id "2,3"
   # 多卡会自动使用 device_map='auto' 分配模型层

3. 评测 LoRA 微调模型:
   python evaluate.py --model_type lora \\
       --base_model_path /path/to/base_model \\
       --adapter_path /path/to/lora_adapter \\
       --gpu_id auto
   # 自动创建目录: experiments/lora_<adapter名>_<时间戳>/

4. 评测外部 API 模型:
   python evaluate.py --model_type external \\
       --external_model deepseek-v4-pro
   # 自动创建目录: experiments/external_deepseek-v4-pro_<时间戳>/
        """
    )
    
    # ========== 核心参数 ==========
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=["base", "lora", "external"],
                       help="模型类型: base(基座), lora(LoRA微调), external(外部API)")
    
    parser.add_argument("--data", type=str, default=ARGS_CONFIG["data"],
                       help=f"评测数据文件（支持 .json 或 .jsonl 格式，默认: {ARGS_CONFIG['data']}）")
    
    # ========== 模型路径参数 ==========
    parser.add_argument("--base_model_path", type=str, default=ARGS_CONFIG["base_model_path"],
                       help="基座模型路径（base/lora 类型需要）")
    
    parser.add_argument("--adapter_path", type=str, default=ARGS_CONFIG["adapter_path"],
                       help="LoRA adapter 路径（lora 类型需要）")
    
    # ========== 外部 API 参数 ==========
    parser.add_argument("--external_model", type=str, default=ARGS_CONFIG["external_model"],
                       help=f"外部 API 模型名称（默认: {ARGS_CONFIG['external_model']}）")
    
    parser.add_argument("--api_key", type=str, default=ARGS_CONFIG["api_key"],
                       help="外部 API Key（deepseek-v4-pro 默认从 DEEPSEEK_API_KEY 读取，否则从 API_KEY 读取）")
    
    parser.add_argument("--base_url", type=str, default=ARGS_CONFIG["base_url"],
                       help="外部 API Base URL（deepseek-v4-pro 默认 https://api.deepseek.com，否则从 BASE_URL 读取）")
    
    parser.add_argument("--api_timeout", type=int, default=ARGS_CONFIG["api_timeout"],
                       help=f"外部 API 超时时间（秒，默认: {ARGS_CONFIG['api_timeout']}）")
    
    # ========== 推理参数 ==========
    parser.add_argument("--max_new_tokens", type=int, 
                       default=ARGS_CONFIG["max_new_tokens"],
                       help=f"生成最大新 token（默认: {ARGS_CONFIG['max_new_tokens']}）")
    
    parser.add_argument("--batch_size", type=int,
                       default=ARGS_CONFIG["batch_size"],
                       help=f"推理 batch size（默认: {ARGS_CONFIG['batch_size']}，仅本地模型生效）")
    
    parser.add_argument("--max_workers", type=int,
                       default=ARGS_CONFIG["max_workers"],
                       help=f"外部 API 并发数（默认: {ARGS_CONFIG['max_workers']}）")
    
    parser.add_argument("--temperature", type=float,
                       default=ARGS_CONFIG["temperature"],
                       help=f"采样温度（默认: {ARGS_CONFIG['temperature']}）")
    
    parser.add_argument("--do_sample", action="store_true",
                       help="开启采样（默认关闭，使用贪婪解码）")
    
    # ========== 量化和精度 ==========
    parser.add_argument("--use_4bit", action="store_true",
                       help="使用 4bit 量化加载（仅本地模型）")
    
    parser.add_argument("--use_fp16", action="store_true",
                       help="使用 FP16 推理（仅本地模型）")
    
    parser.add_argument("--use_bf16", action="store_true",
                       help="使用 BF16 推理（仅本地模型）")
    
    # ========== GPU 配置 ==========
    gpu_default = ARGS_CONFIG["gpu_id"]
    if gpu_default is None:
        gpu_default_str = "-1"
    elif isinstance(gpu_default, str):
        gpu_default_str = gpu_default
    else:
        gpu_default_str = str(gpu_default)
    
    parser.add_argument("--gpu_id", type=str, default=gpu_default_str,
                       help=f"指定 GPU ID（默认: {ARGS_CONFIG['gpu_id']}。支持格式: -1=CPU, auto=自动, 0=单卡, '2,3'=多卡）")
    
    # ========== 其他参数 ==========
    parser.add_argument("--enable_thinking", action="store_true",
                       help="传给 apply_chat_template 的 enable_thinking")
    
    parser.add_argument("--limit", type=int, default=ARGS_CONFIG["limit"],
                       help=f"只评测前 N 条（默认: {ARGS_CONFIG['limit']}，0 表示全量）")
    
    parser.add_argument("--save_predictions", action="store_true",
                       help="保存 predictions.jsonl（逐样本预测）")
    
    # 设置 action="store_true" 参数的默认值
    parser.set_defaults(
        do_sample=ARGS_CONFIG["do_sample"],
        use_4bit=ARGS_CONFIG["use_4bit"],
        use_fp16=ARGS_CONFIG["use_fp16"],
        use_bf16=ARGS_CONFIG["use_bf16"],
        enable_thinking=ARGS_CONFIG["enable_thinking"],
        save_predictions=ARGS_CONFIG["save_predictions"],
    )
    
    parser.add_argument("--show_examples", type=int,
                       default=ARGS_CONFIG["show_examples"],
                       help=f"打印前 N 条样本对比（默认: {ARGS_CONFIG['show_examples']}，0关闭）")
    
    args = parser.parse_args()
    
    # 处理 gpu_id 参数（保持字符串格式以支持多卡）
    def parse_gpu_id(value):
        """预处理 gpu_id 参数，保持字符串格式用于多卡支持"""
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in ["none", "null", ""]:
                return None
            if value == "-1":
                return -1
            # 保持其他格式不变：'auto', '0', '2,3' 等
            return value
        if isinstance(value, int):
            return value if value >= 0 else -1
        return None
    
    # 处理 gpu_id 参数
    args.gpu_id = parse_gpu_id(args.gpu_id)
    
    # ========== 自动生成实验目录名 ==========
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if args.model_type == "base":
        # 使用 base_model_path 的最后一部分作为模型名称
        if not args.base_model_path:
            raise ValueError("--model_type=base 需要提供 --base_model_path")
        model_name = os.path.basename(args.base_model_path.rstrip("/"))
        experiment_name = f"{args.model_type}_{model_name}_{timestamp}"
    elif args.model_type == "lora":
        # 需要基础模型和 adapter 路径
        if not args.base_model_path:
            raise ValueError("--model_type=lora 需要提供 --base_model_path")
        if not args.adapter_path:
            raise ValueError("--model_type=lora 需要提供 --adapter_path")
        base_model_name = os.path.basename(args.base_model_path.rstrip("/"))
        adapter_name = os.path.basename(args.adapter_path.rstrip("/"))
        experiment_name = f"{args.model_type}_{base_model_name}_{timestamp}"
    elif args.model_type == "external":
        # 使用 external_model 名称
        model_name = args.external_model
        experiment_name = f"{args.model_type}_{model_name}_{timestamp}"
    else:
        raise ValueError(f"未知的模型类型: {args.model_type}")
    
    # ========== 创建输出目录 ==========
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _results_dir = os.path.join(os.path.dirname(_script_dir), "results")
    output_dir = os.path.join(_results_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"开始评测: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"输出目录: {output_dir}")
    
    # 记录 GPU 配置
    if args.model_type != "external":
        logger.info(f"GPU 配置: {args.gpu_id}")
        if torch.cuda.is_available():
            logger.info(f"可用 GPU 数量: {torch.cuda.device_count()}")
            logger.info(f"CUDA 版本: {torch.version.cuda}")
        else:
            logger.info("CUDA 不可用")
    
    # ========== 保存配置 ==========
    config = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": args.model_type,
        "data": args.data,
        "model_config": {},
        "inference_config": {
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "max_workers": args.max_workers,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "enable_thinking": args.enable_thinking,
        },
        "metrics_available": METRICS_AVAILABLE,
    }
    
    if args.model_type == "base":
        config["model_config"] = {
            "base_model_path": args.base_model_path,
            "use_4bit": args.use_4bit,
            "use_fp16": args.use_fp16,
            "use_bf16": args.use_bf16,
        }
    elif args.model_type == "lora":
        config["model_config"] = {
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path,
            "use_4bit": args.use_4bit,
            "use_fp16": args.use_fp16,
            "use_bf16": args.use_bf16,
        }
    elif args.model_type == "external":
        model_defaults = get_external_model_defaults(args.external_model)
        config["model_config"] = {
            "external_model": args.external_model,
            "base_url": args.base_url or model_defaults.get("base_url"),
            "api_timeout": args.api_timeout,
        }
    
    config_path = os.path.join(output_dir, "config.json")
    save_json(config_path, config)
    logger.info(f"已保存配置: {config_path}")
    
    # ========== 加载数据 ==========
    logger.info(f"加载数据: {args.data}")
    samples = load_data(args.data)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
    logger.info(f"样本数: {len(samples)}")
    
    # 打印数据格式信息
    if samples:
        sample_keys = list(samples[0].keys())
        logger.info(f"数据字段: {', '.join(sample_keys)}")
        if "task_type" in samples[0]:
            task_counts = defaultdict(int)
            for s in samples:
                task_counts[s.get("task_type", "unknown")] += 1
            logger.info(f"任务分布: {dict(task_counts)}")
    
    # ========== 加载模型 ==========
    model, tokenizer = load_model_and_tokenizer(args, logger)
    
    # ========== 构建 prompts（仅本地模型需要）==========
    if args.model_type == "external":
        # 外部 API 直接使用原始样本
        logger.info("外部 API 模式，跳过 prompt 构建")
        prompts = None
        references = [(s.get("output") or "").strip() for s in samples]
        task_types = [identify_task_type(s) for s in samples]
    else:
        # 本地模型需要构建 prompts
        logger.info("构建 prompts...")
        prompts: List[str] = []
        references: List[str] = []
        task_types: List[str] = []
        
        for s in samples:
            user_content = build_user_content(s)
            messages_prompt = [{"role": "user", "content": user_content}]
            prompt_text = apply_chat_template_compat(
                tokenizer,
                messages_prompt,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            
            prompts.append(prompt_text)
            references.append((s.get("output") or "").strip())
            task_types.append(identify_task_type(s))
    
    # ========== 生成预测 ==========
    logger.info("开始生成预测...")
    t0 = time.time()
    
    if args.model_type == "external":
        predictions = generate_with_external_api(
            samples=samples,
            model_name=args.external_model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            timeout=args.api_timeout,
            max_workers=args.max_workers,
            logger=logger,
        )
    else:
        predictions = generate_with_hf_model(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            do_sample=args.do_sample,
            temperature=args.temperature,
            logger=logger,
        )
    
    dt = time.time() - t0
    logger.info(f"生成完成: {len(predictions)} 条, 用时 {dt:.1f}s, 平均 {dt/max(len(predictions),1):.3f}s/条")
    
    # ========== 分组并计算指标 ==========
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
    
    # NER
    if "ner" in task_data and len(task_data["ner"]["predictions"]) > 0:
        logger.info("计算 NER 指标...")
        ner_field_metrics = compute_field_level_ner_metrics(
            task_data["ner"]["predictions"],
            task_data["ner"]["references"],
            logger=logger,
        )
        metrics["ner"]["field_metrics"] = ner_field_metrics
        metrics["ner"]["avg"] = ner_field_metrics["AVERAGE"]
    
    # Action
    if "action" in task_data and len(task_data["action"]["predictions"]) > 0:
        logger.info("计算 Action 指标...")
        action_metrics = compute_action_metrics(
            task_data["action"]["predictions"],
            task_data["action"]["references"],
            logger=logger,
        )
        metrics["action"] = action_metrics
    
    # ========== 打印摘要 ==========
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
        bertscore_f1 = metrics["action"].get("bertscore_f1")
        bertscore_text = f"{bertscore_f1:.4f}" if bertscore_f1 is not None else "None"
        logger.info(f"[Action] BLEU-4={metrics['action']['bleu_4']:.4f} | EM={metrics['action']['em']:.4f} | BERTScore-F1={bertscore_text}")
    else:
        logger.info("[Action] 无 Action 样本")
    
    logger.info("=" * 90 + "\n")
    
    # ========== 示例输出 ==========
    if args.show_examples and args.show_examples > 0:
        n = min(args.show_examples, len(samples))
        logger.info(f"示例输出（前 {n} 条）:")
        for i in range(n):
            tt = task_types[i]
            logger.info("-" * 90)
            logger.info(f"#{i} [{tt}]")
            if prompts is not None:
                logger.info(f"PROMPT(head): {prompts[i][:200].replace(chr(10),' ')} ...")
            else:
                # external 模式，显示原始 input
                user_content = build_user_content(samples[i])
                logger.info(f"INPUT(head): {user_content[:200].replace(chr(10),' ')} ...")
            logger.info(f"PRED: {predictions[i][:500]}")
            logger.info(f"REF : {references[i][:500]}")
    
    # ========== 保存结果 ==========
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


# =========================
# 参数配置区域（放在最后，方便修改）
# =========================
ARGS_CONFIG = {
    # 数据参数
    "data": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "seed-67", "test.json"),
    
    # 模型路径参数
    "base_model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Model", "Qwen3-8B"),
    "adapter_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3-8B-LoRA-0.8"),
    
    # 外部 API 参数
    "external_model": "deepseek-v4-pro",
    "api_key": None,  # deepseek-v4-pro 默认从 DEEPSEEK_API_KEY 读取，其它模型从 API_KEY 读取
    "base_url": None,  # deepseek-v4-pro 默认 https://api.deepseek.com，其它模型从 BASE_URL 读取
    "api_timeout": 300,
    
    # 推理参数
    "max_new_tokens": 512,
    "batch_size": 12,
    "max_workers": 100,
    "temperature": 0.0,
    "do_sample": False,
    
    # 量化和精度参数
    "use_4bit": False,
    "use_fp16": False,
    "use_bf16": True,  
    
    # GPU 配置
    "gpu_id": "auto",  # GPU 配置：-1=CPU, 'auto'=自动, '0'=单卡, '2,3'=多卡（逗号分隔）
    
    # 其他参数
    "enable_thinking": False,
    "limit": 0,  # 0 表示全量
    "save_predictions": True,
    "show_examples": 0,  # 0 表示不显示
}



if __name__ == "__main__":
    main()
