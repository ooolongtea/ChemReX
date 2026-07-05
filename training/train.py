#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LoRA/QLoRA training script for the MAReX extraction model.

The script trains Qwen3-8B with a weighted multi-task objective:
reaction-role extraction (NER) + action-sequence generation.
All default settings are kept in CONFIG for reproducible experiments.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


CONFIG = {
    "model_path": "./Qwen3-8B",
    "data_dir": "./data/seed-67",
    "output_dir": None,
    "action_weight": 0.1,
    "ner_weight": 0.9,
    "max_length": 1024,
    "max_new_tokens": 512,
    "learning_rate": 2e-5,
    "num_epochs": 2,
    "per_device_batch_size": 8,
    "grad_accum_steps": 4,
    "warmup_steps": 400,
    "seed": 67,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "gpu_ids": "6",
    "use_fp16": False,
    "use_bf16": True,
    "logging_steps": 10,
    "save_steps": 2000,
    "eval_steps": 2000,
    "save_config": True,
    "enable_thinking": False,
    "debug": False,
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
    logger = logging.getLogger("marex_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"), encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_json_data(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def identify_task_type(sample: Dict[str, Any]) -> str:
    value = sample.get("task_type", "ner")
    value = str(value).strip().lower()
    if value in {"action", "1"}:
        return "action"
    return "ner"


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def parse_json_output(text: str) -> Any:
    if not text:
        return None
    text = text.strip()

    lower_text = text.lower()
    if "```json" in lower_text:
        start = lower_text.find("```json") + len("```json")
        end = text.find("```", start)
        if end != -1:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_ner_output(text: str) -> Dict[str, str]:
    parsed = parse_json_output(text)
    if not isinstance(parsed, dict):
        return {}
    normalized = {}
    for key, value in parsed.items():
        field = FIELD_NAME_MAPPING.get(str(key).lower(), str(key).upper())
        normalized[field] = "" if value is None else str(value)
    return normalized


def parse_action_output(text: str) -> List[str]:
    parsed = parse_json_output(text)
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    if not text:
        return []
    text = text.strip().replace("；", ";").replace("，", ",")
    delimiter = "," if "," in text else ";"
    return [item.strip() for item in text.split(delimiter) if item.strip()]


def normalize_multi_value(value: str) -> set:
    if not value:
        return set()
    value = str(value).replace("；", ";").replace("，", ",")
    items = []
    for chunk in value.split(";"):
        items.extend(chunk.split(","))
    return {item.strip() for item in items if item.strip()}


def compute_field_level_ner_metrics(preds: List[str], refs: List[str]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for field in NER_FIELDS:
        true_positive = false_positive = false_negative = 0
        for pred, ref in zip(preds, refs):
            pred_items = normalize_multi_value(parse_ner_output(pred).get(field, ""))
            ref_items = normalize_multi_value(parse_ner_output(ref).get(field, ""))
            true_positive += len(pred_items & ref_items)
            false_positive += len(pred_items - ref_items)
            false_negative += len(ref_items - pred_items)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[field] = {"precision": precision, "recall": recall, "f1": f1}

    metrics["AVERAGE"] = {
        "precision": sum(metrics[field]["precision"] for field in NER_FIELDS) / len(NER_FIELDS),
        "recall": sum(metrics[field]["recall"] for field in NER_FIELDS) / len(NER_FIELDS),
        "f1": sum(metrics[field]["f1"] for field in NER_FIELDS) / len(NER_FIELDS),
    }
    return metrics


def compute_action_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    bleu_scores, exact_matches, pred_texts, ref_texts = [], [], [], []
    smoother = SmoothingFunction().method1

    for pred, ref in zip(preds, refs):
        pred_actions = parse_action_output(pred)
        ref_actions = parse_action_output(ref)
        if not ref_actions:
            continue

        try:
            bleu_scores.append(
                sentence_bleu(
                    [ref_actions],
                    pred_actions,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoother,
                )
            )
        except Exception:
            bleu_scores.append(0.0)

        exact_matches.append(float(pred_actions == ref_actions))
        pred_texts.append(" ".join(pred_actions))
        ref_texts.append(" ".join(ref_actions))

    bertscore_f1 = 0.0
    if pred_texts and ref_texts:
        try:
            _, _, f1 = bert_score(pred_texts, ref_texts, lang="zh", verbose=False)
            bertscore_f1 = f1.mean().item()
        except Exception:
            bertscore_f1 = 0.0

    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {"bleu_4": mean(bleu_scores), "em": mean(exact_matches), "bertscore_f1": bertscore_f1}


class MultiTaskDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length: int, enable_thinking: bool = False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        for sample in self.samples:
            sample["task_type"] = identify_task_type(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        user_content = sample.get("instruction", "").strip()
        assistant_content = sample.get("output", "").strip()

        full_text = apply_chat_template(
            self.tokenizer,
            [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}],
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        prompt_text = apply_chat_template(
            self.tokenizer,
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        full_tokens = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False)
        prompt_tokens = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length, padding=False)

        labels = full_tokens["input_ids"].copy()
        prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
            "eval_input_ids": prompt_tokens["input_ids"],
            "eval_attention_mask": prompt_tokens["attention_mask"],
            "task_type_id": 0 if sample["task_type"] == "ner" else 1,
            "reference": assistant_content,
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        self.tokenizer.padding_side = "right"
        max_train_len = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            pad_len = max_train_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        self.tokenizer.padding_side = "left"
        max_eval_len = max(len(feature["eval_input_ids"]) for feature in features)
        eval_input_ids, eval_attention_mask = [], []
        for feature in features:
            pad_len = max_eval_len - len(feature["eval_input_ids"])
            eval_input_ids.append([self.tokenizer.pad_token_id] * pad_len + feature["eval_input_ids"])
            eval_attention_mask.append([0] * pad_len + feature["eval_attention_mask"])

        self.tokenizer.padding_side = "right"
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eval_input_ids": torch.tensor(eval_input_ids, dtype=torch.long),
            "eval_attention_mask": torch.tensor(eval_attention_mask, dtype=torch.long),
            "task_types": [feature["task_type_id"] for feature in features],
            "references": [feature["reference"] for feature in features],
        }


class WeightedTrainer(Trainer):
    def __init__(self, task_weights, max_new_tokens: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights
        self.max_new_tokens = max_new_tokens
        if getattr(self, "tokenizer", None) is None and getattr(self, "processing_class", None) is not None:
            self.tokenizer = self.processing_class
        if getattr(self, "tokenizer", None) is not None:
            self.processing_class = self.tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = inputs.copy()
        task_types = inputs.pop("task_types", None)
        inputs.pop("references", None)
        inputs.pop("eval_input_ids", None)
        inputs.pop("eval_attention_mask", None)

        outputs = model(**inputs)
        labels = inputs["labels"]
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        token_losses = nn.CrossEntropyLoss(reduction="none")(shift_logits, shift_labels)
        valid_mask = (shift_labels != -100).float()

        if task_types and self.task_weights:
            batch_size = len(task_types)
            seq_len = shift_labels.size(0) // batch_size
            weights = torch.ones_like(token_losses)
            for i, task_id in enumerate(task_types):
                task_name = "ner" if task_id == 0 else "action"
                weights[i * seq_len : (i + 1) * seq_len] = self.task_weights.get(task_name, 1.0)
            loss = (token_losses * weights * valid_mask).sum() / valid_mask.sum()
        else:
            loss = (token_losses * valid_mask).sum() / valid_mask.sum()

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs) if "labels" in inputs else None
            if prediction_loss_only:
                return loss, None, None

            tokenizer = self.tokenizer or getattr(self, "processing_class", None)
            if tokenizer is None:
                raise RuntimeError("Tokenizer is required for generation-based evaluation.")
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

            try:
                generated_ids = model.generate(
                    input_ids=inputs["eval_input_ids"],
                    attention_mask=inputs["eval_attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated_ids = generated_ids[:, inputs["eval_input_ids"].shape[1] :]
            except Exception:
                batch_size = inputs["eval_input_ids"].shape[0]
                generated_ids = torch.full((batch_size, 1), pad_token_id, dtype=torch.long)

            return loss, generated_ids, inputs.get("labels")


class LoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        self.best_f1 = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.logger.info(
                "Step %5d | Epoch %.2f | Loss %.4f | LR %.2e",
                state.global_step,
                logs.get("epoch", 0),
                logs["loss"],
                logs.get("learning_rate", 0),
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        self.logger.info("Evaluation at step %s", state.global_step)
        if "eval_ner_avg_f1" in metrics:
            f1 = metrics["eval_ner_avg_f1"]
            self.logger.info(
                "NER: F1 %.4f | P %.4f | R %.4f",
                f1,
                metrics.get("eval_ner_avg_precision", 0),
                metrics.get("eval_ner_avg_recall", 0),
            )
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.logger.info("New best NER F1: %.4f", f1)
        if "eval_action_bleu_4" in metrics:
            self.logger.info(
                "Action: BLEU-4 %.4f | EM %.4f | BERTScore %.4f",
                metrics["eval_action_bleu_4"],
                metrics.get("eval_action_em", 0),
                metrics.get("eval_action_bertscore_f1", 0),
            )


def build_compute_metrics(tokenizer, eval_dataset):
    def compute_metrics(eval_preds: EvalPrediction):
        predictions = []
        for pred_ids in eval_preds.predictions:
            try:
                pred_ids = pred_ids.tolist() if hasattr(pred_ids, "tolist") else pred_ids
                valid_ids = [int(token_id) for token_id in pred_ids if 0 <= int(token_id) < tokenizer.vocab_size]
                predictions.append(tokenizer.decode(valid_ids, skip_special_tokens=True))
            except Exception:
                predictions.append("")

        samples = eval_dataset.samples[: len(predictions)]
        grouped = defaultdict(lambda: {"predictions": [], "references": []})
        for pred, sample in zip(predictions, samples):
            task_type = sample.get("task_type", "ner")
            grouped[task_type]["predictions"].append(pred)
            grouped[task_type]["references"].append(sample.get("output", ""))

        metrics = {}
        if "ner" in grouped:
            field_metrics = compute_field_level_ner_metrics(grouped["ner"]["predictions"], grouped["ner"]["references"])
            average = field_metrics["AVERAGE"]
            metrics.update(
                {
                    "ner_avg_precision": average["precision"],
                    "ner_avg_recall": average["recall"],
                    "ner_avg_f1": average["f1"],
                }
            )
            for field in NER_FIELDS:
                metrics[f"ner_{field.lower()}_f1"] = field_metrics[field]["f1"]

        if "action" in grouped:
            action_metrics = compute_action_metrics(grouped["action"]["predictions"], grouped["action"]["references"])
            metrics.update(
                {
                    "action_bleu_4": action_metrics["bleu_4"],
                    "action_em": action_metrics["em"],
                    "action_bertscore_f1": action_metrics["bertscore_f1"],
                }
            )
        return metrics

    return compute_metrics


def build_output_dir(cfg: Dict[str, Any]) -> str:
    if cfg["output_dir"] is not None:
        return cfg["output_dir"]
    model_path = Path(cfg["model_path"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(model_path.parent / f"{model_path.name}-LoRA-{timestamp}")


def load_model_and_tokenizer(cfg: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    compute_dtype = torch.bfloat16 if cfg["use_bf16"] else torch.float16 if cfg["use_fp16"] else torch.float32
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, lora_config), tokenizer


def main():
    cfg = CONFIG
    if cfg["gpu_ids"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu_ids"]
    torch.manual_seed(cfg["seed"])

    output_dir = build_output_dir(cfg)
    logger = setup_logging(output_dir)
    logger.info("Training started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Output directory: %s", output_dir)

    if cfg.get("save_config", True):
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    data_dir = Path(cfg["data_dir"])
    train_samples = load_json_data(str(data_dir / "train.json"))
    valid_samples = load_json_data(str(data_dir / "valid.json"))
    logger.info("Data: train=%d, valid=%d", len(train_samples), len(valid_samples))

    logger.info("Loading model: %s", cfg["model_path"])
    model, tokenizer = load_model_and_tokenizer(cfg)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    logger.info("Trainable parameters: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    train_dataset = MultiTaskDataset(train_samples, tokenizer, cfg["max_length"], cfg["enable_thinking"])
    valid_dataset = (
        MultiTaskDataset(valid_samples, tokenizer, cfg["max_length"], cfg["enable_thinking"]) if valid_samples else None
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"] if valid_dataset else None,
        eval_strategy="steps" if valid_dataset else "no",
        save_strategy="steps" if valid_dataset else "no",
        save_total_limit=1,
        fp16=cfg["use_fp16"],
        bf16=cfg["use_bf16"],
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=cfg["seed"],
        load_best_model_at_end=bool(valid_dataset),
        metric_for_best_model="eval_ner_avg_f1" if valid_dataset else None,
        greater_is_better=True,
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        task_weights={"ner": cfg["ner_weight"], "action": cfg["action_weight"]},
        max_new_tokens=cfg["max_new_tokens"],
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollator(tokenizer),
        compute_metrics=build_compute_metrics(tokenizer, valid_dataset) if valid_dataset else None,
        processing_class=tokenizer,
        callbacks=[LoggingCallback(logger)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if valid_dataset:
        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)

    logger.info("Training finished: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
