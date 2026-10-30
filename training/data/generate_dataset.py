#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集生成脚本
根据 meta.json 配置生成 train、valid、test 数据集
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any
import argparse


class DatasetGenerator:
    def __init__(self, meta_path: str, instructions_path: str, output_base_dir: str, seed: int = 42):
        """
        初始化数据集生成器
        
        Args:
            meta_path: meta.json 文件路径
            instructions_path: instructions.md 文件路径
            output_base_dir: 输出基础目录
            seed: 随机种子
        """
        self.meta_path = meta_path
        self.instructions_path = instructions_path
        self.output_base_dir = output_base_dir
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        
        # 加载配置和提示词
        self.meta_config = self._load_meta()
        self.instructions = self._load_instructions()
        
    def _load_meta(self) -> Dict:
        """加载 meta.json 配置"""
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_instructions(self) -> Dict[str, str]:
        """加载 instructions.md 中的提示词模板"""
        instructions = {}
        with open(self.instructions_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析不同任务类型的提示词
        sections = content.split('\n# ')
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.strip().split('\n', 1)
            if len(lines) == 2:
                # 移除开头的 # 符号（如果存在）
                key = lines[0].strip().lstrip('#').strip()  # 如 NER_ZH, Action_EN
                template = lines[1].strip()
                instructions[key] = template
        
        return instructions
    
    def _format_ner_output(self, record: Dict, language: str) -> str:
        """
        格式化 NER 任务的输出为 JSON 格式
        
        Args:
            record: 数据记录
            language: 语言类型
        
        Returns:
            格式化的 JSON 字符串
        """
        output_data = {
            "reaction_product": record.get("reaction_product"),
            "starting_material": record.get("starting_material"),
            "reagent_catalyst": record.get("reagent_catalyst"),
            "solvent": record.get("solvent"),
            "temperature": record.get("temperature"),
            "time": record.get("time"),
            "yield_percent": record.get("yield_percent")
        }
        
        # 格式化为 Markdown JSON 代码块
        json_str = json.dumps(output_data, ensure_ascii=False, indent=2)
        return f"```json\n{json_str}\n```"
    
    def _format_action_output(self, record: Dict, language: str) -> str:
        """
        格式化 Action 任务的输出
        
        Args:
            record: 数据记录
            language: 语言类型
        
        Returns:
            动作序列字符串
        """
        action_sequence = record.get("action_sequence", "")
        if not action_sequence:
            return ""
        
        # 对于英文，可能需要将中文逗号转换为英文逗号
        if language == "EN":
            action_sequence = action_sequence.replace("；", ",").replace(";", ",")
        
        return action_sequence
    
    def _create_data_entry(self, record: Dict, task_type: str, language: str, source: str) -> Dict[str, Any]:
        """
        创建单条数据条目
        
        Args:
            record: 原始数据记录
            task_type: 任务类型 (NER/Action)
            language: 语言类型 (EN/ZH)
            source: 数据源
        
        Returns:
            格式化的数据条目
        """
        text = record.get("text", "")
        
        # 获取对应的 instruction 模板
        instruction_key = f"{task_type}_{language}"
        instruction_template = self.instructions.get(instruction_key, "")
        
        # 替换模板中的 {{TEXT}} 为实际文本
        instruction = instruction_template.replace("{{TEXT}}", text)
        
        # 根据任务类型格式化输出
        if task_type == "NER":
            output = self._format_ner_output(record, language)
        elif task_type == "Action":
            output = self._format_action_output(record, language)
        else:
            output = ""
        
        return {
            "instruction": instruction,
            "input": text,
            "output": output,
            "task_type": task_type,
            "language": language,
            "source": source
        }
    
    def _load_data_file(self, file_path: str) -> List[Dict]:
        """加载数据文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是列表则直接返回，否则包装成列表
        if isinstance(data, list):
            return data
        else:
            return [data]
    
    def generate_dataset(self, train_ratio: float = 0.8, valid_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        生成数据集
        
        Args:
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1"
        
        all_data = []
        
        # 遍历所有配置的文件
        for file_key, file_config in self.meta_config.get("files", {}).items():
            file_path = file_config["path"]
            tasks = file_config["tasks"]
            language = file_config["language"]
            source = file_config.get("source", "Unknown")
            
            print(f"处理文件: {file_path}")
            print(f"  任务类型: {tasks}")
            print(f"  语言: {language}")
            print(f"  数据源: {source}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"  警告: 文件不存在，跳过: {file_path}")
                continue
            
            # 加载数据
            try:
                records = self._load_data_file(file_path)
                print(f"  加载了 {len(records)} 条记录")
            except Exception as e:
                print(f"  错误: 无法加载文件 {file_path}: {e}")
                continue
            
            # 为每条记录和每个任务类型生成数据条目
            for record in records:
                for task_type in tasks:
                    # 对于 Action 任务，检查是否有 action_sequence 字段
                    if task_type == "Action" and not record.get("action_sequence"):
                        continue
                    
                    entry = self._create_data_entry(record, task_type, language, source)
                    
                    # 过滤掉输出为空的条目
                    if entry["output"]:
                        all_data.append(entry)
            
            print(f"  生成了 {len(all_data)} 条数据条目")
        
        print(f"\n总共生成 {len(all_data)} 条数据")
        
        # 打乱数据
        random.shuffle(all_data)
        
        # 划分数据集
        total_size = len(all_data)
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)
        
        train_data = all_data[:train_size]
        valid_data = all_data[train_size:train_size + valid_size]
        test_data = all_data[train_size + valid_size:]
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_data)} 条")
        print(f"  验证集: {len(valid_data)} 条")
        print(f"  测试集: {len(test_data)} 条")
        
        # 保存数据集
        self._save_datasets(train_data, valid_data, test_data)
    
    def _save_datasets(self, train_data: List[Dict], valid_data: List[Dict], test_data: List[Dict]):
        """保存数据集到文件"""
        # 创建输出目录
        output_dir = Path(self.output_base_dir) / f"seed-{self.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        datasets = {
            "train.json": train_data,
            "valid.json": valid_data,
            "test.json": test_data
        }
        
        for filename, data in datasets.items():
            output_path = output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n已保存: {output_path}")
        
        # 保存数据集统计信息
        stats = self._generate_statistics(train_data, valid_data, test_data)
        stats_path = output_dir / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"已保存统计信息: {stats_path}")
    
    def _generate_statistics(self, train_data: List[Dict], valid_data: List[Dict], test_data: List[Dict]) -> Dict:
        """生成数据集统计信息"""
        def get_stats(data: List[Dict]) -> Dict:
            stats = {
                "total": len(data),
                "by_task": {},
                "by_language": {},
                "by_source": {}
            }
            
            for entry in data:
                task = entry["task_type"]
                lang = entry["language"]
                source = entry["source"]
                
                stats["by_task"][task] = stats["by_task"].get(task, 0) + 1
                stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            
            return stats
        
        return {
            "seed": self.seed,
            "train": get_stats(train_data),
            "valid": get_stats(valid_data),
            "test": get_stats(test_data)
        }


def main():
    parser = argparse.ArgumentParser(description="生成训练、验证和测试数据集")
    parser.add_argument("--meta", type=str, default="meta.json", help="meta.json 文件路径")
    parser.add_argument("--instructions", type=str, default="instructions.md", help="instructions.md 文件路径")
    parser.add_argument("--output", type=str, default="../", help="输出目录路径")
    parser.add_argument("--seed", type=int, default=67, help="随机种子")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = DatasetGenerator(
        meta_path=args.meta,
        instructions_path=args.instructions,
        output_base_dir=args.output,
        seed=args.seed
    )
    
    # 生成数据集
    generator.generate_dataset(
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )
    
    print("\n数据集生成完成！")


if __name__ == "__main__":
    main()
