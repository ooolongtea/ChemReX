#!/usr/bin/env python3
"""
USPTO-ORD-100k 操作序列提取脚本
使用大模型并行提取化学反应的操作序列
支持断点续传功能
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import pickle
from threading import Lock
import argparse

from loguru import logger
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[3]  # 回到 RetroParser 根目录
sys.path.insert(0, str(project_root))

from GPTAPI import GPTAPI
from prompts.prompt_loader import PromptLoader


class CheckpointManager:
    """断点续传管理器"""
    
    def __init__(self, checkpoint_dir: Path, checkpoint_interval: int = 100):
        """
        Args:
            checkpoint_dir: checkpoint 保存目录
            checkpoint_interval: 每处理多少条保存一次
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        
        # checkpoint 文件路径
        self.results_file = self.checkpoint_dir / "checkpoint_action_results.json"
        self.progress_file = self.checkpoint_dir / "checkpoint_action_progress.pkl"
        self.metadata_file = self.checkpoint_dir / "checkpoint_action_metadata.json"
        
        self.lock = Lock()
        
    def has_checkpoint(self) -> bool:
        """检查是否存在有效的 checkpoint"""
        return (self.results_file.exists() and 
                self.progress_file.exists() and 
                self.metadata_file.exists())
    
    def load_checkpoint(self) -> Tuple[List[Dict[str, Any]], Set[int], Dict[str, Any]]:
        """
        加载 checkpoint
        
        Returns:
            (已完成的结果列表, 已处理的索引集合, 元数据)
        """
        try:
            # 加载结果
            with open(self.results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 加载进度
            with open(self.progress_file, 'rb') as f:
                processed_indices = pickle.load(f)
            
            # 加载元数据
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"成功加载 checkpoint - 已完成: {len(results)} 条")
            return results, processed_indices, metadata
            
        except Exception as e:
            logger.error(f"加载 checkpoint 失败: {e}")
            raise
    
    def save_checkpoint(self, 
                       results: List[Dict[str, Any]], 
                       processed_indices: Set[int],
                       metadata: Dict[str, Any]):
        """
        保存 checkpoint
        
        Args:
            results: 已完成的结果列表
            processed_indices: 已处理的索引集合
            metadata: 元数据
        """
        with self.lock:
            try:
                # 保存结果
                with open(self.results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                # 保存进度
                with open(self.progress_file, 'wb') as f:
                    pickle.dump(processed_indices, f)
                
                # 保存元数据
                metadata['last_save_time'] = datetime.now().isoformat()
                metadata['total_processed'] = len(results)
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logger.error(f"保存 checkpoint 失败: {e}")
                raise
    
    def clear_checkpoint(self):
        """清除 checkpoint 文件"""
        try:
            if self.results_file.exists():
                self.results_file.unlink()
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info("Checkpoint 文件已清除")
        except Exception as e:
            logger.warning(f"清除 checkpoint 失败: {e}")


class ActionSequenceExtractor:
    """化学反应操作序列提取器"""
    
    def __init__(self, 
                 model: str = "deepseek-v3",
                 temperature: float = 0.0,
                 max_workers: int = 10,
                 max_retries: int = 3,
                 retry_delay: float = 2.0,
                 checkpoint_manager: Optional[CheckpointManager] = None):
        """
        Args:
            model: 使用的模型名称
            temperature: 温度参数
            max_workers: 并行工作线程数
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            checkpoint_manager: 断点管理器
        """
        self.gpt_api = GPTAPI(model=model, temperature=temperature)
        self.prompt_loader = PromptLoader(prompts_dir=str(project_root / "prompts"))
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.checkpoint_manager = checkpoint_manager
        
        # 加载提取 prompt
        self.extraction_prompt = self.prompt_loader.get_content("extract_en_sequence")
        logger.info(f"初始化完成 - 模型: {model}, 并行数: {max_workers}, 最大重试: {max_retries}")
    
    def parse_action_sequence(self, response: str) -> Optional[str]:
        """
        解析大模型返回的操作序列
        
        Args:
            response: 大模型返回的文本
            
        Returns:
            操作序列字符串，失败返回 None
        """
        if not response:
            return None
        
        # 去除首尾空白
        action_sequence = response.strip()
        
        # 如果响应为空或者包含错误信息
        if not action_sequence or action_sequence.lower() in ["none", "null", "n/a"]:
            return None
        
        return action_sequence
    
    def extract_single(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        提取单条数据的操作序列（带重试机制）
        
        Args:
            item: 包含 text 等字段的字典
            index: 数据索引
            
        Returns:
            添加 action 字段后的完整数据字典
        """
        text = item.get("text", "")
        
        action_sequence = None
        for attempt in range(self.max_retries):
            try:
                # 格式化 prompt
                formatted_prompt = self.extraction_prompt.replace("{{text}}", text)
                
                # 调用大模型
                response = self.gpt_api.answer_wo_vision(
                    prompt=formatted_prompt,
                    content=None
                )
                
                # 解析响应
                action_sequence = self.parse_action_sequence(response)
                
                if action_sequence:
                    break
                else:
                    logger.debug(f"索引 {index}: 未能提取到有效的操作序列")
                
            except Exception as e:
                logger.warning(f"提取失败 (尝试 {attempt + 1}/{self.max_retries}) - 索引: {index}, 错误: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"提取最终失败 - 索引: {index}")
                    action_sequence = None
        
        # 构建结果（保持原有字段顺序，action 放在倒数第二个位置）
        result = {
            "text": item.get("text"),
            "REACTION_PRODUCT": item.get("REACTION_PRODUCT"),
            "STARTING_MATERIAL": item.get("STARTING_MATERIAL"),
            "REAGENT_CATALYST": item.get("REAGENT_CATALYST"),
            "SOLVENT": item.get("SOLVENT"),
            "TEMPERATURE": item.get("TEMPERATURE"),
            "TIME": item.get("TIME"),
            "YIELD_PERCENT": item.get("YIELD_PERCENT"),
            "action": action_sequence,  # 新增字段，倒数第二个
            "rxn": item.get("rxn")      # 最后一个字段
        }
        
        return result
    
    def extract_batch(self, 
                     data: List[Dict[str, Any]],
                     resume: bool = False) -> List[Dict[str, Any]]:
        """
        批量并行提取（支持断点续传）
        
        Args:
            data: 数据列表
            resume: 是否从断点恢复
            
        Returns:
            提取后的数据列表
        """
        results = []
        processed_indices = set()
        completed_count = 0
        
        # 如果是恢复模式，加载之前的进度
        if resume and self.checkpoint_manager and self.checkpoint_manager.has_checkpoint():
            logger.info("检测到 checkpoint，正在恢复...")
            results, processed_indices, metadata = self.checkpoint_manager.load_checkpoint()
            completed_count = len(results)
            logger.info(f"从 checkpoint 恢复 - 已完成: {completed_count} 条")
            
            # 过滤出未处理的数据（使用索引）
            remaining_items = [(i, item) for i, item in enumerate(data) if i not in processed_indices]
            logger.info(f"剩余待处理: {len(remaining_items)} 条")
        else:
            remaining_items = list(enumerate(data))
            completed_count = 0
        
        total = len(data)
        
        if len(remaining_items) == 0:
            logger.success("所有数据已处理完成！")
            return results
        
        logger.info(f"开始处理 {len(remaining_items)} 条数据，使用 {self.max_workers} 个并行线程")
        
        # 用于线程安全的结果收集
        results_lock = Lock()
        save_counter = [completed_count]  # 使用列表以便在闭包中修改
        
        def process_and_save(index_item_tuple):
            """处理单条数据并定期保存"""
            index, item = index_item_tuple
            result = self.extract_single(item, index)
            
            with results_lock:
                results.append(result)
                processed_indices.add(index)
                save_counter[0] += 1
                
                # 定期保存 checkpoint
                if (self.checkpoint_manager and 
                    save_counter[0] % self.checkpoint_manager.checkpoint_interval == 0):
                    metadata = {
                        'total_items': total,
                        'processed': save_counter[0],
                        'timestamp': datetime.now().isoformat()
                    }
                    self.checkpoint_manager.save_checkpoint(results, processed_indices, metadata)
                    logger.info(f"💾 Checkpoint 已保存 - 进度: {save_counter[0]}/{total}")
            
            return result
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_and_save, item): item 
                for item in remaining_items
            }
            
            # 使用 tqdm 显示进度
            with tqdm(total=len(remaining_items), 
                     initial=0,
                     desc="提取进度", 
                     unit="条",
                     postfix={'已完成': completed_count}) as pbar:
                for future in as_completed(future_to_item):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"处理过程出错: {e}")
                    pbar.update(1)
                    pbar.set_postfix({'总进度': f"{save_counter[0]}/{total}"})
        
        # 最后保存一次
        if self.checkpoint_manager:
            metadata = {
                'total_items': total,
                'processed': save_counter[0],
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            self.checkpoint_manager.save_checkpoint(results, processed_indices, metadata)
            logger.info(f"💾 最终 checkpoint 已保存")
        
        logger.success(f"全部完成！共处理 {len(results)} 条数据")
        return results


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='USPTO-ORD-100k 操作序列提取')
    parser.add_argument('--resume', action='store_true', help='从断点恢复')
    parser.add_argument('--clear-checkpoint', action='store_true', help='清除现有checkpoint')
    args = parser.parse_args()
    
    # 配置日志
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"action_extraction_{timestamp}.log"
    
    logger.add(
        log_file,
        rotation="500 MB",
        encoding="utf-8",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info("=" * 60)
    logger.info("USPTO-ORD-100k 操作序列提取脚本启动")
    logger.info("=" * 60)
    
    # 文件路径
    input_file = Path(__file__).parent / "uspto_ord_with_rxn.json"
    output_file = Path(__file__).parent / f"uspto_ord_with_action_{timestamp}.json"
    checkpoint_dir = Path(__file__).parent / "checkpoints_action"
    
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 初始化 CheckpointManager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=100
    )
    
    # 如果需要清除checkpoint
    if args.clear_checkpoint:
        logger.info("清除现有 checkpoint...")
        checkpoint_manager.clear_checkpoint()
    
    # 1. 读取原始数据
    logger.info(f"读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"读取完成，共 {len(data)} 条数据")
    
    # 统计 rxn 字段
    rxn_count = sum(1 for item in data if item.get('rxn') is not None)
    logger.info(f"含有效 rxn 字段的数据: {rxn_count}/{len(data)}")
    
    # 2. 初始化提取器（传入 checkpoint_manager）
    extractor = ActionSequenceExtractor(
        model="deepseek-v3",
        temperature=0.0,
        max_workers=16,  # 并行数，根据 API 限制调整
        max_retries=3,
        retry_delay=2.0,
        checkpoint_manager=checkpoint_manager  # 传入断点管理器
    )
    
    # 3. 批量提取（支持断点恢复）
    start_time = time.time()
    results = extractor.extract_batch(data, resume=args.resume)
    elapsed_time = time.time() - start_time
    
    logger.info(f"提取耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    logger.info(f"平均速度: {len(results)/elapsed_time:.2f} 条/秒")
    
    # 4. 保存结果
    logger.info(f"保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 5. 统计
    action_count = sum(1 for item in results if item.get('action') is not None)
    fail_count = len(results) - action_count
    success_rate = (action_count / len(results) * 100) if results else 0
    
    logger.info(f"提取成功: {action_count}/{len(results)} ({success_rate:.2f}%)")
    logger.info(f"提取失败: {fail_count}/{len(results)} ({100-success_rate:.2f}%)")
    
    # 6. 清除 checkpoint（任务完成后）
    logger.info("任务完成，清除 checkpoint...")
    checkpoint_manager.clear_checkpoint()
    
    logger.success("=" * 60)
    logger.success(f"任务完成！结果已保存到: {output_file}")
    logger.success(f"成功: {action_count}, 失败: {fail_count}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
