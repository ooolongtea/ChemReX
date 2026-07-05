"""
Prompt加载器 - 从Markdown文件加载prompt模板

使用 python-frontmatter 解析带有YAML Frontmatter的Markdown文件
"""

import os
import json
from typing import Dict, Any
from loguru import logger

try:
    import frontmatter
except ImportError:
    logger.warning(
        "python-frontmatter 未安装，请运行: pip install python-frontmatter"
    )
    frontmatter = None


class PromptLoader:
    """Prompt加载器 - 从MD文件加载prompt（支持版本管理）"""
    
    def __init__(self, prompts_dir: str = None):
        """
        Args:
            prompts_dir: prompts目录路径（默认为当前文件所在目录）
        """
        if prompts_dir is None:
            prompts_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.prompts_dir = prompts_dir
        self._cache = {}
        self._versions = None
        self._load_versions()
    
    def _load_versions(self):
        """加载版本配置文件"""
        versions_file = os.path.join(self.prompts_dir, "versions.json")
        try:
            if os.path.exists(versions_file):
                with open(versions_file, 'r', encoding='utf-8') as f:
                    self._versions = json.load(f)
            else:
                self._versions = {}
                logger.warning(f"versions.json 不存在，使用默认版本")
        except Exception as e:
            logger.error(f"加载versions.json失败: {e}")
            self._versions = {}
    
    def get_version(self, prompt_name: str) -> str:
        """
        获取prompt的当前版本
        
        Args:
            prompt_name: prompt名称
            
        Returns:
            版本号（如 "v1"）
        """
        if self._versions and prompt_name in self._versions:
            return self._versions[prompt_name]
        return "v1"  # 默认版本
    
    def load_prompt(self, prompt_name: str, version: str = None) -> Dict[str, Any]:
        """
        加载prompt文件（支持版本管理）
        
        Args:
            prompt_name: prompt名称（如 "reaction_extraction"）
            version: 指定版本（如 "v1"），如果为None则从versions.json读取
            
        Returns:
            包含metadata和content的字典:
            {
                "metadata": {...},  # YAML frontmatter
                "content": "..."    # prompt内容
            }
        """
        # 确定版本
        if version is None:
            version = self.get_version(prompt_name)
        
        # 检查缓存（使用 prompt_name:version 作为key）
        cache_key = f"{prompt_name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 构建文件路径：prompts/prompt_name/version.md
        prompt_dir = os.path.join(self.prompts_dir, prompt_name)
        file_path = os.path.join(prompt_dir, f"{version}.md")
        
        # 如果新结构不存在，尝试旧结构（向后兼容）
        if not os.path.exists(file_path):
            old_path = os.path.join(self.prompts_dir, f"{prompt_name}.md")
            if os.path.exists(old_path):
                logger.warning(f"使用旧结构: {old_path}")
                file_path = old_path
            else:
                raise FileNotFoundError(f"Prompt文件不存在: {file_path}")
        
        # 读取文件
        if frontmatter is None:
            # 如果没有安装frontmatter，使用简单的文本读取
            logger.warning(f"使用简单文本读取模式（未安装frontmatter）")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 简单解析YAML frontmatter（如果存在）
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        # 跳过frontmatter，只取内容
                        content = parts[2].strip()
                
                result = {
                    "metadata": {},
                    "content": content
                }
        else:
            # 使用frontmatter库解析
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            result = {
                "metadata": post.metadata,
                "content": post.content.strip()
            }
        
        # 缓存结果（使用 prompt_name:version 作为key）
        self._cache[cache_key] = result
        
        logger.debug(f"加载prompt: {prompt_name} (版本: {version}, {len(result['content'])} 字符)")
        
        return result
    
    def get_content(self, prompt_name: str) -> str:
        """
        只获取prompt内容（不含metadata）
        
        Args:
            prompt_name: prompt文件名（不含.md后缀）
            
        Returns:
            prompt内容字符串
        """
        return self.load_prompt(prompt_name)["content"]
    
    def get_metadata(self, prompt_name: str) -> Dict[str, Any]:
        """
        只获取prompt的metadata
        
        Args:
            prompt_name: prompt文件名（不含.md后缀）
            
        Returns:
            metadata字典
        """
        return self.load_prompt(prompt_name)["metadata"]
    
    def list_prompts(self) -> list:
        """列出所有可用的prompt（支持新版本管理结构）"""
        prompts = []
        
        # 扫描目录结构
        for item in os.listdir(self.prompts_dir):
            item_path = os.path.join(self.prompts_dir, item)
            
            # 跳过隐藏文件和特殊文件
            if item.startswith('_') or item.startswith('.'):
                continue
            
            # 如果是目录，检查是否是prompt目录（包含版本文件）
            if os.path.isdir(item_path):
                # 检查目录下是否有 .md 文件
                for file in os.listdir(item_path):
                    if file.endswith('.md'):
                        prompts.append(item)
                        break
            # 如果是 .md 文件（旧结构），也添加
            elif item.endswith('.md'):
                prompts.append(item[:-3])  # 去掉.md后缀
        
        return sorted(set(prompts))
    
    def format_prompt(self, prompt_name: str, **variables) -> str:
        """
        格式化prompt模板（替换变量）
        
        支持两种语法:
        - {{ variable }} - Jinja2风格
        - {variable} - Python format风格
        
        Args:
            prompt_name: prompt文件名
            **variables: 要替换的变量
            
        Returns:
            格式化后的prompt
        """
        content = self.get_content(prompt_name)
        
        # 先处理 Jinja2 风格的 {{ variable }}
        import re
        for key, value in variables.items():
            # 替换 {{ key }} 和 {{key}}（有无空格都支持）
            pattern = r'\{\{\s*' + re.escape(key) + r'\s*\}\}'
            content = re.sub(pattern, str(value), content)
        
        # 再尝试处理 Python format 风格的 {variable}（如果还有的话）
        try:
            content = content.format(**variables)
        except (KeyError, ValueError):
            # 如果还有未替换的变量，记录警告但继续
            pass
        
        # 检查是否还有未替换的变量
        remaining = re.findall(r'\{\{\s*(\w+)\s*\}\}', content)
        if remaining:
            logger.warning(f"Prompt {prompt_name} 有未替换的变量: {remaining}")
        
        return content
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()


# 创建全局实例
_default_loader = None


def get_loader(prompts_dir: str = None) -> PromptLoader:
    """获取全局PromptLoader实例"""
    global _default_loader
    
    if _default_loader is None:
        _default_loader = PromptLoader(prompts_dir)
    
    return _default_loader


def load_prompt(prompt_name: str) -> str:
    """便捷函数：加载prompt内容"""
    return get_loader().get_content(prompt_name)


# 使用示例
if __name__ == "__main__":
    # 测试加载器
    loader = PromptLoader()
    
    print("=" * 60)
    print("可用的prompt文件:")
    for name in loader.list_prompts():
        print(f"  - {name}")
    
    print("\n" + "=" * 60)
    print("版本配置:")
    if loader._versions:
        for name, version in loader._versions.items():
            print(f"  {name}: {version}")
    else:
        print("  无版本配置")
    
    print("\n" + "=" * 60)
    print("测试加载 reaction_extraction:")
    try:
        prompt_data = loader.load_prompt("reaction_extraction")
        print(f"\n元数据:")
        for key, value in prompt_data["metadata"].items():
            print(f"  {key}: {value}")
        
        print(f"\n内容长度: {len(prompt_data['content'])} 字符")
        print(f"内容预览:\n{prompt_data['content'][:200]}...")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Prompt加载器测试完成")

