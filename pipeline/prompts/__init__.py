"""
Prompts模块 - 管理所有的prompt模板（支持版本管理）

使用方式：
```python
from prompts import PromptLoader

# 创建加载器
loader = PromptLoader()

# 加载prompt（自动从versions.json读取版本）
prompt_data = loader.load_prompt("reaction_extraction")

# 格式化prompt（填充模板变量）
formatted = loader.format_prompt("reaction_extraction", content="...", source_type="Patent")
```
"""

from .prompt_loader import PromptLoader, get_loader, load_prompt

__all__ = [
    "PromptLoader",
    "get_loader",
    "load_prompt",
]

