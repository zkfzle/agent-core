# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
处理器抽象基类

所有处理器（Parser、Chunker、Extractor）的基类。
"""
from abc import ABC, abstractmethod
from typing import Any


class Processor(ABC):
    """处理器抽象基类，所有处理器（Parser、Chunker、Extractor）的基类"""
    
    @abstractmethod
    async def process(self, *args: Any, **kwargs: Any) -> Any:
        """
        处理数据（抽象方法，子类必须实现）
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            处理结果
        """
        pass
