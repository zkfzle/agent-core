# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
提取器抽象基类

继承 Processor，用于提取三元组等。
"""
from abc import abstractmethod
from typing import List, Any

from openjiuwen.core.retrieval.indexing.processor.base import Processor
from openjiuwen.core.retrieval.common.document import TextChunk
from openjiuwen.core.retrieval.common.triple import Triple


class Extractor(Processor):
    """提取器抽象基类（继承 Processor，用于提取三元组等）"""
    
    @abstractmethod
    async def extract(
        self,
        chunks: List[TextChunk],
        **kwargs: Any,
    ) -> List[Triple]:
        """
        提取信息（如三元组）
        
        Args:
            chunks: 文本块列表
            **kwargs: 额外参数
            
        Returns:
            提取结果列表（如三元组列表）
        """
        pass
    
    async def process(self, chunks: List[TextChunk], **kwargs: Any) -> List[Triple]:
        """
        处理文本块（实现 Processor 的 process 方法）
        
        Args:
            chunks: 文本块列表
            **kwargs: 额外参数
            
        Returns:
            提取结果列表
        """
        return await self.extract(chunks, **kwargs)
