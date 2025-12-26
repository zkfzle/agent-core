# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
嵌入模型抽象基类

提供嵌入模型的统一接口。
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class Embedding(ABC):
    """嵌入模型抽象基类"""
    
    @abstractmethod
    async def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        """嵌入查询文本"""
        pass
    
    @abstractmethod
    async def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """嵌入文档文本"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入维度"""
        pass
