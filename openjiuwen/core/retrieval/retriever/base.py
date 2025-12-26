# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
检索器抽象基类

提供检索器的统一接口。
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
from typing import Literal

from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult


class Retriever(ABC):
    """检索器抽象基类"""
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        mode: Literal["vector", "sparse", "hybrid"] = "hybrid",
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """
        检索文档
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            score_threshold: 分数阈值
            mode: 检索模式（vector=向量检索，sparse=稀疏检索/BM25，hybrid=混合检索）
            **kwargs: 额外参数
            
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[List[RetrievalResult]]:
        """批量检索"""
        pass
    
    async def close(self) -> None:
        """关闭检索器，释放资源"""
