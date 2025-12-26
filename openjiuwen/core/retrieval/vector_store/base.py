# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
向量存储抽象基类

提供向量存储的统一接口。
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from openjiuwen.core.retrieval.common.retrieval_result import SearchResult


class VectorStore(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    async def add(
        self,
        data: dict | List[dict],
        batch_size: int | None = 128,
        **kwargs: Any,
    ) -> None:
        """添加向量"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """
        向量搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回数量
            filters: 元数据过滤条件
            **kwargs: 额外参数
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    async def sparse_search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """
        稀疏搜索（BM25）
        
        Args:
            query_text: 查询文本
            top_k: 返回数量
            filters: 元数据过滤条件
            **kwargs: 额外参数
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """
        混合搜索（稀疏检索 + 向量检索）
        
        Args:
            query_text: 查询文本
            query_vector: 查询向量（可选，如果提供则使用，否则需要先嵌入）
            top_k: 返回数量
            alpha: 混合权重（0=纯稀疏检索，1=纯向量检索，0.5=平衡）
            filters: 元数据过滤条件
            **kwargs: 额外参数
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """删除向量"""
        pass
