# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
稀疏检索器实现

基于 BM25 的稀疏检索器。
"""
from typing import Any, List, Optional, Dict
from typing import Literal

from openjiuwen.core.retrieval.retriever.base import Retriever
from openjiuwen.core.retrieval.vector_store.base import VectorStore
from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult


class SparseRetriever(Retriever):
    """稀疏检索器实现（BM25）"""

    def __init__(
        self,
        vector_store: VectorStore,
        **kwargs: Any,
    ):
        """
        初始化稀疏检索器
        
        Args:
            vector_store: 向量存储实例（需要支持稀疏搜索）
        """
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        mode: Literal["vector", "sparse", "hybrid"] = "sparse",
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """
        检索文档（稀疏检索）
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            score_threshold: 分数阈值
            mode: 检索模式（此检索器只支持 sparse）
            **kwargs: 额外参数
            
        Returns:
            检索结果列表
        """
        if mode != "sparse":
            raise ValueError(f"SparseRetriever only supports 'sparse' mode, got {mode}")

        # 执行稀疏搜索
        search_results = await self.vector_store.sparse_search(
            query_text=query,
            top_k=top_k,
            filters=None,
        )

        # 转换为 RetrievalResult
        retrieval_results = []
        for result in search_results:
            retrieval_result = RetrievalResult(
                text=result.text,
                score=result.score,
                metadata=result.metadata,
                doc_id=result.metadata.get("doc_id"),
                chunk_id=result.id,
            )
            retrieval_results.append(retrieval_result)

        return retrieval_results

    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[List[RetrievalResult]]:
        """批量检索"""
        import asyncio

        # 并发执行多个检索
        tasks = [self.retrieve(query, top_k=top_k, **kwargs) for query in queries]
        results = await asyncio.gather(*tasks)
        return results

    async def close(self) -> None:
        """关闭检索器"""
        import inspect

        if self.vector_store:
            close_fn = getattr(self.vector_store, "close", None)
            if close_fn:
                if inspect.iscoroutinefunction(close_fn):
                    await close_fn()
                else:
                    close_fn()
