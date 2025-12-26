# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
向量检索器实现

基于向量存储的检索器实现。
"""
from typing import Any, List, Optional, Dict
from typing import Literal

from openjiuwen.core.retrieval.retriever.base import Retriever
from openjiuwen.core.retrieval.vector_store.base import VectorStore
from openjiuwen.core.retrieval.embedding.base import Embedding
from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult


class VectorRetriever(Retriever):
    """向量检索器实现"""

    def __init__(
        self,
        vector_store: VectorStore,
        embed_model: Optional[Embedding] = None,
        **kwargs: Any,
    ):
        """
        初始化向量检索器
        
        Args:
            vector_store: 向量存储实例
            embed_model: 嵌入模型实例（向量检索必需）
        """
        self.vector_store = vector_store
        self.embed_model = embed_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        mode: Literal["vector", "sparse", "hybrid"] = "vector",
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """
        检索文档（向量检索）
        
        Args:
            query: 查询字符串
            top_k: 返回数量
            score_threshold: 分数阈值
            mode: 检索模式（只支持 vector=向量检索）
            **kwargs: 额外参数
            
        Returns:
            检索结果列表
        """
        if mode != "vector":
            raise ValueError(f"VectorRetriever only supports 'vector' mode, got {mode}")

        if score_threshold is not None and mode != "vector":
            raise ValueError("score_threshold is only supported when mode='vector'")

        # 向量检索
        if self.embed_model is None:
            raise ValueError("embed_model is required for vector search")
        
        query_vector = await self.embed_model.embed_query(query)
        search_results = await self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )
        # 若向量检索无结果，回退 BM25
        if not search_results:
            search_results = await self.vector_store.sparse_search(
                query_text=query,
                top_k=top_k,
                filters=None,
            )

        # 转换为 RetrievalResult
        retrieval_results = []
        for result in search_results:
            if score_threshold is not None and result.score is not None and result.score < score_threshold:
                continue
            
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
        tasks = [
            self.retrieve(query, top_k=top_k, **kwargs) for query in queries
        ]
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
