# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
混合检索器测试用例
"""
from unittest.mock import AsyncMock

import pytest

from openjiuwen.core.retrieval.retriever.hybrid_retriever import HybridRetriever
from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult, SearchResult


@pytest.fixture
def mock_vector_store():
    """创建模拟向量存储"""
    store = AsyncMock()
    store.hybrid_search = AsyncMock(return_value=[
        SearchResult(id="1", text="Hybrid result 1", score=0.95, metadata={"doc_id": "doc_1"}),
        SearchResult(id="2", text="Hybrid result 2", score=0.85, metadata={"doc_id": "doc_2"}),
    ])
    store.search = AsyncMock(return_value=[
        SearchResult(id="1", text="Vector result", score=0.9, metadata={}),
    ])
    store.sparse_search = AsyncMock(return_value=[
        SearchResult(id="1", text="Sparse result", score=0.8, metadata={}),
    ])
    return store


@pytest.fixture
def mock_embed_model():
    """创建模拟嵌入模型"""
    model = AsyncMock()
    model.embed_query = AsyncMock(return_value=[0.1] * 384)
    return model


class TestHybridRetriever:
    """混合检索器测试"""

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_mode(self, mock_vector_store, mock_embed_model):
        """测试混合检索模式"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        results = await retriever.retrieve("test query", top_k=5, mode="hybrid")
        assert len(results) == 2
        mock_vector_store.hybrid_search.assert_called_once()
        mock_embed_model.embed_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_vector_mode(self, mock_vector_store, mock_embed_model):
        """测试向量检索模式"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        results = await retriever.retrieve("test query", top_k=5, mode="vector")
        assert len(results) == 1
        mock_vector_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_sparse_mode(self, mock_vector_store):
        """测试稀疏检索模式"""
        retriever = HybridRetriever(vector_store=mock_vector_store)
        results = await retriever.retrieve("test query", top_k=5, mode="sparse")
        assert len(results) == 1
        mock_vector_store.sparse_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_custom_alpha(self, mock_vector_store, mock_embed_model):
        """测试使用自定义 alpha 参数"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
            alpha=0.7,
        )
        results = await retriever.retrieve("test query", top_k=5, alpha=0.8)
        # 验证使用了自定义的 alpha
        call_args = mock_vector_store.hybrid_search.call_args
        assert call_args[1]["alpha"] == 0.8

    @pytest.mark.asyncio
    async def test_retrieve_vector_mode_without_embed_model(self, mock_vector_store):
        """测试向量模式但没有嵌入模型"""
        retriever = HybridRetriever(vector_store=mock_vector_store)
        with pytest.raises(ValueError, match="embed_model is required"):
            await retriever.retrieve("test query", mode="vector")

    @pytest.mark.asyncio
    async def test_retrieve_with_score_threshold(self, mock_vector_store, mock_embed_model):
        """测试使用分数阈值（仅向量模式支持）"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        results = await retriever.retrieve(
            "test query", top_k=5, mode="vector", score_threshold=0.85
        )
        # 应该过滤掉 score < 0.85 的结果
        for result in results:
            assert result.score >= 0.85

    @pytest.mark.asyncio
    async def test_retrieve_score_threshold_invalid_mode(self, mock_vector_store, mock_embed_model):
        """测试分数阈值在非向量模式下使用"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        with pytest.raises(ValueError, match="score_threshold is only supported"):
            await retriever.retrieve("test query", mode="hybrid", score_threshold=0.5)

    @pytest.mark.asyncio
    async def test_batch_retrieve(self, mock_vector_store, mock_embed_model):
        """测试批量检索"""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        queries = ["query 1", "query 2"]
        results_list = await retriever.batch_retrieve(queries, top_k=5)
        assert len(results_list) == 2

