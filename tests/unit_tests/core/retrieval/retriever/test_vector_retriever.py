# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
向量检索器测试用例
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from openjiuwen.core.retrieval.retriever.vector_retriever import VectorRetriever
from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult, SearchResult


@pytest.fixture
def mock_vector_store():
    """创建模拟向量存储"""
    store = AsyncMock()
    store.search = AsyncMock(return_value=[
        SearchResult(id="1", text="Result 1", score=0.95, metadata={"doc_id": "doc_1"}),
        SearchResult(id="2", text="Result 2", score=0.85, metadata={"doc_id": "doc_2"}),
    ])
    store.sparse_search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_embed_model():
    """创建模拟嵌入模型"""
    model = AsyncMock()
    model.embed_query = AsyncMock(return_value=[0.1] * 384)
    model.embed_documents = AsyncMock(return_value=[[0.1] * 384] * 2)
    return model


class TestVectorRetriever:
    """向量检索器测试"""

    @pytest.mark.asyncio
    async def test_retrieve_success(self, mock_vector_store, mock_embed_model):
        """测试检索成功"""
        retriever = VectorRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        results = await retriever.retrieve("test query", top_k=5)
        assert len(results) == 2
        assert results[0].text == "Result 1"
        assert results[0].score == 0.95
        mock_embed_model.embed_query.assert_called_once()
        mock_vector_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_score_threshold(self, mock_vector_store, mock_embed_model):
        """测试使用分数阈值"""
        retriever = VectorRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        results = await retriever.retrieve("test query", top_k=5, score_threshold=0.9)
        assert len(results) == 1  # 只有 score >= 0.9 的结果
        assert results[0].score >= 0.9

    @pytest.mark.asyncio
    async def test_retrieve_fallback_to_sparse(self, mock_vector_store, mock_embed_model):
        """测试向量检索无结果时回退到稀疏检索"""
        mock_vector_store.search = AsyncMock(return_value=[])
        mock_vector_store.sparse_search = AsyncMock(return_value=[
            SearchResult(id="1", text="Sparse result", score=0.8, metadata={}),
        ])
        
        retriever = VectorRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        results = await retriever.retrieve("test query", top_k=5)
        assert len(results) == 1
        assert results[0].text == "Sparse result"
        mock_vector_store.sparse_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_invalid_mode(self, mock_vector_store, mock_embed_model):
        """测试无效的检索模式"""
        retriever = VectorRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        with pytest.raises(ValueError, match="only supports 'vector' mode"):
            await retriever.retrieve("test query", mode="sparse")

    @pytest.mark.asyncio
    async def test_retrieve_without_embed_model(self, mock_vector_store):
        """测试没有嵌入模型时检索"""
        retriever = VectorRetriever(vector_store=mock_vector_store)
        with pytest.raises(ValueError, match="embed_model is required"):
            await retriever.retrieve("test query")

    @pytest.mark.asyncio
    async def test_batch_retrieve(self, mock_vector_store, mock_embed_model):
        """测试批量检索"""
        retriever = VectorRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
        )
        queries = ["query 1", "query 2"]
        results_list = await retriever.batch_retrieve(queries, top_k=5)
        assert len(results_list) == 2
        assert len(results_list[0]) == 2
        assert len(results_list[1]) == 2

