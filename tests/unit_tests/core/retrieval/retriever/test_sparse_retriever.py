# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
稀疏检索器测试用例
"""
import pytest
from unittest.mock import AsyncMock

from openjiuwen.core.retrieval.retriever.sparse_retriever import SparseRetriever
from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult, SearchResult


@pytest.fixture
def mock_vector_store():
    """创建模拟向量存储"""
    store = AsyncMock()
    store.sparse_search = AsyncMock(return_value=[
        SearchResult(id="1", text="Result 1", score=0.95, metadata={"doc_id": "doc_1"}),
        SearchResult(id="2", text="Result 2", score=0.85, metadata={"doc_id": "doc_2"}),
    ])
    return store


class TestSparseRetriever:
    """稀疏检索器测试"""

    @pytest.mark.asyncio
    async def test_retrieve_success(self, mock_vector_store):
        """测试检索成功"""
        retriever = SparseRetriever(vector_store=mock_vector_store)
        results = await retriever.retrieve("test query", top_k=5)
        assert len(results) == 2
        assert results[0].text == "Result 1"
        assert results[0].score == 0.95
        mock_vector_store.sparse_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_invalid_mode(self, mock_vector_store):
        """测试无效的检索模式"""
        retriever = SparseRetriever(vector_store=mock_vector_store)
        with pytest.raises(ValueError, match="only supports 'sparse' mode"):
            await retriever.retrieve("test query", mode="vector")

    @pytest.mark.asyncio
    async def test_batch_retrieve(self, mock_vector_store):
        """测试批量检索"""
        retriever = SparseRetriever(vector_store=mock_vector_store)
        queries = ["query 1", "query 2"]
        results_list = await retriever.batch_retrieve(queries, top_k=5)
        assert len(results_list) == 2
        assert len(results_list[0]) == 2
        assert len(results_list[1]) == 2

