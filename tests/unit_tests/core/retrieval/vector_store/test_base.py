# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
向量存储抽象基类测试用例
"""
import pytest
from unittest.mock import AsyncMock

from openjiuwen.core.retrieval.vector_store.base import VectorStore
from openjiuwen.core.retrieval.common.retrieval_result import SearchResult


class ConcreteVectorStore(VectorStore):
    """具体向量存储实现，用于测试抽象基类"""

    async def add(self, data, batch_size=None, **kwargs):
        pass

    async def search(self, query_vector, top_k=5, filters=None, **kwargs):
        return []

    async def sparse_search(self, query_text, top_k=5, filters=None, **kwargs):
        return []

    async def hybrid_search(
        self, query_text, query_vector=None, top_k=5, alpha=0.5, filters=None, **kwargs
    ):
        return []

    async def delete(self, ids=None, filter_expr=None, **kwargs):
        return True


class TestVectorStore:
    """向量存储抽象基类测试"""

    @pytest.mark.asyncio
    async def test_add(self):
        """测试添加向量"""
        store = ConcreteVectorStore()
        # 不应该抛出异常
        await store.add({"id": "1", "text": "test", "embedding": [0.1] * 384})

    @pytest.mark.asyncio
    async def test_search(self):
        """测试向量搜索"""
        store = ConcreteVectorStore()
        results = await store.search([0.1] * 384, top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_sparse_search(self):
        """测试稀疏搜索"""
        store = ConcreteVectorStore()
        results = await store.sparse_search("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        """测试混合搜索"""
        store = ConcreteVectorStore()
        results = await store.hybrid_search(
            "test query", query_vector=[0.1] * 384, top_k=5
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_delete(self):
        """测试删除向量"""
        store = ConcreteVectorStore()
        result = await store.delete(ids=["1", "2"])
        assert result is True

