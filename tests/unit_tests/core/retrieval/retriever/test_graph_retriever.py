# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
图检索器测试用例
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openjiuwen.core.retrieval.retriever.graph_retriever import GraphRetriever
from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult


@pytest.fixture
def mock_chunk_retriever():
    """创建模拟块检索器"""
    retriever = AsyncMock()
    retriever.retrieve = AsyncMock(return_value=[
        RetrievalResult(
            text="Chunk result 1",
            score=0.9,
            chunk_id="chunk_1",
            doc_id="doc_1",
            metadata={"chunk_id": "chunk_1", "doc_id": "doc_1"},
        ),
        RetrievalResult(
            text="Chunk result 2",
            score=0.8,
            chunk_id="chunk_2",
            doc_id="doc_1",
            metadata={"chunk_id": "chunk_2", "doc_id": "doc_1"},
        ),
    ])
    return retriever


@pytest.fixture
def mock_triple_retriever():
    """创建模拟三元组检索器"""
    retriever = AsyncMock()
    retriever.retrieve = AsyncMock(return_value=[
        RetrievalResult(
            text="Triple result 1",
            score=0.85,
            metadata={"chunk_id": "chunk_3", "doc_id": "doc_1"},
        ),
    ])
    return retriever


@pytest.fixture
def mock_vector_store():
    """创建模拟向量存储"""
    store = MagicMock()
    store.collection_name = None
    return store


@pytest.fixture
def mock_embed_model():
    """创建模拟嵌入模型"""
    model = AsyncMock()
    model.embed_query = AsyncMock(return_value=[0.1] * 384)
    return model


class TestGraphRetriever:
    """图检索器测试"""

    def test_init_with_retrievers(self, mock_chunk_retriever, mock_triple_retriever):
        """测试使用检索器初始化"""
        retriever = GraphRetriever(
            chunk_retriever=mock_chunk_retriever,
            triple_retriever=mock_triple_retriever,
        )
        assert retriever.chunk_retriever == mock_chunk_retriever
        assert retriever.triple_retriever == mock_triple_retriever

    def test_init_with_vector_store(self, mock_vector_store, mock_embed_model):
        """测试使用向量存储初始化"""
        retriever = GraphRetriever(
            vector_store=mock_vector_store,
            embed_model=mock_embed_model,
            chunk_collection="chunks",
            triple_collection="triples",
        )
        assert retriever.vector_store == mock_vector_store
        assert retriever.embed_model == mock_embed_model
        assert retriever.chunk_collection == "chunks"
        assert retriever.triple_collection == "triples"

    @pytest.mark.asyncio
    async def test_retrieve_score_threshold_invalid_mode(self, mock_chunk_retriever):
        """测试非向量模式使用分数阈值"""
        retriever = GraphRetriever(chunk_retriever=mock_chunk_retriever)
        
        with pytest.raises(ValueError, match="score_threshold is only supported"):
            await retriever.retrieve(
                "test query", top_k=5, mode="sparse", score_threshold=0.8
            )

    @pytest.mark.asyncio
    async def test_graph_expansion_empty_chunks(self, mock_chunk_retriever):
        """测试图扩展（空初始块）"""
        retriever = GraphRetriever(chunk_retriever=mock_chunk_retriever)
        
        results = await retriever.graph_expansion(
            query="test query",
            chunks=[],
            topk=5,
            mode="hybrid",
        )
        # 应该返回空列表或回退结果
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_close(self, mock_chunk_retriever, mock_triple_retriever):
        """测试关闭检索器"""
        mock_chunk_retriever.close = AsyncMock()
        mock_triple_retriever.close = AsyncMock()
        
        retriever = GraphRetriever(
            chunk_retriever=mock_chunk_retriever,
            triple_retriever=mock_triple_retriever,
        )
        
        await retriever.close()
        mock_chunk_retriever.close.assert_called_once()
        mock_triple_retriever.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_sync_close(self, mock_chunk_retriever):
        """测试关闭检索器（同步 close 方法）"""
        mock_chunk_retriever.close = MagicMock()
        
        retriever = GraphRetriever(chunk_retriever=mock_chunk_retriever)
        
        await retriever.close()
        mock_chunk_retriever.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_close_method(self):
        """测试关闭没有 close 方法的检索器"""
        mock_retriever = MagicMock()
        del mock_retriever.close  # 移除 close 方法
        
        retriever = GraphRetriever(chunk_retriever=mock_retriever)
        # 不应该抛出异常
        await retriever.close()

