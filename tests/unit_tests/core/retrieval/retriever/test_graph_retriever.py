# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
图检索器测试用例
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_allowed_modes(self):
        """测试允许的模式"""
        retriever = GraphRetriever()
        modes = retriever._allowed_modes()
        assert "vector" in modes
        assert "bm25" in modes
        assert "hybrid" in modes
        assert modes["vector"] == {"vector"}
        assert modes["bm25"] == {"sparse"}
        assert modes["hybrid"] == {"vector", "sparse", "hybrid"}

    def test_ensure_mode_allowed_no_index_type(self):
        """测试未设置 index_type 时的模式检查"""
        retriever = GraphRetriever()
        # 不应该抛出异常
        retriever._ensure_mode_allowed("vector")
        retriever._ensure_mode_allowed("sparse")
        retriever._ensure_mode_allowed("hybrid")

    def test_ensure_mode_allowed_with_index_type(self):
        """测试设置 index_type 时的模式检查"""
        retriever = GraphRetriever()
        retriever.index_type = "vector"
        
        # 应该允许 vector 模式
        retriever._ensure_mode_allowed("vector")
        
        # 应该拒绝其他模式
        with pytest.raises(ValueError, match="不兼容"):
            retriever._ensure_mode_allowed("sparse")
        
        with pytest.raises(ValueError, match="不兼容"):
            retriever._ensure_mode_allowed("hybrid")

    def test_ensure_mode_allowed_invalid_index_type(self):
        """测试无效的 index_type"""
        retriever = GraphRetriever()
        retriever.index_type = "invalid"
        
        with pytest.raises(ValueError, match="Unsupported index_type"):
            retriever._ensure_mode_allowed("vector")

    def test_retriever_supports_mode(self):
        """测试检索器模式支持检查"""
        from openjiuwen.core.retrieval.retriever.vector_retriever import VectorRetriever
        from openjiuwen.core.retrieval.retriever.sparse_retriever import SparseRetriever
        from openjiuwen.core.retrieval.retriever.hybrid_retriever import HybridRetriever

        retriever = GraphRetriever()
        
        mock_vector = MagicMock(spec=VectorRetriever)
        assert retriever._retriever_supports_mode(mock_vector, "vector") is True
        assert retriever._retriever_supports_mode(mock_vector, "sparse") is False

        mock_sparse = MagicMock(spec=SparseRetriever)
        assert retriever._retriever_supports_mode(mock_sparse, "sparse") is True
        assert retriever._retriever_supports_mode(mock_sparse, "vector") is False

    def test_get_retriever_for_mode_with_fixed_retriever(self, mock_chunk_retriever):
        """测试使用固定检索器获取检索器"""
        retriever = GraphRetriever(chunk_retriever=mock_chunk_retriever)
        
        with patch.object(retriever, "_retriever_supports_mode", return_value=True):
            result = retriever._get_retriever_for_mode("vector", is_chunk=True)
            assert result == mock_chunk_retriever

    def test_get_retriever_for_mode_unsupported(self, mock_chunk_retriever):
        """测试固定检索器不支持的模式"""
        retriever = GraphRetriever(chunk_retriever=mock_chunk_retriever)
        
        with patch.object(retriever, "_retriever_supports_mode", return_value=False):
            with pytest.raises(ValueError, match="does not support mode"):
                retriever._get_retriever_for_mode("sparse", is_chunk=True)

    def test_get_retriever_for_mode_no_vector_store(self):
        """测试没有向量存储时动态创建检索器"""
        retriever = GraphRetriever()
        
        with pytest.raises(ValueError, match="vector_store is required"):
            retriever._get_retriever_for_mode("vector", is_chunk=True)

    def test_get_retriever_for_mode_no_collection(self, mock_vector_store):
        """测试没有集合名称时动态创建检索器"""
        retriever = GraphRetriever(vector_store=mock_vector_store)
        
        with pytest.raises(ValueError, match="collection is required"):
            retriever._get_retriever_for_mode("sparse", is_chunk=True)

    def test_get_retriever_for_mode_vector_no_embed_model(self, mock_vector_store):
        """测试向量模式但没有嵌入模型"""
        retriever = GraphRetriever(
            vector_store=mock_vector_store,
            chunk_collection="chunks",
        )
        
        with pytest.raises(ValueError, match="embed_model is required"):
            retriever._get_retriever_for_mode("vector", is_chunk=True)

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

