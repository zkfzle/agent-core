# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Milvus 索引管理器测试用例
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openjiuwen.core.retrieval.indexing.indexer.milvus_indexer import MilvusIndexer
from openjiuwen.core.retrieval.common.config import IndexConfig
from openjiuwen.core.retrieval.common.document import TextChunk


@pytest.fixture
def mock_embed_model():
    """创建模拟嵌入模型"""
    model = AsyncMock()
    model.embed_documents = AsyncMock(return_value=[[0.1] * 384] * 2)
    model.dimension = 384
    return model


class TestMilvusIndexer:
    """Milvus 索引管理器测试"""

    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    def test_init_success(self, mock_client_class):
        """测试初始化成功"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        assert indexer.milvus_uri == "http://localhost:19530"
        assert indexer.client == mock_client
        mock_client_class.assert_called_once_with(
            uri="http://localhost:19530",
            token=None,
        )

    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    def test_init_with_token(self, mock_client_class):
        """测试使用 Token 初始化"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(
            milvus_uri="http://localhost:19530",
            milvus_token="test_token",
        )
        assert indexer.milvus_token == "test_token"
        mock_client_class.assert_called_once_with(
            uri="http://localhost:19530",
            token="test_token",
        )

    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    def test_init_with_custom_fields(self, mock_client_class):
        """测试使用自定义字段初始化"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(
            milvus_uri="http://localhost:19530",
            text_field="custom_text",
            vector_field="custom_vector",
            doc_id_field="custom_doc_id",
        )
        assert indexer.text_field == "custom_text"
        assert indexer.vector_field == "custom_vector"
        assert indexer.doc_id_field == "custom_doc_id"

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusVectorStore")
    async def test_build_index_vector_type(self, mock_store_class, mock_client_class, mock_embed_model):
        """测试构建向量索引"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_store = AsyncMock()
        mock_store.add = AsyncMock()
        mock_store_class.return_value = mock_store

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        chunks = [
            TextChunk(id_="1", text="chunk 1", doc_id="doc_1"),
            TextChunk(id_="2", text="chunk 2", doc_id="doc_1"),
        ]
        config = IndexConfig(index_name="test_index", index_type="vector")

        with patch.object(indexer, "_ensure_collection", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = None
            result = await indexer.build_index(chunks, config, mock_embed_model)
            assert result is True
            mock_embed_model.embed_documents.assert_called_once()
            mock_store.add.assert_called_once()

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusVectorStore")
    async def test_build_index_bm25_type(self, mock_store_class, mock_client_class):
        """测试构建 BM25 索引"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_store = AsyncMock()
        mock_store.add = AsyncMock()
        mock_store_class.return_value = mock_store

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        chunks = [TextChunk(id_="1", text="chunk 1", doc_id="doc_1")]
        config = IndexConfig(index_name="test_index", index_type="bm25")

        with patch.object(indexer, "_ensure_collection", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = None
            result = await indexer.build_index(chunks, config)
            assert result is True
            mock_store.add.assert_called_once()

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    async def test_build_index_vector_type_without_embed_model(self, mock_client_class):
        """测试向量索引但没有嵌入模型"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        chunks = [TextChunk(id_="1", text="chunk 1", doc_id="doc_1")]
        config = IndexConfig(index_name="test_index", index_type="vector")

        with patch.object(indexer, "_ensure_collection", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = None
            result = await indexer.build_index(chunks, config)
            assert result is False  # 应该失败

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    async def test_update_index(self, mock_client_class, mock_embed_model):
        """测试更新索引"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        chunks = [TextChunk(id_="1", text="updated chunk", doc_id="doc_1")]
        config = IndexConfig(index_name="test_index", index_type="vector")

        with patch.object(indexer, "delete_index", new_callable=AsyncMock) as mock_delete, \
             patch.object(indexer, "build_index", new_callable=AsyncMock) as mock_build:
            mock_delete.return_value = True
            mock_build.return_value = True

            result = await indexer.update_index(chunks, "doc_1", config, mock_embed_model)
            assert result is True
            mock_delete.assert_called_once()
            mock_build.assert_called_once()

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    async def test_delete_index_success(self, mock_client_class):
        """测试删除索引成功"""
        mock_client = MagicMock()
        mock_client.delete.return_value = {"delete_count": 2}
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        result = await indexer.delete_index("doc_1", "test_index")
        assert result is True

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    async def test_delete_index_not_found(self, mock_client_class):
        """测试删除不存在的索引"""
        mock_client = MagicMock()
        mock_client.delete.return_value = {"delete_count": 0}
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        result = await indexer.delete_index("doc_1", "test_index")
        assert result is False

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    async def test_index_exists_true(self, mock_client_class):
        """测试索引存在"""
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        result = await indexer.index_exists("test_index")
        assert result is True

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    async def test_index_exists_false(self, mock_client_class):
        """测试索引不存在"""
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        result = await indexer.index_exists("nonexistent_index")
        assert result is False

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    async def test_get_index_info_exists(self, mock_client_class):
        """测试获取存在的索引信息"""
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.get_collection_stats.return_value = {"row_count": 100}
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        info = await indexer.get_index_info("test_index")
        assert info["exists"] is True
        assert info["collection_name"] == "test_index"
        assert "count" in info

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    async def test_get_index_info_not_exists(self, mock_client_class):
        """测试获取不存在的索引信息"""
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        info = await indexer.get_index_info("nonexistent_index")
        assert info["exists"] is False

    @patch("openjiuwen.core.retrieval.indexing.indexer.milvus_indexer.MilvusClient")
    def test_close(self, mock_client_class):
        """测试关闭索引管理器"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        indexer = MilvusIndexer(milvus_uri="http://localhost:19530")
        # 不应该抛出异常
        indexer.close()

