# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ChromaDB 索引管理器测试用例
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import shutil

from openjiuwen.core.retrieval.indexing.indexer.chroma_indexer import ChromaIndexer
from openjiuwen.core.retrieval.common.config import IndexConfig
from openjiuwen.core.retrieval.common.document import TextChunk


@pytest.fixture
def temp_chroma_path():
    """创建临时 ChromaDB 路径"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embed_model():
    """创建模拟嵌入模型"""
    model = AsyncMock()
    model.embed_documents = AsyncMock(return_value=[[0.1] * 384] * 2)
    return model


class TestChromaIndexer:
    """ChromaDB 索引管理器测试"""

    def test_init_success(self, temp_chroma_path):
        """测试初始化成功"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        assert indexer.chroma_path == temp_chroma_path
        assert indexer.client is not None

    def test_init_empty_path(self):
        """测试空路径初始化"""
        with pytest.raises(ValueError, match="chroma_path is required"):
            ChromaIndexer(chroma_path="")

    def test_init_whitespace_path(self):
        """测试空白路径初始化"""
        with pytest.raises(ValueError, match="chroma_path is required"):
            ChromaIndexer(chroma_path="   ")

    def test_init_with_custom_fields(self, temp_chroma_path):
        """测试使用自定义字段初始化"""
        indexer = ChromaIndexer(
            chroma_path=temp_chroma_path,
            text_field="custom_text",
            vector_field="custom_vector",
            doc_id_field="custom_doc_id",
        )
        assert indexer.text_field == "custom_text"
        assert indexer.vector_field == "custom_vector"
        assert indexer.doc_id_field == "custom_doc_id"

    @pytest.mark.asyncio
    async def test_build_index_vector_type(self, temp_chroma_path, mock_embed_model):
        """测试构建向量索引"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        chunks = [
            TextChunk(id_="1", text="chunk 1", doc_id="doc_1"),
            TextChunk(id_="2", text="chunk 2", doc_id="doc_1"),
        ]
        config = IndexConfig(index_name="test_index", index_type="vector")
        
        with patch("openjiuwen.core.retrieval.indexing.indexer.chroma_indexer.ChromaVectorStore") as mock_store_class:
            mock_store = AsyncMock()
            mock_store.add = AsyncMock()
            mock_store_class.return_value = mock_store

            result = await indexer.build_index(chunks, config, mock_embed_model)
            assert result is True
            mock_embed_model.embed_documents.assert_called_once()
            mock_store.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_index_bm25_type(self, temp_chroma_path):
        """测试构建 BM25 索引"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        chunks = [
            TextChunk(id_="1", text="chunk 1", doc_id="doc_1"),
        ]
        config = IndexConfig(index_name="test_index", index_type="bm25")
        
        with patch("openjiuwen.core.retrieval.indexing.indexer.chroma_indexer.ChromaVectorStore") as mock_store_class:
            mock_store = AsyncMock()
            mock_store.add = AsyncMock()
            mock_store_class.return_value = mock_store

            result = await indexer.build_index(chunks, config)
            assert result is True
            mock_store.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_index_hybrid_type(self, temp_chroma_path, mock_embed_model):
        """测试构建混合索引"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        chunks = [
            TextChunk(id_="1", text="chunk 1", doc_id="doc_1"),
        ]
        config = IndexConfig(index_name="test_index", index_type="hybrid")
        
        with patch("openjiuwen.core.retrieval.indexing.indexer.chroma_indexer.ChromaVectorStore") as mock_store_class:
            mock_store = AsyncMock()
            mock_store.add = AsyncMock()
            mock_store_class.return_value = mock_store

            result = await indexer.build_index(chunks, config, mock_embed_model)
            assert result is True
            mock_embed_model.embed_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_index_vector_type_without_embed_model(self, temp_chroma_path):
        """测试向量索引但没有嵌入模型"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        chunks = [TextChunk(id_="1", text="chunk 1", doc_id="doc_1")]
        config = IndexConfig(index_name="test_index", index_type="vector")
        
        result = await indexer.build_index(chunks, config)
        assert result is False  # 应该失败

    @pytest.mark.asyncio
    async def test_build_index_with_exception(self, temp_chroma_path):
        """测试构建索引时发生异常"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        chunks = [TextChunk(id_="1", text="chunk 1", doc_id="doc_1")]
        config = IndexConfig(index_name="test_index", index_type="bm25")
        
        with patch("openjiuwen.core.retrieval.indexing.indexer.chroma_indexer.ChromaVectorStore") as mock_store_class:
            mock_store = AsyncMock()
            mock_store.add = AsyncMock(side_effect=Exception("Store error"))
            mock_store_class.return_value = mock_store

            result = await indexer.build_index(chunks, config)
            assert result is False

    @pytest.mark.asyncio
    async def test_update_index(self, temp_chroma_path, mock_embed_model):
        """测试更新索引"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
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
    async def test_delete_index_success(self, temp_chroma_path):
        """测试删除索引成功"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        
        # 先创建一个集合
        collection = indexer.client.get_or_create_collection(name="test_index")
        collection.add(
            ids=["1", "2"],
            documents=["chunk 1", "chunk 2"],
            metadatas=[{"document_id": "doc_1"}, {"document_id": "doc_1"}],
        )

        result = await indexer.delete_index("doc_1", "test_index")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_index_not_found(self, temp_chroma_path):
        """测试删除不存在的索引"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        result = await indexer.delete_index("doc_1", "nonexistent_index")
        # 应该返回 False（集合不存在或没有匹配的记录）
        assert result is False

    @pytest.mark.asyncio
    async def test_index_exists_true(self, temp_chroma_path):
        """测试索引存在"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        # 创建集合
        indexer.client.get_or_create_collection(name="test_index")
        
        result = await indexer.index_exists("test_index")
        assert result is True

    @pytest.mark.asyncio
    async def test_index_exists_false(self, temp_chroma_path):
        """测试索引不存在"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        result = await indexer.index_exists("nonexistent_index")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_index_info_not_exists(self, temp_chroma_path):
        """测试获取不存在的索引信息"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        info = await indexer.get_index_info("nonexistent_index")
        assert info["exists"] is False

    def test_close(self, temp_chroma_path):
        """测试关闭索引管理器"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        # 不应该抛出异常
        indexer.close()

