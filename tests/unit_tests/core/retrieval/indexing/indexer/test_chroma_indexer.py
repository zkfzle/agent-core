# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ChromaDB index manager test cases
"""
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openjiuwen.core.retrieval.indexing.indexer.chroma_indexer import ChromaIndexer
from openjiuwen.core.retrieval.common.config import IndexConfig
from openjiuwen.core.retrieval.common.document import TextChunk


@pytest.fixture
def temp_chroma_path():
    """Create temporary ChromaDB path"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embed_model():
    """Create mock embedding model"""
    model = AsyncMock()
    model.embed_documents = AsyncMock(return_value=[[0.1] * 384] * 2)
    return model


class TestChromaIndexer:
    """ChromaDB index manager tests"""

    @classmethod
    def test_init_success(cls, temp_chroma_path):
        """Test successful initialization"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        assert indexer.chroma_path == temp_chroma_path
        assert indexer.client is not None

    @staticmethod
    def test_init_empty_path():
        """Test initialization with empty path"""
        with pytest.raises(ValueError, match="chroma_path is required"):
            ChromaIndexer(chroma_path="")

    @staticmethod
    def test_init_whitespace_path():
        """Test initialization with whitespace path"""
        with pytest.raises(ValueError, match="chroma_path is required"):
            ChromaIndexer(chroma_path="   ")

    @classmethod
    def test_init_with_custom_fields(cls, temp_chroma_path):
        """Test initialization with custom fields"""
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
        """Test building vector index"""
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
        """Test building BM25 index"""
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
        """Test building hybrid index"""
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
        """Test vector index but without embedding model"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        chunks = [TextChunk(id_="1", text="chunk 1", doc_id="doc_1")]
        config = IndexConfig(index_name="test_index", index_type="vector")
        
        result = await indexer.build_index(chunks, config)
        assert result is False  # Should fail

    @pytest.mark.asyncio
    async def test_build_index_with_exception(self, temp_chroma_path):
        """Test exception during index building"""
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
        """Test updating index"""
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
        """Test deleting index successfully"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        
        # First create a collection
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
        """Test deleting non-existent index"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        result = await indexer.delete_index("doc_1", "nonexistent_index")
        # Should return False (collection does not exist or no matching records)
        assert result is False

    @pytest.mark.asyncio
    async def test_index_exists_true(self, temp_chroma_path):
        """Test index exists"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        # Create collection
        indexer.client.get_or_create_collection(name="test_index")
        
        result = await indexer.index_exists("test_index")
        assert result is True

    @pytest.mark.asyncio
    async def test_index_exists_false(self, temp_chroma_path):
        """Test index does not exist"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        result = await indexer.index_exists("nonexistent_index")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_index_info_not_exists(self, temp_chroma_path):
        """Test getting non-existent index information"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        info = await indexer.get_index_info("nonexistent_index")
        assert info["exists"] is False

    @classmethod
    def test_close(cls, temp_chroma_path):
        """Test closing index manager"""
        indexer = ChromaIndexer(chroma_path=temp_chroma_path)
        # Should not raise exception
        indexer.close()

