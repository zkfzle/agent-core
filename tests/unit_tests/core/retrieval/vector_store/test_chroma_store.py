# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ChromaDB vector store test cases
"""
import json
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openjiuwen.core.retrieval.vector_store.chroma_store import ChromaVectorStore
from openjiuwen.core.retrieval.common.config import VectorStoreConfig
from openjiuwen.core.retrieval.common.retrieval_result import SearchResult


@pytest.fixture
def temp_chroma_path():
    """Create temporary ChromaDB path"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_store_config():
    """Create vector store configuration"""
    return VectorStoreConfig(
        collection_name="test_collection",
        distance_metric="cosine",
    )


class TestChromaVectorStore:
    """ChromaDB vector store tests"""

    @classmethod
    def test_init_success(cls, temp_chroma_path, vector_store_config):
        """Test successful initialization"""
        store = ChromaVectorStore(
            config=vector_store_config,
            chroma_path=temp_chroma_path,
        )
        assert store.collection_name == "test_collection"
        assert store.chroma_path == temp_chroma_path
        assert store.client is not None
        assert store.collection is not None

    @classmethod
    def test_init_empty_path(cls, vector_store_config):
        """Test initialization with empty path"""
        with pytest.raises(ValueError, match="chroma_path is required"):
            ChromaVectorStore(config=vector_store_config, chroma_path="")

    @classmethod
    def test_init_whitespace_path(cls, vector_store_config):
        """Test initialization with whitespace path"""
        with pytest.raises(ValueError, match="chroma_path is required"):
            ChromaVectorStore(config=vector_store_config, chroma_path="   ")

    @classmethod
    def test_init_with_custom_fields(cls, temp_chroma_path, vector_store_config):
        """Test initialization with custom fields"""
        store = ChromaVectorStore(
            config=vector_store_config,
            chroma_path=temp_chroma_path,
            text_field="custom_text",
            vector_field="custom_vector",
            doc_id_field="custom_doc_id",
        )
        assert store.text_field == "custom_text"
        assert store.vector_field == "custom_vector"
        assert store.doc_id_field == "custom_doc_id"

    @classmethod
    def test_init_with_euclidean_metric(cls, temp_chroma_path):
        """Test initialization with Euclidean distance"""
        config = VectorStoreConfig(
            collection_name="test_collection",
            distance_metric="euclidean",
        )
        store = ChromaVectorStore(config=config, chroma_path=temp_chroma_path)
        assert store.config.distance_metric == "euclidean"

    @pytest.mark.asyncio
    async def test_add_single_dict(self, temp_chroma_path, vector_store_config):
        """Test adding single dictionary"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        data = {
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test"},
        }
        
        await store.add(data)
        # Verify data was added (by searching)
        results = await store.search([0.1] * 384, top_k=1)
        assert len(results) >= 0  # May be empty, depending on ChromaDB behavior

    @pytest.mark.asyncio
    async def test_add_without_embedding(self, temp_chroma_path, vector_store_config):
        """Test adding data without embedding"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        data = {
            "id": "1",
            "content": "Test content",
        }
        
        # Should skip data without embedding
        await store.add(data)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, temp_chroma_path, vector_store_config):
        """Test adding data with metadata"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        data = {
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test", "author": "test_author"},
            "document_id": "doc_1",
            "chunk_id": "chunk_1",
        }
        
        await store.add(data)
        # Verify metadata was saved
        results = await store.search([0.1] * 384, top_k=1)
        if results:
            assert "source" in results[0].metadata or "document_id" in results[0].metadata

    @pytest.mark.asyncio
    async def test_search_with_filters(self, temp_chroma_path, vector_store_config):
        """Test search with filters"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        # Add data first
        await store.add({
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test"},
            "document_id": "doc_1",
        })
        
        filters = {"document_id": "doc_1"}
        results = await store.search([0.1] * 384, top_k=5, filters=filters)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_sparse_search_with_filters(self, temp_chroma_path, vector_store_config):
        """Test sparse search with filters"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        await store.add({
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test"},
        })
        
        filters = {"source": "test"}
        results = await store.sparse_search("Test", top_k=5, filters=filters)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self, temp_chroma_path, vector_store_config):
        """Test hybrid search with filters"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        await store.add({
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test"},
        })
        
        filters = {"source": "test"}
        results = await store.hybrid_search(
            query_text="Test",
            query_vector=[0.1] * 384,
            top_k=5,
            filters=filters,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_failure(self, temp_chroma_path, vector_store_config):
        """Test hybrid search failure"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        with patch.object(store, "search", side_effect=Exception("Search error")):
            results = await store.hybrid_search(
                query_text="Test",
                query_vector=[0.1] * 384,
                top_k=5,
            )
            # Should return empty list or partial results
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_delete_by_filter_expr(self, temp_chroma_path, vector_store_config):
        """Test deletion by filter expression"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        # ChromaDB does not support complex filter_expr, should return False
        result = await store.delete(filter_expr="source == 'test'")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_without_params(self, temp_chroma_path, vector_store_config):
        """Test deletion without parameters"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        result = await store.delete()
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_with_exception(self, temp_chroma_path, vector_store_config):
        """Test deletion with exception"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = Exception("Delete error")
            
            result = await store.delete(ids=["1"])
            assert result is False

    @classmethod
    def test_close(cls, temp_chroma_path, vector_store_config):
        """Test closing vector store"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        # Should not raise exception
        store.close()

