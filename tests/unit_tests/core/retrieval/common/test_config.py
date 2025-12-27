# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Configuration class test cases
"""
import pytest
from pydantic import ValidationError

from openjiuwen.core.retrieval.common.config import (
    KnowledgeBaseConfig,
    RetrievalConfig,
    IndexConfig,
    VectorStoreConfig,
    EmbeddingConfig,
)


class TestKnowledgeBaseConfig:
    """Knowledge base configuration tests"""

    def test_create_with_defaults(self):
        """Test creating configuration with default values"""
        config = KnowledgeBaseConfig(kb_id="test_kb")
        assert config.kb_id == "test_kb"
        assert config.index_type == "hybrid"
        assert config.use_graph is False
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_create_with_custom_values(self):
        """Test creating configuration with custom values"""
        config = KnowledgeBaseConfig(
            kb_id="test_kb",
            index_type="vector",
            use_graph=True,
            chunk_size=1024,
            chunk_overlap=100,
        )
        assert config.kb_id == "test_kb"
        assert config.index_type == "vector"
        assert config.use_graph is True
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100

    def test_invalid_index_type(self):
        """Test invalid index type"""
        with pytest.raises(ValidationError):
            KnowledgeBaseConfig(kb_id="test_kb", index_type="invalid")

    def test_missing_kb_id(self):
        """Test missing required kb_id"""
        with pytest.raises(ValidationError):
            KnowledgeBaseConfig()


class TestRetrievalConfig:
    """Retrieval configuration tests"""

    def test_create_with_defaults(self):
        """Test creating configuration with default values"""
        config = RetrievalConfig()
        assert config.top_k == 5
        assert config.score_threshold is None
        assert config.use_graph is None
        assert config.agentic is False
        assert config.graph_expansion is False
        assert config.filters is None

    def test_create_with_custom_values(self):
        """Test creating configuration with custom values"""
        config = RetrievalConfig(
            top_k=10,
            score_threshold=0.7,
            use_graph=True,
            agentic=True,
            graph_expansion=True,
            filters={"doc_id": "test"},
        )
        assert config.top_k == 10
        assert config.score_threshold == 0.7
        assert config.use_graph is True
        assert config.agentic is True
        assert config.graph_expansion is True
        assert config.filters == {"doc_id": "test"}


class TestIndexConfig:
    """Index configuration tests"""

    def test_create_with_defaults(self):
        """Test creating configuration with default values"""
        config = IndexConfig(index_name="test_index")
        assert config.index_name == "test_index"
        assert config.index_type == "hybrid"

    def test_create_with_custom_values(self):
        """Test creating configuration with custom values"""
        config = IndexConfig(index_name="test_index", index_type="vector")
        assert config.index_name == "test_index"
        assert config.index_type == "vector"

    def test_invalid_index_type(self):
        """Test invalid index type"""
        with pytest.raises(ValidationError):
            IndexConfig(index_name="test_index", index_type="invalid")

    def test_missing_index_name(self):
        """Test missing required index_name"""
        with pytest.raises(ValidationError):
            IndexConfig()


class TestVectorStoreConfig:
    """Vector store configuration tests"""

    def test_create_with_defaults(self):
        """Test creating configuration with default values"""
        config = VectorStoreConfig(collection_name="test_collection")
        assert config.collection_name == "test_collection"
        assert config.distance_metric == "cosine"

    def test_create_with_custom_values(self):
        """Test creating configuration with custom values"""
        config = VectorStoreConfig(
            collection_name="test_collection",
            distance_metric="euclidean",
        )
        assert config.collection_name == "test_collection"
        assert config.distance_metric == "euclidean"

    def test_invalid_distance_metric(self):
        """Test invalid distance metric"""
        with pytest.raises(ValidationError):
            VectorStoreConfig(
                collection_name="test_collection",
                distance_metric="invalid",
            )

    def test_missing_collection_name(self):
        """Test missing required collection_name"""
        with pytest.raises(ValidationError):
            VectorStoreConfig()


class TestEmbeddingConfig:
    """Embedding model configuration tests"""

    def test_create_with_required_fields(self):
        """Test creating configuration with required fields"""
        config = EmbeddingConfig(model_name="test_model")
        assert config.model_name == "test_model"
        assert config.api_key is None
        assert config.base_url is None

    def test_create_with_all_fields(self):
        """Test creating configuration with all fields"""
        config = EmbeddingConfig(
            model_name="test_model",
            api_key="test_key",
            base_url="https://api.example.com",
        )
        assert config.model_name == "test_model"
        assert config.api_key == "test_key"
        assert config.base_url == "https://api.example.com"

    def test_missing_model_name(self):
        """Test missing required model_name"""
        with pytest.raises(ValidationError):
            EmbeddingConfig()

