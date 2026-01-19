# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
OpenAI embedding model implementation test cases
"""

from unittest.mock import Mock, patch

import httpx
import openai
import pytest
from openai.types import Embedding

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.retrieval import EmbeddingConfig, OpenAIEmbedding


@pytest.fixture
def embedding_config():
    """Create embedding model configuration"""
    return EmbeddingConfig(
        model_name="test-model",
        api_key="test-api-key",
        base_url="https://api.example.com/v1/embeddings",
    )


@pytest.fixture
def embedding_config_no_key():
    """Create embedding model configuration without API key"""
    return EmbeddingConfig(
        model_name="test-model",
        base_url="https://api.example.com/v1/embeddings",
    )


class TestOpenAIEmbedding:
    """API embedding model tests"""

    @classmethod
    def test_init_with_api_key(cls, embedding_config):
        """Test initialization with API key"""
        model = OpenAIEmbedding(config=embedding_config)
        assert model.model_name == "test-model"
        assert model.api_key == "test-api-key"
        assert model.api_url == "https://api.example.com/v1"

    @classmethod
    def test_init_without_api_key(cls, embedding_config_no_key):
        """Test initialization without API key"""
        model = OpenAIEmbedding(config=embedding_config_no_key)
        assert model.api_key is None

    @classmethod
    def test_init_with_extra_headers(cls, embedding_config):
        """Test initialization with extra headers"""
        extra_headers = {"X-Custom-Header": "custom-value"}
        model = OpenAIEmbedding(config=embedding_config, extra_headers=extra_headers)

    @classmethod
    def test_init_with_custom_params(cls, embedding_config):
        """Test initialization with custom parameters"""
        model = OpenAIEmbedding(
            config=embedding_config,
            timeout=120,
            max_retries=5,
            max_batch_size=16,
        )
        assert model.timeout == 120
        assert model.max_retries == 5
        assert model.max_batch_size == 16

    @pytest.mark.asyncio
    async def test_embed_query_success_embedding_format(self, embedding_config):
        """Test embedding query text successfully (embedding format)"""
        mock_response = Mock(data=[Mock(embedding=[0.1] * 384)])

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            model = OpenAIEmbedding(config=embedding_config)
            embedding = await model.embed_query("test query")
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_query_success_embeddings_format(self, embedding_config):
        """Test embedding query text successfully (embeddings format)"""
        mock_response = Mock(data=[Mock(embedding=[0.1] * 384)])

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            model = OpenAIEmbedding(config=embedding_config)
            embedding = await model.embed_query("test query")
            assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_embed_query_success_data_format(self, embedding_config):
        """Test embedding query text successfully (data format)"""
        mock_response = Mock(data=[Mock(embedding=[0.1] * 384)])

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            model = OpenAIEmbedding(config=embedding_config)
            embedding = await model.embed_query("test query")
            assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_embed_query_empty_text(self, embedding_config):
        """Test embedding empty text"""
        model = OpenAIEmbedding(config=embedding_config)
        with pytest.raises(JiuWenBaseException, match="Empty text provided"):
            await model.embed_query("   ")

    @pytest.mark.asyncio
    async def test_embed_query_retry_on_failure(self, embedding_config):
        """Test retry on request failure"""
        mock_request = httpx.Request("get", "")

        with patch("asyncio.to_thread") as mock_to_thread:
            # First attempt fails, second succeeds
            mock_to_thread.side_effect = [
                openai.APIConnectionError(message="Connection error", request=mock_request),
                Mock(data=[Mock(embedding=[0.1] * 384)]),
            ]

            model = OpenAIEmbedding(config=embedding_config, max_retries=3)
            embedding = await model.embed_query("test query")
            assert len(embedding) == 384
            assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_query_max_retries_exceeded(self, embedding_config):
        """Test exceeding maximum retry count"""
        mock_request = httpx.Request("get", "")

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = openai.APIConnectionError(message="Connection error", request=mock_request)
            model = OpenAIEmbedding(config=embedding_config, max_retries=2)
            with pytest.raises(JiuWenBaseException, match="Failed to get embedding"):
                await model.embed_query("test query")
            assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_query_invalid_response_format(self, embedding_config):
        """Test invalid response format"""
        mock_response = Mock(data={"invalid": "format"})

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            model = OpenAIEmbedding(config=embedding_config)
            with pytest.raises(JiuWenBaseException, match="No embeddings in response"):
                await model.embed_query("test query")

    @pytest.mark.asyncio
    async def test_embed_documents_success(self, embedding_config):
        """Test embedding document list successfully"""
        mock_data = [Embedding(index=-1, embedding=[x / 10] * 384, object="embedding") for x in range(1, 4)]
        mock_response = Mock(data=mock_data)

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            model = OpenAIEmbedding(config=embedding_config)
            texts = ["text 1", "text 2", "text 3"]
            embeddings = await model.embed_documents(texts)
            assert len(embeddings) == 3
            assert all(len(emb) == 384 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_embed_documents_with_batching(self, embedding_config):
        """Test batch embedding documents"""
        mock_data = [Embedding(index=-1, embedding=[0.1] * 384, object="embedding")]
        mock_response = Mock(data=mock_data)

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            model = OpenAIEmbedding(config=embedding_config, max_batch_size=1)
            texts = ["text 1", "text 2", "text 3", "text 4"]
            embeddings = await model.embed_documents(texts, batch_size=1)
            # Should process in 4 batches
            assert mock_to_thread.call_count == 4
            assert len(embeddings) == 4

    @pytest.mark.asyncio
    async def test_embed_documents_respects_max_batch_size(self, embedding_config):
        """Test respecting maximum batch size"""
        mock_data = [Embedding(index=-1, embedding=[0.1] * 384, object="embedding")]
        mock_response = Mock(data=mock_data)

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            model = OpenAIEmbedding(config=embedding_config, max_batch_size=2)
            texts = ["text 1", "text 2", "text 3"]
            # Even if batch_size=5 is specified, it should be limited to max_batch_size=2
            embeddings = await model.embed_documents(texts, batch_size=5)
            # Should process in 2 batches (2 + 1)
            assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_documents_empty_list(self, embedding_config):
        """Test embedding empty list"""
        model = OpenAIEmbedding(config=embedding_config)
        with pytest.raises(JiuWenBaseException, match="Empty texts list provided"):
            await model.embed_documents([])

    @pytest.mark.asyncio
    async def test_embed_documents_with_empty_texts(self, embedding_config):
        """Test list containing empty texts"""
        model = OpenAIEmbedding(config=embedding_config)
        with pytest.raises(JiuWenBaseException, match="chunks are empty"):
            await model.embed_documents(["text 1", "   ", "text 2"])

    @pytest.mark.asyncio
    async def test_embed_documents_all_empty(self, embedding_config):
        """Test all texts are empty"""
        model = OpenAIEmbedding(config=embedding_config)
        with pytest.raises(JiuWenBaseException):
            await model.embed_documents(["   ", "  ", ""])

    @pytest.mark.asyncio
    async def test_dimension_from_response(self, embedding_config):
        """Test getting dimension from response"""
        mock_response = Mock(data=[Mock(embedding=[0.1] * 768)])

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            model = OpenAIEmbedding(config=embedding_config)
            # First call embed_query to set dimension
            await model.embed_query("test")
            # Dimension should have been retrieved from response
            assert model.dimension == 768
