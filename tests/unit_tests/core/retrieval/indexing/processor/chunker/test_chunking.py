# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Text chunker test cases
"""
import pytest
from unittest.mock import MagicMock, patch

from openjiuwen.core.retrieval.indexing.processor.chunker.chunking import TextChunker
from openjiuwen.core.retrieval.common.document import Document


class TestTextChunker:
    """Text chunker tests"""

    def test_init_with_char_unit(self):
        """Test initialization with character unit"""
        chunker = TextChunker(chunk_size=512, chunk_overlap=50, chunk_unit="char")
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert isinstance(chunker.chunker, type(chunker.get_chunker(512, 50, "char", None)))

    def test_init_with_token_unit_no_tiktoken(self):
        """Test initialization with token unit but tiktoken unavailable"""
        with patch("openjiuwen.core.retrieval.indexing.processor.chunker.chunking.tiktoken", None):
            with pytest.raises(ValueError, match="requires embed_model with tokenizer or tiktoken"):
                TextChunker(
                    chunk_size=512,
                    chunk_overlap=50,
                    chunk_unit="token",
                )

    def test_init_with_preprocess_options(self):
        """Test initialization with preprocess options"""
        chunker = TextChunker(
            chunk_size=512,
            chunk_overlap=50,
            preprocess_options={
                "normalize_whitespace": True,
                "remove_url_email": True,
            },
        )
        assert len(chunker.pipeline.preprocessors) == 2

    def test_init_with_normalize_whitespace(self):
        """Test initialization with whitespace normalization"""
        chunker = TextChunker(
            chunk_size=512,
            chunk_overlap=50,
            preprocess_options={"normalize_whitespace": True},
        )
        assert len(chunker.pipeline.preprocessors) == 1

    def test_init_with_remove_url_email(self):
        """Test initialization with URL/email removal"""
        chunker = TextChunker(
            chunk_size=512,
            chunk_overlap=50,
            preprocess_options={"remove_url_email": True},
        )
        assert len(chunker.pipeline.preprocessors) == 1

    def test_init_without_preprocess_options(self):
        """Test initialization without preprocess options"""
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        assert len(chunker.pipeline.preprocessors) == 0

    def test_chunk_documents_with_preprocessing(self):
        """Test chunking documents (with preprocessing)"""
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=10,
            preprocess_options={"normalize_whitespace": True},
        )
        documents = [
            Document(
                id_="doc_1",
                text="This   is   document   1",
                metadata={"source": "test"},
            ),
        ]
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) > 0
        # Verify whitespace is normalized
        assert "   " not in chunks[0].text
        assert all(chunk.doc_id == "doc_1" for chunk in chunks)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)

    def test_chunk_documents_without_preprocessing(self):
        """Test chunking documents (without preprocessing)"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        documents = [
            Document(
                id_="doc_1",
                text="This is document 1",
                metadata={"source": "test"},
            ),
        ]
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) > 0
        assert all(chunk.doc_id == "doc_1" for chunk in chunks)

    def test_chunk_documents_multiple_docs(self):
        """Test chunking multiple documents"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        documents = [
            Document(id_="doc_1", text="This is document 1"),
            Document(id_="doc_2", text="This is document 2"),
        ]
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) > 0
        doc_ids = {chunk.doc_id for chunk in chunks}
        assert "doc_1" in doc_ids
        assert "doc_2" in doc_ids

    def test_chunk_documents_preserves_metadata(self):
        """Test preserving metadata"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        documents = [
            Document(
                id_="doc_1",
                text="This is document 1",
                metadata={"source": "test", "author": "test_author"},
            ),
        ]
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) > 0
        assert all("source" in chunk.metadata for chunk in chunks)
        assert all(chunk.metadata["source"] == "test" for chunk in chunks)
        assert all("author" in chunk.metadata for chunk in chunks)

    def test_get_chunker_char_unit(self):
        """Test getting character chunker"""
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        result = chunker.get_chunker(512, 50, "char", None)
        from openjiuwen.core.retrieval.indexing.processor.chunker.char_chunker import CharChunker
        assert isinstance(result, CharChunker)

    def test_get_chunker_token_unit_adjusts_size(self):
        """Test token chunker automatically adjusts size"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 256
        mock_embed_model = MagicMock()
        mock_embed_model.tokenizer = mock_tokenizer

        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        result = chunker.get_chunker(512, 50, "token", mock_embed_model)
        # chunk_size should be adjusted to 256
        assert result.chunk_size == 256

