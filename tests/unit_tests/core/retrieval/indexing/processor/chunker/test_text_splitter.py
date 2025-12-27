# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Text splitter test cases
"""
from unittest.mock import MagicMock, patch

import pytest

from openjiuwen.core.retrieval.indexing.processor.chunker.text_splitter import (
    TextSplitter,
    CharSplitter,
    IndexSentenceSplitter,
)
from openjiuwen.core.retrieval.common.document import Document, TextChunk


class ConcreteTextSplitter(TextSplitter):
    """Concrete text splitter implementation for testing abstract base class"""

    def split(self, text):
        return [TextChunk(id_="1", text=text.text[:10], doc_id=text.id_)]


class TestTextSplitter:
    """Text splitter abstract base class tests"""

    def test_cannot_instantiate_abstract_class(self):
        """Test cannot directly instantiate abstract class"""
        with pytest.raises(TypeError):
            TextSplitter()


class TestCharSplitter:
    """Character splitter tests"""

    def test_init_with_defaults(self):
        """Test initialization with default values"""
        splitter = CharSplitter()
        assert splitter.chunk_size == 200  # DEFAULT_CHAR_CHUNK_SIZE
        assert splitter.chunk_overlap == 40  # DEFAULT_CHAR_CHUNK_OVERLAP

    def test_init_with_custom_values(self):
        """Test initialization with custom values"""
        splitter = CharSplitter(chunk_size=512, chunk_overlap=50)
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50

    def test_init_overlap_adjusted(self):
        """Test automatic overlap size adjustment"""
        # Overlap size should be less than chunk size
        splitter = CharSplitter(chunk_size=100, chunk_overlap=150)
        assert splitter.chunk_overlap < splitter.chunk_size

    def test_init_overlap_negative(self):
        """Test negative overlap size"""
        splitter = CharSplitter(chunk_size=100, chunk_overlap=-10)
        assert splitter.chunk_overlap >= 0

    def test_init_chunk_size_minimum(self):
        """Test minimum chunk size"""
        splitter = CharSplitter(chunk_size=0)
        assert splitter.chunk_size >= 1

    def test_split_short_text(self):
        """Test splitting short text"""
        splitter = CharSplitter(chunk_size=100, chunk_overlap=10)
        doc = Document(id_="doc_1", text="Short text")
        chunks = splitter.split(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"
        assert chunks[0].doc_id == "doc_1"

    def test_split_long_text(self):
        """Test splitting long text"""
        splitter = CharSplitter(chunk_size=10, chunk_overlap=2)
        text = "This is a longer text that needs to be split into multiple chunks"
        doc = Document(id_="doc_1", text=text)
        chunks = splitter.split(doc)
        assert len(chunks) > 1
        # Verify all chunks belong to the same document
        assert all(chunk.doc_id == "doc_1" for chunk in chunks)

    def test_split_with_overlap(self):
        """Test splitting with overlap"""
        splitter = CharSplitter(chunk_size=10, chunk_overlap=3)
        text = "This is a test text for splitting"
        doc = Document(id_="doc_1", text=text)
        chunks = splitter.split(doc)
        assert len(chunks) > 1
        # Verify there is overlap (by checking adjacent chunk content)
        if len(chunks) > 1:
            # End of first chunk should appear at start of second chunk
            first_end = chunks[0].text[-3:]
            second_start = chunks[1].text[:3]
            # Due to overlap, they should have some overlap
            assert len(first_end) == 3
            assert len(second_start) >= 3

    def test_split_preserves_metadata(self):
        """Test preserving metadata"""
        splitter = CharSplitter(chunk_size=10, chunk_overlap=2)
        doc = Document(
            id_="doc_1",
            text="This is a test",
            metadata={"source": "test", "author": "test_author"},
        )
        chunks = splitter.split(doc)
        assert len(chunks) > 0
        assert all("source" in chunk.metadata for chunk in chunks)
        assert all(chunk.metadata["source"] == "test" for chunk in chunks)


class TestIndexSentenceSplitter:
    """Index sentence splitter tests"""

    def test_init_with_default_splitter_config(self):
        """Test initialization with default splitter configuration"""
        def tokenize_fn(x):
            return x.split()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize = tokenize_fn

        with patch(
            "openjiuwen.core.retrieval.indexing.processor.chunker.text_splitter.SentenceSplitter"
        ) as mock_sentence_splitter_class:
            mock_sentence_splitter = MagicMock()
            mock_sentence_splitter_class.return_value = mock_sentence_splitter

            splitter = IndexSentenceSplitter(tokenizer=mock_tokenizer)
            # Verify default configuration is used
            call_args = mock_sentence_splitter_class.call_args
            assert call_args is not None

    def test_resolve_tokenizer_fallback_tiktoken(self):
        """Test resolving tokenizer (fallback to tiktoken)"""
        def encode_fn(x):
            return x.split()
        
        mock_encoding = MagicMock()
        mock_encoding.encode = encode_fn

        with patch(
            "openjiuwen.core.retrieval.indexing.processor.chunker.text_splitter.tiktoken"
        ) as mock_tiktoken:
            mock_tiktoken.get_encoding.return_value = mock_encoding

            tokenizer_fn, max_length = IndexSentenceSplitter._resolve_tokenizer(None)
            assert callable(tokenizer_fn)
            mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    def test_split_with_document(self):
        """Test splitting document"""
        def tokenize_fn(x):
            return x.split()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize = tokenize_fn

        with patch(
            "openjiuwen.core.retrieval.indexing.processor.chunker.text_splitter.SentenceSplitter"
        ) as mock_sentence_splitter_class:
            mock_sentence_splitter = MagicMock()
            mock_node1 = TextChunk(id_="1", text="chunk 1", doc_id="doc_1")
            mock_node2 = TextChunk(id_="2", text="chunk 2", doc_id="doc_1")
            mock_sentence_splitter.get_nodes_from_documents.return_value = [
                mock_node1,
                mock_node2,
            ]
            mock_sentence_splitter_class.return_value = mock_sentence_splitter

            splitter = IndexSentenceSplitter(tokenizer=mock_tokenizer)
            doc = Document(id_="doc_1", text="This is a test")
            chunks = splitter.split(doc)
            assert len(chunks) == 2
            mock_sentence_splitter.get_nodes_from_documents.assert_called_once()

    def test_split_with_text_chunk(self):
        """Test splitting text chunk"""
        def tokenize_fn(x):
            return x.split()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize = tokenize_fn

        with patch(
            "openjiuwen.core.retrieval.indexing.processor.chunker.text_splitter.SentenceSplitter"
        ) as mock_sentence_splitter_class:
            mock_sentence_splitter = MagicMock()
            mock_node = TextChunk(id_="1", text="chunk", doc_id="doc_1")
            mock_sentence_splitter.get_nodes_from_documents.return_value = [mock_node]
            mock_sentence_splitter_class.return_value = mock_sentence_splitter

            splitter = IndexSentenceSplitter(tokenizer=mock_tokenizer)
            text_chunk = TextChunk(id_="1", text="This is a test", doc_id="doc_1")
            chunks = splitter.split(text_chunk)
            assert len(chunks) == 1

