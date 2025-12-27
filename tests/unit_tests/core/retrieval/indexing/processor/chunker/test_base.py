# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Text chunker abstract base class test cases
"""
from unittest.mock import MagicMock

import pytest

from openjiuwen.core.retrieval.indexing.processor.chunker.base import Chunker
from openjiuwen.core.retrieval.common.document import Document


class ConcreteChunker(Chunker):
    """Concrete chunker implementation for testing abstract base class"""

    def chunk_text(self, text: str):
        # Simple chunking implementation: one chunk per 10 characters
        chunks = []
        for i in range(0, len(text), 10):
            chunks.append(text[i:i + 10])
        return chunks


class TestChunker:
    """Text chunker abstract base class tests"""

    def test_init_with_defaults(self):
        """Test initialization with default values"""
        chunker = ConcreteChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.length_function == len

    def test_init_with_custom_values(self):
        """Test initialization with custom values"""
        custom_length_fn = lambda x: len(x.split())
        chunker = ConcreteChunker(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=custom_length_fn,
        )
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 100
        assert chunker.length_function == custom_length_fn

    def test_init_invalid_overlap(self):
        """Test invalid overlap size"""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ConcreteChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_text(self):
        """Test chunking text"""
        chunker = ConcreteChunker()
        text = "This is a test text for chunking"
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_documents(self):
        """Test chunking document list"""
        chunker = ConcreteChunker()
        documents = [
            Document(id_="doc_1", text="This is document 1"),
            Document(id_="doc_2", text="This is document 2"),
        ]
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) > 0
        assert all(chunk.doc_id in ["doc_1", "doc_2"] for chunk in chunks)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)
        assert all("total_chunks" in chunk.metadata for chunk in chunks)

    def test_chunk_documents_with_metadata(self):
        """Test chunking documents with metadata"""
        chunker = ConcreteChunker()
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

    @pytest.mark.asyncio
    async def test_process(self):
        """Test processing documents (implements Processor interface)"""
        chunker = ConcreteChunker()
        documents = [Document(id_="doc_1", text="Test document")]
        chunks = await chunker.process(documents)
        assert len(chunks) > 0
        assert all(chunk.doc_id == "doc_1" for chunk in chunks)

