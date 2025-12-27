# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
文本分块器抽象基类测试用例
"""
import pytest
from unittest.mock import MagicMock

from openjiuwen.core.retrieval.indexing.processor.chunker.base import Chunker
from openjiuwen.core.retrieval.common.document import Document


class ConcreteChunker(Chunker):
    """具体分块器实现，用于测试抽象基类"""

    def chunk_text(self, text: str):
        # 简单的分块实现：每 10 个字符一个块
        chunks = []
        for i in range(0, len(text), 10):
            chunks.append(text[i:i + 10])
        return chunks


class TestChunker:
    """文本分块器抽象基类测试"""

    def test_init_with_defaults(self):
        """测试使用默认值初始化"""
        chunker = ConcreteChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.length_function == len

    def test_init_with_custom_values(self):
        """测试使用自定义值初始化"""
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
        """测试无效的重叠大小"""
        with pytest.raises(ValueError, match="chunk_overlap 必须小于 chunk_size"):
            ConcreteChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_text(self):
        """测试分块文本"""
        chunker = ConcreteChunker()
        text = "This is a test text for chunking"
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_documents(self):
        """测试分块文档列表"""
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
        """测试分块带元数据的文档"""
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
        """测试处理文档（实现 Processor 接口）"""
        chunker = ConcreteChunker()
        documents = [Document(id_="doc_1", text="Test document")]
        chunks = await chunker.process(documents)
        assert len(chunks) > 0
        assert all(chunk.doc_id == "doc_1" for chunk in chunks)

