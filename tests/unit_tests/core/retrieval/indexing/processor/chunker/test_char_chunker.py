# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
固定大小分块器测试用例
"""
import pytest
from unittest.mock import patch, MagicMock

from openjiuwen.core.retrieval.indexing.processor.chunker.char_chunker import CharChunker
from openjiuwen.core.retrieval.common.document import Document, TextChunk


class TestCharChunker:
    """固定大小分块器测试"""

    def test_init_with_defaults(self):
        """测试使用默认值初始化"""
        chunker = CharChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50

    def test_init_with_custom_values(self):
        """测试使用自定义值初始化"""
        chunker = CharChunker(chunk_size=256, chunk_overlap=25)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 25

    @patch("openjiuwen.core.retrieval.indexing.processor.chunker.char_chunker.CharSplitter")
    def test_chunk_text_success(self, mock_splitter_class):
        """测试分块文本成功"""
        # 模拟 CharSplitter
        mock_splitter = MagicMock()
        mock_node1 = TextChunk(id_="1", text="chunk 1", doc_id="doc_1")
        mock_node2 = TextChunk(id_="2", text="chunk 2", doc_id="doc_1")
        mock_splitter.split.return_value = [mock_node1, mock_node2]
        mock_splitter_class.return_value = mock_splitter

        chunker = CharChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk_text("This is a test text")
        assert len(chunks) == 2
        assert chunks[0] == "chunk 1"
        assert chunks[1] == "chunk 2"
        mock_splitter_class.assert_called_once_with(
            chunk_size=10, chunk_overlap=2
        )

    def test_chunk_text_empty(self):
        """测试分块空文本"""
        chunker = CharChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_chunk_text_none(self):
        """测试分块 None"""
        chunker = CharChunker()
        chunks = chunker.chunk_text(None)
        assert chunks == []

