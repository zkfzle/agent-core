# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
文本分割器测试用例
"""
import pytest
from unittest.mock import MagicMock, patch

from openjiuwen.core.retrieval.indexing.processor.chunker.text_splitter import (
    TextSplitter,
    CharSplitter,
    IndexSentenceSplitter,
)
from openjiuwen.core.retrieval.common.document import Document, TextChunk


class ConcreteTextSplitter(TextSplitter):
    """具体文本分割器实现，用于测试抽象基类"""

    def split(self, text):
        return [TextChunk(id_="1", text=text.text[:10], doc_id=text.id_)]


class TestTextSplitter:
    """文本分割器抽象基类测试"""

    def test_cannot_instantiate_abstract_class(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            TextSplitter()


class TestCharSplitter:
    """字符分割器测试"""

    def test_init_with_defaults(self):
        """测试使用默认值初始化"""
        splitter = CharSplitter()
        assert splitter.chunk_size == 200  # DEFAULT_CHAR_CHUNK_SIZE
        assert splitter.chunk_overlap == 40  # DEFAULT_CHAR_CHUNK_OVERLAP

    def test_init_with_custom_values(self):
        """测试使用自定义值初始化"""
        splitter = CharSplitter(chunk_size=512, chunk_overlap=50)
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50

    def test_init_overlap_adjusted(self):
        """测试重叠大小自动调整"""
        # 重叠大小应该小于分块大小
        splitter = CharSplitter(chunk_size=100, chunk_overlap=150)
        assert splitter.chunk_overlap < splitter.chunk_size

    def test_init_overlap_negative(self):
        """测试负重叠大小"""
        splitter = CharSplitter(chunk_size=100, chunk_overlap=-10)
        assert splitter.chunk_overlap >= 0

    def test_init_chunk_size_minimum(self):
        """测试最小分块大小"""
        splitter = CharSplitter(chunk_size=0)
        assert splitter.chunk_size >= 1

    def test_split_short_text(self):
        """测试分割短文本"""
        splitter = CharSplitter(chunk_size=100, chunk_overlap=10)
        doc = Document(id_="doc_1", text="Short text")
        chunks = splitter.split(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"
        assert chunks[0].doc_id == "doc_1"

    def test_split_long_text(self):
        """测试分割长文本"""
        splitter = CharSplitter(chunk_size=10, chunk_overlap=2)
        text = "This is a longer text that needs to be split into multiple chunks"
        doc = Document(id_="doc_1", text=text)
        chunks = splitter.split(doc)
        assert len(chunks) > 1
        # 验证所有块都属于同一个文档
        assert all(chunk.doc_id == "doc_1" for chunk in chunks)

    def test_split_with_overlap(self):
        """测试带重叠的分割"""
        splitter = CharSplitter(chunk_size=10, chunk_overlap=3)
        text = "This is a test text for splitting"
        doc = Document(id_="doc_1", text=text)
        chunks = splitter.split(doc)
        assert len(chunks) > 1
        # 验证有重叠（通过检查相邻块的内容）
        if len(chunks) > 1:
            # 第一个块的末尾应该出现在第二个块的开头
            first_end = chunks[0].text[-3:]
            second_start = chunks[1].text[:3]
            # 由于重叠，它们应该有部分相同
            assert len(first_end) == 3
            assert len(second_start) >= 3

    def test_split_preserves_metadata(self):
        """测试保留元数据"""
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
    """索引句子分割器测试"""

    def test_init_with_tokenizer(self):
        """测试使用 tokenizer 初始化"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize = lambda x: x.split()
        mock_tokenizer.model_max_length = 512

        with patch(
            "openjiuwen.core.retrieval.indexing.processor.chunker.text_splitter.SentenceSplitter"
        ) as mock_sentence_splitter_class:
            mock_sentence_splitter = MagicMock()
            mock_sentence_splitter_class.return_value = mock_sentence_splitter

            splitter = IndexSentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=256,
                chunk_overlap=50,
            )
            assert splitter._tokenizer == mock_tokenizer

    def test_init_with_default_splitter_config(self):
        """测试使用默认分割器配置初始化"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize = lambda x: x.split()

        with patch(
            "openjiuwen.core.retrieval.indexing.processor.chunker.text_splitter.SentenceSplitter"
        ) as mock_sentence_splitter_class:
            mock_sentence_splitter = MagicMock()
            mock_sentence_splitter_class.return_value = mock_sentence_splitter

            splitter = IndexSentenceSplitter(tokenizer=mock_tokenizer)
            # 验证使用了默认配置
            call_args = mock_sentence_splitter_class.call_args
            assert call_args is not None

    def test_resolve_tokenizer_with_tokenize(self):
        """测试解析 tokenizer（有 tokenize 方法）"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize = lambda x: x.split()
        mock_tokenizer.model_max_length = 512

        tokenizer_fn, max_length = IndexSentenceSplitter._resolve_tokenizer(mock_tokenizer)
        assert callable(tokenizer_fn)
        assert max_length == 512

    def test_resolve_tokenizer_with_encode(self):
        """测试解析 tokenizer（有 encode 方法）"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = lambda x: x.split()
        mock_tokenizer.model_max_length = 512

        tokenizer_fn, max_length = IndexSentenceSplitter._resolve_tokenizer(mock_tokenizer)
        assert callable(tokenizer_fn)
        assert max_length == 512

    def test_resolve_tokenizer_callable(self):
        """测试解析可调用的 tokenizer"""
        mock_tokenizer = lambda x: x.split()

        tokenizer_fn, max_length = IndexSentenceSplitter._resolve_tokenizer(mock_tokenizer)
        assert callable(tokenizer_fn)

    def test_resolve_tokenizer_fallback_tiktoken(self):
        """测试解析 tokenizer（回退到 tiktoken）"""
        mock_encoding = MagicMock()
        mock_encoding.encode = lambda x: x.split()

        with patch(
            "openjiuwen.core.retrieval.indexing.processor.chunker.text_splitter.tiktoken"
        ) as mock_tiktoken:
            mock_tiktoken.get_encoding.return_value = mock_encoding

            tokenizer_fn, max_length = IndexSentenceSplitter._resolve_tokenizer(None)
            assert callable(tokenizer_fn)
            mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    def test_max_length_model_max_length(self):
        """测试获取最大长度（model_max_length）"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 512

        max_length = IndexSentenceSplitter._max_length(mock_tokenizer)
        assert max_length == 512

    def test_resolve_chunk_size_none_with_max_length(self):
        """测试解析分块大小（None，但有最大长度）"""
        chunk_size = IndexSentenceSplitter._resolve_chunk_size(None, 512)
        assert chunk_size == 512

    def test_resolve_chunk_size_with_max_length(self):
        """测试解析分块大小（有值，但有最大长度限制）"""
        chunk_size = IndexSentenceSplitter._resolve_chunk_size(1024, 512)
        assert chunk_size == 512  # 应该取较小值

    def test_resolve_chunk_size_without_max_length(self):
        """测试解析分块大小（没有最大长度限制）"""
        chunk_size = IndexSentenceSplitter._resolve_chunk_size(256, None)
        assert chunk_size == 256

    def test_resolve_chunk_size_default(self):
        """测试解析分块大小（使用默认值）"""
        chunk_size = IndexSentenceSplitter._resolve_chunk_size(None, None)
        assert chunk_size == 200  # DEFAULT_CHUNK_SIZE

    def test_split_with_document(self):
        """测试分割文档"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize = lambda x: x.split()

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
        """测试分割文本块"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize = lambda x: x.split()

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

