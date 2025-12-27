# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
文本分块器测试用例
"""
import pytest
from unittest.mock import MagicMock, patch

from openjiuwen.core.retrieval.indexing.processor.chunker.chunking import TextChunker
from openjiuwen.core.retrieval.common.document import Document


class TestTextChunker:
    """文本分块器测试"""

    def test_init_with_char_unit(self):
        """测试使用字符单位初始化"""
        chunker = TextChunker(chunk_size=512, chunk_overlap=50, chunk_unit="char")
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert isinstance(chunker.chunker, type(chunker.get_chunker(512, 50, "char", None)))

    def test_init_with_token_unit_no_tiktoken(self):
        """测试使用 token 单位但 tiktoken 不可用"""
        with patch("openjiuwen.core.retrieval.indexing.processor.chunker.chunking.tiktoken", None):
            with pytest.raises(ValueError, match="requires embed_model with tokenizer or tiktoken"):
                TextChunker(
                    chunk_size=512,
                    chunk_overlap=50,
                    chunk_unit="token",
                )

    def test_init_with_preprocess_options(self):
        """测试使用预处理选项初始化"""
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
        """测试使用空白标准化初始化"""
        chunker = TextChunker(
            chunk_size=512,
            chunk_overlap=50,
            preprocess_options={"normalize_whitespace": True},
        )
        assert len(chunker.pipeline.preprocessors) == 1

    def test_init_with_remove_url_email(self):
        """测试使用 URL/邮箱移除初始化"""
        chunker = TextChunker(
            chunk_size=512,
            chunk_overlap=50,
            preprocess_options={"remove_url_email": True},
        )
        assert len(chunker.pipeline.preprocessors) == 1

    def test_init_without_preprocess_options(self):
        """测试不使用预处理选项初始化"""
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        assert len(chunker.pipeline.preprocessors) == 0

    def test_chunk_documents_with_preprocessing(self):
        """测试分块文档（带预处理）"""
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
        # 验证空白已标准化
        assert "   " not in chunks[0].text
        assert all(chunk.doc_id == "doc_1" for chunk in chunks)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)

    def test_chunk_documents_without_preprocessing(self):
        """测试分块文档（不带预处理）"""
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
        """测试分块多个文档"""
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
        """测试保留元数据"""
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
        """测试获取字符分块器"""
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        result = chunker.get_chunker(512, 50, "char", None)
        from openjiuwen.core.retrieval.indexing.processor.chunker.char_chunker import CharChunker
        assert isinstance(result, CharChunker)

    def test_get_chunker_token_unit_adjusts_size(self):
        """测试 token 分块器自动调整大小"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 256
        mock_embed_model = MagicMock()
        mock_embed_model.tokenizer = mock_tokenizer

        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        result = chunker.get_chunker(512, 50, "token", mock_embed_model)
        # chunk_size 应该被调整为 256
        assert result.chunk_size == 256

