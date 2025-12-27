# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
文本分割器抽象基类测试用例
"""
import pytest
from unittest.mock import MagicMock

from openjiuwen.core.retrieval.indexing.processor.spliter.base import Splitter
from openjiuwen.core.retrieval.common.document import Document, TextChunk


class ConcreteSplitter(Splitter):
    """具体分割器实现，用于测试抽象基类"""

    def __call__(self, doc: str):
        # 简单的分割实现：每 10 个字符一个块
        chunks = []
        for i in range(0, len(doc), 10):
            start = i
            end = min(i + 10, len(doc))
            chunks.append((doc[start:end], start, end))
        return chunks


class TestSplitter:
    """文本分割器抽象基类测试"""

    def test_init_with_defaults(self):
        """测试使用默认值初始化"""
        splitter = ConcreteSplitter()
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50
        assert splitter.tokenizer is None
        assert splitter.tokenizer_enc is None
        assert splitter.tokenizer_dec is None

    def test_init_with_custom_values(self):
        """测试使用自定义值初始化"""
        splitter = ConcreteSplitter(chunk_size=1024, chunk_overlap=100)
        assert splitter.chunk_size == 1024
        assert splitter.chunk_overlap == 100

    def test_init_with_tokenizer(self):
        """测试使用 tokenizer 初始化"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = lambda x: x.split()
        mock_tokenizer.decode = lambda x: " ".join(x)

        splitter = ConcreteSplitter(tokenizer=mock_tokenizer)
        assert splitter.tokenizer == mock_tokenizer
        assert splitter.tokenizer_enc is not None
        assert splitter.tokenizer_dec is not None

    def test_init_with_callable_tokenizer(self):
        """测试使用可调用的 tokenizer 初始化"""
        def tokenizer_func(text):
            return text.split()

        splitter = ConcreteSplitter(tokenizer=tokenizer_func)
        assert splitter.tokenizer == tokenizer_func
        assert splitter.tokenizer_enc == tokenizer_func
        assert splitter.tokenizer_dec is None

    def test_init_invalid_chunk_size(self):
        """测试无效的分块大小"""
        with pytest.raises(ValueError, match="chunk_size 必须大于 0"):
            ConcreteSplitter(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size 必须大于 0"):
            ConcreteSplitter(chunk_size=-1)

    def test_init_invalid_chunk_overlap(self):
        """测试无效的重叠大小"""
        with pytest.raises(ValueError, match="chunk_overlap 必须大于等于 0"):
            ConcreteSplitter(chunk_overlap=-1)

    def test_init_overlap_greater_than_size(self):
        """测试重叠大小大于等于分块大小"""
        with pytest.raises(ValueError, match="chunk_overlap.*必须小于 chunk_size"):
            ConcreteSplitter(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap.*必须小于 chunk_size"):
            ConcreteSplitter(chunk_size=100, chunk_overlap=150)

    def test_call(self):
        """测试调用分割方法"""
        splitter = ConcreteSplitter()
        text = "This is a test text for splitting"
        chunks = splitter(text)
        assert len(chunks) > 0
        assert all(isinstance(chunk, tuple) and len(chunk) == 3 for chunk in chunks)
        assert all(isinstance(chunk[0], str) for chunk in chunks)
        assert all(isinstance(chunk[1], int) for chunk in chunks)
        assert all(isinstance(chunk[2], int) for chunk in chunks)

    def test_get_nodes_from_documents(self):
        """测试从文档列表获取节点"""
        splitter = ConcreteSplitter()
        documents = [
            Document(id_="doc_1", text="This is document 1"),
            Document(id_="doc_2", text="This is document 2"),
        ]
        nodes = splitter.get_nodes_from_documents(documents)
        assert len(nodes) > 0
        assert all(isinstance(node, TextChunk) for node in nodes)
        assert all(node.doc_id in ["doc_1", "doc_2"] for node in nodes)

    def test_get_nodes_from_documents_empty_doc(self):
        """测试从空文档获取节点"""
        splitter = ConcreteSplitter()
        documents = [
            Document(id_="doc_1", text=""),
            Document(id_="doc_2", text="This is document 2"),
        ]
        nodes = splitter.get_nodes_from_documents(documents)
        # 空文档应该被跳过
        assert len(nodes) > 0
        assert all(node.doc_id == "doc_2" for node in nodes)

    def test_get_nodes_from_documents_none_doc(self):
        """测试从 None 文档获取节点"""
        splitter = ConcreteSplitter()
        documents = [None, Document(id_="doc_2", text="This is document 2")]
        nodes = splitter.get_nodes_from_documents(documents)
        # None 文档应该被跳过
        assert len(nodes) > 0
        assert all(node.doc_id == "doc_2" for node in nodes)

    def test_split_text(self):
        """测试分割文本（仅返回文本列表）"""
        splitter = ConcreteSplitter()
        text = "This is a test text"
        chunks = splitter.split_text(text)
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert len(chunks) > 0

    def test_get_token_count_with_tokenizer(self):
        """测试使用 tokenizer 获取 token 数量"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = lambda x: x.split()

        splitter = ConcreteSplitter(tokenizer=mock_tokenizer)
        count = splitter._get_token_count("This is a test")
        assert count == 4  # 4 个单词

    def test_get_token_count_without_tokenizer(self):
        """测试不使用 tokenizer 获取 token 数量（返回字符数）"""
        splitter = ConcreteSplitter()
        count = splitter._get_token_count("This is a test")
        assert count == len("This is a test")  # 字符数

    def test_get_token_count_with_list_tokens(self):
        """测试获取列表形式的 token 数量"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = lambda x: x.split()

        splitter = ConcreteSplitter(tokenizer=mock_tokenizer)
        count = splitter._get_token_count("test")
        assert isinstance(count, int)
        assert count > 0

    def test_cannot_instantiate_abstract_class(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            Splitter()

