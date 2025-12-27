# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
句子分割器测试用例
"""
import pytest
from unittest.mock import MagicMock, patch

from openjiuwen.core.retrieval.indexing.processor.spliter.splitter import SentenceSplitter


@pytest.fixture
def mock_tokenizer():
    """创建模拟 tokenizer"""
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: x.split()
    tokenizer.decode = lambda x: " ".join(x)
    return tokenizer


@pytest.fixture
def mock_segmenter():
    """创建模拟句子分割器"""
    segmenter = MagicMock()
    return segmenter


class TestSentenceSplitter:
    """句子分割器测试"""

    def test_init_with_defaults(self, mock_tokenizer):
        """测试使用默认值初始化"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            assert splitter.chunk_size == 512
            assert splitter.chunk_overlap == 50
            assert splitter.tokenizer == mock_tokenizer
            mock_segmenter_class.assert_called_once_with(language="zh", clean=False)

    def test_init_with_custom_language(self, mock_tokenizer):
        """测试使用自定义语言初始化"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
                lan="en",
            )
            mock_segmenter_class.assert_called_once_with(language="en", clean=False)

    def test_call_empty_text(self, mock_tokenizer):
        """测试分割空文本"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            chunks = splitter("")
            assert chunks == []

    def test_call_whitespace_only(self, mock_tokenizer):
        """测试分割仅包含空白字符的文本"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            chunks = splitter("   \n\t   ")
            assert chunks == []

    def test_call_single_sentence(self, mock_tokenizer):
        """测试分割单个句子"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter.segment.return_value = ["This is a test sentence."]
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            chunks = splitter("This is a test sentence.")
            assert len(chunks) == 1
            assert chunks[0][0] == "This is a test sentence."
            assert chunks[0][1] == 0
            assert chunks[0][2] == len("This is a test sentence.")

    def test_call_multiple_sentences(self, mock_tokenizer):
        """测试分割多个句子"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter.segment.return_value = [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
            ]
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            text = "First sentence. Second sentence. Third sentence."
            chunks = splitter(text)
            assert len(chunks) >= 1
            # 验证所有句子都被处理
            all_text = " ".join(chunk[0] for chunk in chunks)
            assert "First sentence" in all_text
            assert "Second sentence" in all_text
            assert "Third sentence" in all_text

    def test_call_long_sentence(self, mock_tokenizer):
        """测试分割超长句子"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            long_sentence = " ".join(["word"] * 1000)  # 很长的句子
            mock_segmenter.segment.return_value = [long_sentence]
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=100,  # 较小的分块大小
                chunk_overlap=10,
            )
            chunks = splitter(long_sentence)
            # 超长句子应该被单独作为一个块
            assert len(chunks) >= 1

    def test_sentences_with_spans(self, mock_tokenizer):
        """测试获取带位置的句子"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter.segment.return_value = [
                "First sentence.",
                "Second sentence.",
            ]
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            text = "First sentence. Second sentence."
            spans = splitter._sentences_with_spans(text)
            assert len(spans) == 2
            assert all(isinstance(span, tuple) and len(span) == 3 for span in spans)
            assert spans[0][0] == "First sentence."
            assert spans[1][0] == "Second sentence."

    def test_sentences_with_spans_empty_sentences(self, mock_tokenizer):
        """测试获取带位置的句子（包含空句子）"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter.segment.return_value = [
                "First sentence.",
                "   ",  # 空句子
                "Second sentence.",
            ]
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            text = "First sentence.    Second sentence."
            spans = splitter._sentences_with_spans(text)
            # 空句子应该被跳过
            assert len(spans) == 2

    def test_sentences_with_spans_not_found(self, mock_tokenizer):
        """测试句子在文本中找不到的情况"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            # 返回一个在文本中找不到的句子
            mock_segmenter.segment.return_value = ["Not in text"]
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            text = "Different text"
            spans = splitter._sentences_with_spans(text)
            # 找不到的句子应该被跳过
            assert len(spans) == 0

    def test_flush_no_sentences(self, mock_tokenizer):
        """测试刷新空句子列表"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            chunks, next_sents = splitter._flush([], [])
            assert chunks == []
            assert next_sents == []

    def test_flush_with_sentences(self, mock_tokenizer):
        """测试刷新句子列表"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=50,
            )
            cur_sents = [
                ("First sentence.", 0, 16),
                ("Second sentence.", 17, 33),
            ]
            chunks, next_sents = splitter._flush([], cur_sents)
            assert len(chunks) == 1
            assert chunks[0][0] == "First sentence. Second sentence."
            assert chunks[0][1] == 0
            assert chunks[0][2] == 33

    def test_flush_with_overlap(self, mock_tokenizer):
        """测试刷新时处理重叠"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=10,  # 设置重叠
            )
            cur_sents = [
                ("First sentence.", 0, 16),
                ("Second sentence.", 17, 33),
                ("Third sentence.", 34, 49),
            ]
            chunks, next_sents = splitter._flush([], cur_sents)
            assert len(chunks) == 1
            # 如果有重叠，next_sents 应该包含一些句子
            if splitter.chunk_overlap > 0:
                # 重叠逻辑会根据 token 数量决定保留哪些句子
                assert isinstance(next_sents, list)

    def test_flush_with_overlap_zero(self, mock_tokenizer):
        """测试重叠为 0 时的刷新"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=512,
                chunk_overlap=0,  # 无重叠
            )
            cur_sents = [
                ("First sentence.", 0, 16),
                ("Second sentence.", 17, 33),
            ]
            chunks, next_sents = splitter._flush([], cur_sents)
            assert len(chunks) == 1
            assert next_sents == []  # 无重叠时应该为空

    def test_call_combines_sentences(self, mock_tokenizer):
        """测试合并句子到块中"""
        with patch("openjiuwen.core.retrieval.indexing.processor.spliter.splitter.Segmenter") as mock_segmenter_class:
            mock_segmenter = MagicMock()
            mock_segmenter.segment.return_value = [
                "Short sentence 1.",
                "Short sentence 2.",
                "Short sentence 3.",
            ]
            mock_segmenter_class.return_value = mock_segmenter

            splitter = SentenceSplitter(
                tokenizer=mock_tokenizer,
                chunk_size=100,  # 足够大的分块大小
                chunk_overlap=10,
            )
            text = "Short sentence 1. Short sentence 2. Short sentence 3."
            chunks = splitter(text)
            # 如果分块大小足够，应该合并多个句子
            assert len(chunks) >= 1
            # 验证句子被合并
            if len(chunks) == 1:
                assert "Short sentence 1" in chunks[0][0]
                assert "Short sentence 2" in chunks[0][0]
                assert "Short sentence 3" in chunks[0][0]

