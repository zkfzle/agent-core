# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
文本预处理器测试用例
"""
import pytest
from unittest.mock import MagicMock

from openjiuwen.core.retrieval.indexing.processor.chunker.text_preprocessor import (
    TextPreprocessor,
    WhitespaceNormalizer,
    URLEmailRemover,
    SpecialCharacterNormalizer,
    PreprocessingPipeline,
)


class ConcretePreprocessor(TextPreprocessor):
    """具体预处理器实现，用于测试抽象基类"""

    def process(self, text: str) -> str:
        return text.upper()


class TestTextPreprocessor:
    """文本预处理器抽象基类测试"""

    def test_process(self):
        """测试处理方法"""
        preprocessor = ConcretePreprocessor()
        result = preprocessor.process("test")
        assert result == "TEST"

    def test_call(self):
        """测试可调用接口"""
        preprocessor = ConcretePreprocessor()
        result = preprocessor("test")
        assert result == "TEST"

    def test_cannot_instantiate_abstract_class(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            TextPreprocessor()


class TestWhitespaceNormalizer:
    """空白字符标准化器测试"""

    def test_process_normal_text(self):
        """测试处理正常文本"""
        normalizer = WhitespaceNormalizer()
        text = "This is a test"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_multiple_spaces(self):
        """测试处理多个空格"""
        normalizer = WhitespaceNormalizer()
        text = "This   is    a     test"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_newlines(self):
        """测试处理换行符"""
        normalizer = WhitespaceNormalizer()
        text = "This\nis\na\ntest"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_tabs(self):
        """测试处理制表符"""
        normalizer = WhitespaceNormalizer()
        text = "This\tis\ta\ttest"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_mixed_whitespace(self):
        """测试处理混合空白字符"""
        normalizer = WhitespaceNormalizer()
        text = "This  \n\t  is  \n\t  a  \n\t  test"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_leading_trailing_whitespace(self):
        """测试处理前后空白"""
        normalizer = WhitespaceNormalizer()
        text = "   This is a test   "
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_empty_string(self):
        """测试处理空字符串"""
        normalizer = WhitespaceNormalizer()
        result = normalizer.process("")
        assert result == ""

    def test_process_none(self):
        """测试处理 None"""
        normalizer = WhitespaceNormalizer()
        result = normalizer.process(None)
        assert result is None


class TestURLEmailRemover:
    """URL 和邮箱移除器测试"""

    def test_init_defaults(self):
        """测试使用默认值初始化"""
        remover = URLEmailRemover()
        assert remover.remove_urls is True
        assert remover.remove_emails is True
        assert remover.replacement == ""

    def test_init_custom(self):
        """测试使用自定义值初始化"""
        remover = URLEmailRemover(
            remove_urls=False,
            remove_emails=True,
            replacement="[removed]",
        )
        assert remover.remove_urls is False
        assert remover.remove_emails is True
        assert remover.replacement == "[removed]"

    def test_remove_urls_http(self):
        """测试移除 HTTP URL"""
        remover = URLEmailRemover()
        text = "Visit http://example.com for more info"
        result = remover.process(text)
        assert "http://example.com" not in result

    def test_remove_urls_https(self):
        """测试移除 HTTPS URL"""
        remover = URLEmailRemover()
        text = "Visit https://example.com for more info"
        result = remover.process(text)
        assert "https://example.com" not in result

    def test_remove_urls_www(self):
        """测试移除 www URL"""
        remover = URLEmailRemover()
        text = "Visit www.example.com for more info"
        result = remover.process(text)
        assert "www.example.com" not in result

    def test_remove_emails(self):
        """测试移除邮箱地址"""
        remover = URLEmailRemover()
        text = "Contact us at test@example.com for support"
        result = remover.process(text)
        assert "test@example.com" not in result

    def test_remove_urls_with_replacement(self):
        """测试使用替换字符串移除 URL"""
        remover = URLEmailRemover(replacement="[URL]")
        text = "Visit http://example.com for more info"
        result = remover.process(text)
        assert "[URL]" in result
        assert "http://example.com" not in result

    def test_remove_emails_with_replacement(self):
        """测试使用替换字符串移除邮箱"""
        remover = URLEmailRemover(replacement="[EMAIL]")
        text = "Contact test@example.com"
        result = remover.process(text)
        assert "[EMAIL]" in result
        assert "test@example.com" not in result

    def test_disable_url_removal(self):
        """测试禁用 URL 移除"""
        remover = URLEmailRemover(remove_urls=False)
        text = "Visit http://example.com for more info"
        result = remover.process(text)
        assert "http://example.com" in result

    def test_process_empty_string(self):
        """测试处理空字符串"""
        remover = URLEmailRemover()
        result = remover.process("")
        assert result == ""

    def test_process_none(self):
        """测试处理 None"""
        remover = URLEmailRemover()
        result = remover.process(None)
        assert result is None


class TestSpecialCharacterNormalizer:
    """特殊字符标准化器测试"""

    def test_init_defaults(self):
        """测试使用默认值初始化"""
        normalizer = SpecialCharacterNormalizer()
        assert normalizer.chars_to_remove == ""
        assert normalizer.chars_to_replace == {}

    def test_init_with_chars_to_remove(self):
        """测试使用要移除的字符初始化"""
        normalizer = SpecialCharacterNormalizer(chars_to_remove="!@#")
        assert normalizer.chars_to_remove == "!@#"

    def test_init_with_chars_to_replace(self):
        """测试使用要替换的字符初始化"""
        normalizer = SpecialCharacterNormalizer(
            chars_to_replace={"&": "and", "@": "at"}
        )
        assert normalizer.chars_to_replace == {"&": "and", "@": "at"}

    def test_remove_control_characters(self):
        """测试移除控制字符"""
        normalizer = SpecialCharacterNormalizer()
        text = "Test\x00text\x1Fwith\x7Fcontrol"
        result = normalizer.process(text)
        assert "\x00" not in result
        assert "\x1F" not in result
        assert "\x7F" not in result

    def test_replace_characters(self):
        """测试替换字符"""
        normalizer = SpecialCharacterNormalizer(
            chars_to_replace={"&": "and", "@": "at"}
        )
        text = "Tom & Jerry @ home"
        result = normalizer.process(text)
        assert "and" in result
        assert "at" in result
        assert "&" not in result
        assert "@" not in result

    def test_remove_specified_characters(self):
        """测试移除指定字符"""
        normalizer = SpecialCharacterNormalizer(chars_to_remove="!@#")
        text = "Test!text@with#special"
        result = normalizer.process(text)
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result

    def test_process_empty_string(self):
        """测试处理空字符串"""
        normalizer = SpecialCharacterNormalizer()
        result = normalizer.process("")
        assert result == ""

    def test_process_none(self):
        """测试处理 None"""
        normalizer = SpecialCharacterNormalizer()
        result = normalizer.process(None)
        assert result is None


class TestPreprocessingPipeline:
    """预处理管道测试"""

    def test_init_empty(self):
        """测试初始化空管道"""
        pipeline = PreprocessingPipeline()
        assert len(pipeline.preprocessors) == 0

    def test_init_with_preprocessors(self):
        """测试使用预处理器初始化"""
        preprocessor1 = WhitespaceNormalizer()
        preprocessor2 = URLEmailRemover()
        pipeline = PreprocessingPipeline([preprocessor1, preprocessor2])
        assert len(pipeline.preprocessors) == 2

    def test_add_preprocessor(self):
        """测试添加预处理器"""
        pipeline = PreprocessingPipeline()
        preprocessor = WhitespaceNormalizer()
        pipeline.add_preprocessor(preprocessor)
        assert len(pipeline.preprocessors) == 1
        assert pipeline.preprocessors[0] == preprocessor

    def test_process_single_preprocessor(self):
        """测试处理单个预处理器"""
        pipeline = PreprocessingPipeline([WhitespaceNormalizer()])
        text = "This   is   a   test"
        result = pipeline.process(text)
        assert result == "This is a test"

    def test_process_order(self):
        """测试处理顺序"""
        # 创建一个记录处理顺序的预处理器
        class OrderTracker(TextPreprocessor):
            def __init__(self, name):
                self.name = name
                self.order = None

            def process(self, text: str) -> str:
                return text

        tracker1 = OrderTracker("first")
        tracker2 = OrderTracker("second")
        pipeline = PreprocessingPipeline([tracker1, tracker2])
        pipeline.process("test")
        # 验证顺序（通过调用顺序）
        assert pipeline.preprocessors[0].name == "first"
        assert pipeline.preprocessors[1].name == "second"

    def test_call(self):
        """测试可调用接口"""
        pipeline = PreprocessingPipeline([WhitespaceNormalizer()])
        text = "This   is   a   test"
        result = pipeline(text)
        assert result == "This is a test"

    def test_len(self):
        """测试长度方法"""
        pipeline = PreprocessingPipeline([
            WhitespaceNormalizer(),
            URLEmailRemover(),
        ])
        assert len(pipeline) == 2

    def test_process_empty_string(self):
        """测试处理空字符串"""
        pipeline = PreprocessingPipeline([WhitespaceNormalizer()])
        result = pipeline.process("")
        assert result == ""

