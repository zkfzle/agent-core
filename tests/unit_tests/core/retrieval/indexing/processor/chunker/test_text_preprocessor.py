# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Text preprocessor test cases
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
    """Concrete preprocessor implementation for testing abstract base class"""

    def process(self, text: str) -> str:
        return text.upper()


class TestTextPreprocessor:
    """Text preprocessor abstract base class tests"""

    def test_process(self):
        """Test process method"""
        preprocessor = ConcretePreprocessor()
        result = preprocessor.process("test")
        assert result == "TEST"

    def test_call(self):
        """Test callable interface"""
        preprocessor = ConcretePreprocessor()
        result = preprocessor("test")
        assert result == "TEST"

    def test_cannot_instantiate_abstract_class(self):
        """Test cannot directly instantiate abstract class"""
        with pytest.raises(TypeError):
            TextPreprocessor()


class TestWhitespaceNormalizer:
    """Whitespace normalizer tests"""

    def test_process_normal_text(self):
        """Test processing normal text"""
        normalizer = WhitespaceNormalizer()
        text = "This is a test"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_multiple_spaces(self):
        """Test processing multiple spaces"""
        normalizer = WhitespaceNormalizer()
        text = "This   is    a     test"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_newlines(self):
        """Test processing newlines"""
        normalizer = WhitespaceNormalizer()
        text = "This\nis\na\ntest"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_tabs(self):
        """Test processing tabs"""
        normalizer = WhitespaceNormalizer()
        text = "This\tis\ta\ttest"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_mixed_whitespace(self):
        """Test processing mixed whitespace"""
        normalizer = WhitespaceNormalizer()
        text = "This  \n\t  is  \n\t  a  \n\t  test"
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_leading_trailing_whitespace(self):
        """Test processing leading and trailing whitespace"""
        normalizer = WhitespaceNormalizer()
        text = "   This is a test   "
        result = normalizer.process(text)
        assert result == "This is a test"

    def test_process_empty_string(self):
        """Test processing empty string"""
        normalizer = WhitespaceNormalizer()
        result = normalizer.process("")
        assert result == ""

    def test_process_none(self):
        """Test processing None"""
        normalizer = WhitespaceNormalizer()
        result = normalizer.process(None)
        assert result is None


class TestURLEmailRemover:
    """URL and email remover tests"""

    def test_init_defaults(self):
        """Test initialization with default values"""
        remover = URLEmailRemover()
        assert remover.remove_urls is True
        assert remover.remove_emails is True
        assert remover.replacement == ""

    def test_init_custom(self):
        """Test initialization with custom values"""
        remover = URLEmailRemover(
            remove_urls=False,
            remove_emails=True,
            replacement="[removed]",
        )
        assert remover.remove_urls is False
        assert remover.remove_emails is True
        assert remover.replacement == "[removed]"

    def test_remove_urls_http(self):
        """Test removing HTTP URL"""
        remover = URLEmailRemover()
        text = "Visit http://example.com for more info"
        result = remover.process(text)
        assert "http://example.com" not in result

    def test_remove_urls_https(self):
        """Test removing HTTPS URL"""
        remover = URLEmailRemover()
        text = "Visit https://example.com for more info"
        result = remover.process(text)
        assert "https://example.com" not in result

    def test_remove_urls_www(self):
        """Test removing www URL"""
        remover = URLEmailRemover()
        text = "Visit www.example.com for more info"
        result = remover.process(text)
        assert "www.example.com" not in result

    def test_remove_emails(self):
        """Test removing email addresses"""
        remover = URLEmailRemover()
        text = "Contact us at test@example.com for support"
        result = remover.process(text)
        assert "test@example.com" not in result

    def test_remove_urls_with_replacement(self):
        """Test removing URL with replacement string"""
        remover = URLEmailRemover(replacement="[URL]")
        text = "Visit http://example.com for more info"
        result = remover.process(text)
        assert "[URL]" in result
        assert "http://example.com" not in result

    def test_remove_emails_with_replacement(self):
        """Test removing email with replacement string"""
        remover = URLEmailRemover(replacement="[EMAIL]")
        text = "Contact test@example.com"
        result = remover.process(text)
        assert "[EMAIL]" in result
        assert "test@example.com" not in result

    def test_disable_url_removal(self):
        """Test disabling URL removal"""
        remover = URLEmailRemover(remove_urls=False)
        text = "Visit http://example.com for more info"
        result = remover.process(text)
        assert "http://example.com" in result

    def test_process_empty_string(self):
        """Test processing empty string"""
        remover = URLEmailRemover()
        result = remover.process("")
        assert result == ""

    def test_process_none(self):
        """Test processing None"""
        remover = URLEmailRemover()
        result = remover.process(None)
        assert result is None


class TestSpecialCharacterNormalizer:
    """Special character normalizer tests"""

    def test_init_defaults(self):
        """Test initialization with default values"""
        normalizer = SpecialCharacterNormalizer()
        assert normalizer.chars_to_remove == ""
        assert normalizer.chars_to_replace == {}

    def test_init_with_chars_to_remove(self):
        """Test initialization with characters to remove"""
        normalizer = SpecialCharacterNormalizer(chars_to_remove="!@#")
        assert normalizer.chars_to_remove == "!@#"

    def test_init_with_chars_to_replace(self):
        """Test initialization with characters to replace"""
        normalizer = SpecialCharacterNormalizer(
            chars_to_replace={"&": "and", "@": "at"}
        )
        assert normalizer.chars_to_replace == {"&": "and", "@": "at"}

    def test_remove_control_characters(self):
        """Test removing control characters"""
        normalizer = SpecialCharacterNormalizer()
        text = "Test\x00text\x1Fwith\x7Fcontrol"
        result = normalizer.process(text)
        assert "\x00" not in result
        assert "\x1F" not in result
        assert "\x7F" not in result

    def test_replace_characters(self):
        """Test replacing characters"""
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
        """Test removing specified characters"""
        normalizer = SpecialCharacterNormalizer(chars_to_remove="!@#")
        text = "Test!text@with#special"
        result = normalizer.process(text)
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result

    def test_process_empty_string(self):
        """Test processing empty string"""
        normalizer = SpecialCharacterNormalizer()
        result = normalizer.process("")
        assert result == ""

    def test_process_none(self):
        """Test processing None"""
        normalizer = SpecialCharacterNormalizer()
        result = normalizer.process(None)
        assert result is None


class TestPreprocessingPipeline:
    """Preprocessing pipeline tests"""

    def test_init_empty(self):
        """Test initializing empty pipeline"""
        pipeline = PreprocessingPipeline()
        assert len(pipeline.preprocessors) == 0

    def test_init_with_preprocessors(self):
        """Test initialization with preprocessors"""
        preprocessor1 = WhitespaceNormalizer()
        preprocessor2 = URLEmailRemover()
        pipeline = PreprocessingPipeline([preprocessor1, preprocessor2])
        assert len(pipeline.preprocessors) == 2

    def test_add_preprocessor(self):
        """Test adding preprocessor"""
        pipeline = PreprocessingPipeline()
        preprocessor = WhitespaceNormalizer()
        pipeline.add_preprocessor(preprocessor)
        assert len(pipeline.preprocessors) == 1
        assert pipeline.preprocessors[0] == preprocessor

    def test_process_single_preprocessor(self):
        """Test processing single preprocessor"""
        pipeline = PreprocessingPipeline([WhitespaceNormalizer()])
        text = "This   is   a   test"
        result = pipeline.process(text)
        assert result == "This is a test"

    def test_process_order(self):
        """Test processing order"""
        # Create a preprocessor that tracks processing order
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
        # Verify order (by call order)
        assert pipeline.preprocessors[0].name == "first"
        assert pipeline.preprocessors[1].name == "second"

    def test_call(self):
        """Test callable interface"""
        pipeline = PreprocessingPipeline([WhitespaceNormalizer()])
        text = "This   is   a   test"
        result = pipeline(text)
        assert result == "This is a test"

    def test_len(self):
        """Test length method"""
        pipeline = PreprocessingPipeline([
            WhitespaceNormalizer(),
            URLEmailRemover(),
        ])
        assert len(pipeline) == 2

    def test_process_empty_string(self):
        """Test processing empty string"""
        pipeline = PreprocessingPipeline([WhitespaceNormalizer()])
        result = pipeline.process("")
        assert result == ""

