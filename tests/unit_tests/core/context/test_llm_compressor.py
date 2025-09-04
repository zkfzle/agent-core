import sys
import types
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from jiuwen.core.context_engine.base import EngineInput
from jiuwen.core.context_engine.processor.preprocess.llm_compressor import (
    LLMCompressorConfig,
    LLMCompressor,
)
from jiuwen.core.utils.llm.base import BaseChatModel, BaseModelInfo
from jiuwen.core.utils.llm.messages import AIMessage

# Mock modules to avoid import issues
fake_base = types.ModuleType("base")
fake_base.logger = Mock()

fake_exception_module = types.ModuleType("base")
fake_exception_module.JiuWenBaseException = Mock()

sys.modules["jiuwen.core.common.logging.base"] = fake_base
sys.modules["jiuwen.core.common.exception.base"] = fake_exception_module


@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns predictable responses"""
    mock_client = Mock(spec=BaseChatModel)
    mock_client.invoke = Mock(
        return_value=AIMessage(content="Compressed content summary")
    )
    return mock_client


@pytest.fixture
def basic_config():
    """Basic LLMCompressorConfig for testing"""
    return LLMCompressorConfig(
        model_type="siliconflow",
        model_config={"model_name": "test-model"},
        max_length=100,
    )


@pytest.fixture
def compressor_with_mock_llm(basic_config, mock_llm_client):
    """LLMCompressor with mocked LLM client"""
    with patch("jiuwen.core.utils.llm.model_utils.model_factory") as mock_factory:
        mock_factory_instance = Mock()
        mock_factory_instance.get_model.return_value = mock_llm_client
        mock_factory.return_value = mock_factory_instance

        compressor = LLMCompressor(basic_config)
        # Replace the LLM client with our mock
        compressor.llm_client = mock_llm_client
        return compressor


class TestLLMCompressorConfig:
    """Test LLMCompressorConfig validation and defaults"""

    def test_default_values(self):
        """Test that default values are set correctly"""
        config = LLMCompressorConfig()
        assert config.processor_type == "llm_compressor"
        assert config.model_type == "default"
        assert config.max_length == 500
        assert (
            config.compression_prompt
            == "Please compress the following content, retaining key information:"
        )
        assert config.content_format == "{label}: {content}"
        assert config.labels == {
            "user_input": "User Input",
            "chat_history": "Chat History",
            "system_input": "System Input",
        }

    def test_custom_values(self):
        """Test custom configuration values"""
        config = LLMCompressorConfig(
            model_type="custom-model",
            model_config={"temperature": 0.7},
            max_length=200,
            compression_prompt="Custom compression prompt:",
            content_format="[{label}] {content}",
            labels={"user_input": "User", "chat_history": "History"},
        )

        assert config.model_type == "custom-model"
        assert config.max_length == 200
        assert config.compression_prompt == "Custom compression prompt:"
        assert config.content_format == "[{label}] {content}"
        assert config.labels == {"user_input": "User", "chat_history": "History"}

    def test_invalid_max_length(self):
        """Test validation for invalid max_length"""
        with pytest.raises(ValidationError):
            LLMCompressorConfig(max_length="test")


class TestLLMCompressor:
    """Test LLMCompressor functionality"""

    def test_should_compress_short_content(self, compressor_with_mock_llm):
        """Test that short content doesn't need compression"""
        short_content = "Short content"
        assert not compressor_with_mock_llm._should_compress(short_content)

    def test_should_compress_long_content(self, compressor_with_mock_llm):
        """Test that long content needs compression"""
        long_content = "A" * 101  # 101 chars, over max_length=100
        assert compressor_with_mock_llm._should_compress(long_content)

    def test_extract_content_user_input_only(self, compressor_with_mock_llm):
        """Test content extraction with only user input"""
        engine_input = EngineInput(user_input="Hello world")
        result = compressor_with_mock_llm._extract_content(engine_input)
        assert result == "User Input: Hello world"

    def test_extract_content_all_fields(self, compressor_with_mock_llm):
        """Test content extraction with all input fields"""
        engine_input = EngineInput(
            user_input="User question",
            chat_history="Previous conversation",
            system_prompt="System instructions",
        )
        result = compressor_with_mock_llm._extract_content(engine_input)
        expected_lines = [
            "User Input: User question",
            "Chat History: Previous conversation",
            "System Input: System instructions",
        ]
        assert result == "\n".join(expected_lines)

    def test_extract_content_empty_fields(self, compressor_with_mock_llm):
        """Test content extraction with empty fields"""
        engine_input = EngineInput()
        result = compressor_with_mock_llm._extract_content(engine_input)
        assert result == ""

    def test_run_short_content(self, compressor_with_mock_llm):
        """Test run with content that doesn't need compression"""
        engine_input = EngineInput(user_input="Short")
        result = compressor_with_mock_llm.run(engine_input)
        assert result.full_output == "User Input: Short"
        compressor_with_mock_llm.llm_client.invoke.assert_not_called()

    def test_run_long_content(self, compressor_with_mock_llm):
        """Test run with content that needs compression"""
        long_content = "A" * 550
        engine_input = EngineInput(user_input=long_content)
        result = compressor_with_mock_llm.run(engine_input)
        assert result.full_output == "Compressed content summary"
        compressor_with_mock_llm.llm_client.invoke.assert_called_once()

    def test_compress_with_llm_success(self, compressor_with_mock_llm):
        """Test successful LLM compression"""
        content = "Long content that needs compression"
        result = compressor_with_mock_llm._compress_with_llm(content)

        assert result == "Compressed content summary"
        compressor_with_mock_llm.llm_client.invoke.assert_called_once()

    def test_compress_with_llm_failure(self, compressor_with_mock_llm):
        """Test LLM compression failure falls back to truncation"""
        compressor_with_mock_llm.llm_client.invoke.side_effect = Exception("LLM error")

        long_content = "A" * 200
        result = compressor_with_mock_llm._compress_with_llm(long_content)

        # Should truncate to max_length (100) + "..."
        assert result == "A" * 100 + "..."
        compressor_with_mock_llm.llm_client.invoke.assert_called_once()

    def test_custom_content_format(self, basic_config):
        """Test custom content format in extraction"""
        basic_config.content_format = "[{label}] - {content}"
        basic_config.labels = {"user_input": "USER"}

        compressor = LLMCompressor(basic_config)
        engine_input = EngineInput(user_input="test input")
        result = compressor._extract_content(engine_input)

        assert result == "[USER] - test input"

    def test_edge_case_max_length_boundary(self, compressor_with_mock_llm):
        """Test boundary condition for max_length"""
        # Exactly at max_length should not compress
        content_at_limit = "A" * 100
        assert not compressor_with_mock_llm._should_compress(content_at_limit)

        # One character over should compress
        content_over_limit = "A" * 101
        assert compressor_with_mock_llm._should_compress(content_over_limit)


class TestLLMCompressorIntegration:
    """Integration tests with actual ModelFactory"""

    def test_config_attribute_access(self, basic_config):
        """Test that config attributes are accessible"""
        compressor = LLMCompressor(basic_config)

        assert (
            compressor.compression_prompt
            == "Please compress the following content, retaining key information:"
        )
        assert compressor.max_length == 100
        assert compressor.content_format == "{label}: {content}"
        assert compressor.labels == {
            "user_input": "User Input",
            "chat_history": "Chat History",
            "system_input": "System Input",
        }
