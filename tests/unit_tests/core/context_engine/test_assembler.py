#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import pytest
from unittest.mock import patch

from jiuwen.core.context_engine.base import ContextWindow
from jiuwen.core.context_engine.processor.assemble.assembler import (
    AssemblerConfig,
    AssemblerProcessor,
)


class TestAssemblerConfig:
    """Test cases for AssemblerConfig"""

    def test_config_default_values(self):
        """Test that config has correct default values"""
        config = AssemblerConfig()
        assert config.processor_type == "assembler"
        assert config.template_content == ""
        assert config.return_format == "message"
        assert config.variable_mappings == {}
        assert config.default_values == {}

    def test_config_custom_values(self):
        """Test config with custom values"""
        template = "Hello {{name}}, your message: {{message}}"
        config = AssemblerConfig(
            template_content=template,
            return_format="text",
            variable_mappings={"user_input": "message"},
            default_values={"name": "User"},
        )

        assert config.template_content == template
        assert config.return_format == "text"
        assert config.variable_mappings == {"user_input": "message"}
        assert config.default_values == {"name": "User"}


class TestAssemblerProcessor:
    """Test cases for AssemblerProcessor"""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing"""
        return AssemblerConfig(
            template_content="User: {{user_input}}\nSystem: {{system_prompt}}",
            return_format="text",
        )

    @pytest.fixture
    def engine_input(self):
        """Sample EngineInput for testing"""
        return ContextWindow(
            user_input="Hello, how are you?",
            system_prompt="You are a helpful assistant",
            chat_history="Previous conversation",
        )

    def test_initialization(self, basic_config):
        """Test that processor initializes correctly"""
        processor = AssemblerProcessor(basic_config)
        assert processor.config == basic_config
        assert hasattr(processor, "assembler")

    def test_extract_template_variables_basic(self, basic_config, engine_input):
        """Test basic variable extraction from EngineInput"""
        processor = AssemblerProcessor(basic_config)
        variables = processor._extract_template_variables(engine_input)

        assert variables["user_input"] == "Hello, how are you?"
        assert variables["system_prompt"] == "You are a helpful assistant"
        assert variables["chat_history"] == "Previous conversation"

    def test_extract_template_variables_with_mappings(self):
        """Test variable extraction with custom mappings"""
        config = AssemblerConfig(
            template_content="Input: {{msg}}\nSystem: {{sys}}",
            return_format="text",
            variable_mappings={"user_input": "msg", "system_prompt": "sys"},
        )

        processor = AssemblerProcessor(config)
        engine_input = ContextWindow(
            user_input="Test message", system_prompt="Test system"
        )

        variables = processor._extract_template_variables(engine_input)
        assert variables["msg"] == "Test message"
        assert variables["sys"] == "Test system"
        assert "user_input" not in variables  # Should use mapped names

    def test_extract_template_variables_with_defaults(self):
        """Test variable extraction with default values"""
        config = AssemblerConfig(
            template_content="User: {{user_input}}\nRole: {{role}}",
            return_format="text",
            default_values={"role": "assistant"},
        )

        processor = AssemblerProcessor(config)
        engine_input = ContextWindow(user_input="Hello")

        variables = processor._extract_template_variables(engine_input)
        assert variables["user_input"] == "Hello"
        assert variables["role"] == "assistant"  # From defaults

    def test_run_basic_assembly(self, basic_config, engine_input):
        """Test basic assembly functionality"""
        processor = AssemblerProcessor(basic_config)
        result = processor.run(engine_input)

        assert result.full_prompt is not None
        # Should contain both user input and system prompt
        assert "Hello, how are you?" in result.full_prompt
        assert "You are a helpful assistant" in result.full_prompt

    def test_run_message_format(self):
        """Test assembly with message return format"""
        config = AssemblerConfig(
            template_content="`#user#` {{user_input}}", return_format="message"
        )

        processor = AssemblerProcessor(config)
        engine_input = ContextWindow(user_input="Test message")

        result = processor.run(engine_input)
        assert result.full_prompt is not None
        assert isinstance(result.full_prompt, str)

    def test_run_with_missing_required_variables(self):
        """Test behavior when required template variables are missing"""
        config = AssemblerConfig(
            template_content="Required: {{required_var}}\nOptional: {{optional_var}}",
            return_format="text",
        )

        processor = AssemblerProcessor(config)
        engine_input = ContextWindow(user_input="test")  # Missing required_var

        # This should not raise an exception due to error handling
        result = processor.run(engine_input)
        assert result is not None

    def test_factory_integration(self):
        """Test that processor can be created through factory"""
        from jiuwen.core.context_engine.processor.factory import ProcessorFactory

        config_dict = {
            "processor_type": "assembler",
            "template_content": "Test template",
            "return_format": "text",
        }

        factory = ProcessorFactory()
        processor = factory.create_processor(config_dict)

        assert processor is not None
        assert isinstance(processor, AssemblerProcessor)
        assert processor.config.template_content == "Test template"


class TestAssemblerProcessorEdgeCases:
    """Test edge cases for AssemblerProcessor"""

    def test_empty_template(self):
        """Test with empty template"""
        config = AssemblerConfig(template_content="", return_format="text")
        processor = AssemblerProcessor(config)

        engine_input = ContextWindow(user_input="test")
        result = processor.run(engine_input)

        assert result.full_prompt == ""  # Empty template should produce empty output

    def test_none_values_in_input(self):
        """Test handling of None values in EngineInput"""
        config = AssemblerConfig(
            template_content="User: {{user_input}}\nSystem: {{system_prompt}}",
            return_format="text",
        )

        processor = AssemblerProcessor(config)
        engine_input = ContextWindow(user_input="", system_prompt="test system")

        result = processor.run(engine_input)
        # Should handle empty values gracefully
        assert "test system" in result.full_prompt

    def test_complex_variable_mappings(self):
        """Test complex variable mapping scenarios"""
        config = AssemblerConfig(
            template_content="Msg: {{msg}}\nHist: {{hist}}\nSys: {{sys}}",
            return_format="text",
            variable_mappings={"user_input": "msg", "chat_history": "hist"},
            default_values={"sys": "default system"},
        )

        processor = AssemblerProcessor(config)
        engine_input = ContextWindow(
            user_input="user message",
            chat_history="conversation history",
            # system_prompt is empty, should use default
        )

        variables = processor._extract_template_variables(engine_input)
        assert variables["msg"] == "user message"
        assert variables["hist"] == "conversation history"
        assert variables["sys"] == "default system"  # From defaults
