#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

from openjiuwen.agent_builder.nl_to_agent.llm_agent_builder.clarifier.clarifier import Clarifier


class TestClarifier(unittest.TestCase):

    def setUp(self):
        self.mock_model = Mock()
        self.mock_model.chat.return_value = "Mock LLM response"

        self.clarifier = Clarifier(self.mock_model)

    def test_clarify_basic_functionality(self):
        test_messages = [
            {"role": "user", "content": "Create a data analysis agent"}
        ]
        test_resource = {
            "plugins": [
                {"name": "Data query plugin", "description": "Query data from database", "version": "1.0"},
                {"name": "Chart generation plugin", "description": "Generate various charts", "version": "2.1"}
            ]
        }

        result = self.clarifier.clarify(test_messages, test_resource)

        self.mock_model.chat.assert_called_once()

        call_args = self.mock_model.chat.call_args
        messages_passed = call_args[1]['messages']
        method_passed = call_args[1]['method']
        add_prefix_passed = call_args[1]['add_prefix']

        self.assertEqual(len(messages_passed), 2)
        self.assertEqual(messages_passed[0]["role"], "system")
        self.assertEqual(messages_passed[1]["role"], "user")
        self.assertEqual(messages_passed[1]["content"], "Create a data analysis agent")
        self.assertEqual(method_passed, "invoke")
        self.assertEqual(add_prefix_passed, False)

        self.assertIn("Mock LLM response", result)
        self.assertIn("<可用外部资源>", result)
        self.assertIn("【可用插件】", result)
        self.assertIn("plugins", result)
        self.assertIn("</可用外部资源>", result)

    def test_clarify_with_empty_resource(self):
        test_messages = [{"role": "user", "content": "Test message"}]
        empty_resource = {}

        result = self.clarifier.clarify(test_messages, empty_resource)

        self.assertIn("Mock LLM response", result)
        self.assertIn("<可用外部资源>", result)
        self.assertIn("</可用外部资源>", result)

    def test_clarify_with_empty_plugins_list(self):
        test_messages = [{"role": "user", "content": "Test message"}]
        test_resource = {
            "plugins": []
        }

        result = self.clarifier.clarify(test_messages, test_resource)

        self.assertIn("Mock LLM response", result)
        self.assertIn("<可用外部资源>", result)
        self.assertIn("</可用外部资源>", result)

    def test_clarify_with_single_plugin(self):
        test_messages = [{"role": "user", "content": "Test message"}]
        test_resource = {
            "plugins": [
                {"name": "Single plugin", "description": "Only one plugin"}
            ]
        }

        result = self.clarifier.clarify(test_messages, test_resource)

        self.assertIn("Mock LLM response", result)
        self.assertIn("plugins", result)

    def test_clarify_with_complex_plugin_structure(self):
        test_messages = [{"role": "user", "content": "Test message"}]
        test_resource = {
            "plugins": [
                {
                    "name": "Complex plugin 1",
                    "description": "A complex plugin",
                    "version": "1.0",
                    "author": "Team A",
                    "dependencies": ["lib1", "lib2"],
                    "capabilities": ["Feature A", "Feature B", "Feature C"]
                },
                {
                    "name": "Complex plugin 2",
                    "description": "Another complex plugin",
                    "version": "2.1",
                    "author": "Team B",
                    "dependencies": ["lib3"],
                    "capabilities": ["Feature D"]
                }
            ]
        }

        result = self.clarifier.clarify(test_messages, test_resource)

        self.assertIn("Mock LLM response", result)
        self.assertIn("plugins", result)

    def test_clarify_with_missing_plugin_fields(self):
        test_messages = [{"role": "user", "content": "Test message"}]
        test_resource = {
            "plugins": [
                {"name": "Plugin with name only"},
                {"description": "Plugin with description only"},
                {}
            ]
        }

        result = self.clarifier.clarify(test_messages, test_resource)

        self.assertIn("Mock LLM response", result)
        self.assertIn("plugins", result)

    def test_clarify_with_mixed_plugin_formats(self):
        test_messages = [{"role": "user", "content": "Test message"}]
        test_resource = {
            "plugins": [
                {"name": "Standard plugin", "description": "Standard format plugin"},
                "Simple plugin string",
                123,
                ["List format", "plugin"]
            ]
        }

        result = self.clarifier.clarify(test_messages, test_resource)

        self.assertIn("Mock LLM response", result)
        self.assertIn("plugins", result)

    def test_clarify_with_nested_plugin_structure(self):
        test_messages = [{"role": "user", "content": "Test message"}]
        test_resource = {
            "plugins": [
                {
                    "name": "Nested plugin",
                    "description": "Plugin with nested structure",
                    "metadata": {
                        "created": "2024-01-01",
                        "updated": "2024-06-01"
                    },
                    "config": {
                        "settings": {
                            "param1": "value1",
                            "param2": "value2"
                        }
                    }
                }
            ]
        }

        result = self.clarifier.clarify(test_messages, test_resource)

        self.assertIn("Mock LLM response", result)
        self.assertIn("plugins", result)

    def test_clarify_output_format(self):
        test_messages = [{"role": "user", "content": "Test message"}]
        test_resource = {
            "plugins": [
                {"name": "Test plugin 1", "description": "First test plugin"},
                {"name": "Test plugin 2", "description": "Second test plugin"}
            ]
        }

        result = self.clarifier.clarify(test_messages, test_resource)

        self.assertIn("Mock LLM response", result)
        self.assertIn("<可用外部资源>", result)
        self.assertIn("【可用插件】", result)
        self.assertIn("plugins", result)
        self.assertIn("</可用外部资源>", result)

        llm_output_index = result.find("Mock LLM response")
        resource_start_index = result.find("<可用外部资源>")
        resource_end_index = result.find("</可用外部资源>")

        self.assertLess(llm_output_index, resource_start_index)
        self.assertGreater(resource_end_index, resource_start_index)

    def test_clarify_method_parameters(self):
        test_messages = [{"role": "user", "content": "Test message"}]
        test_resource = {
            "plugins": [
                {"name": "Test plugin", "description": "Test description"}
            ]
        }

        result = self.clarifier.clarify(test_messages, test_resource)

        call_args = self.mock_model.chat.call_args
        self.assertEqual(call_args[1]['method'], "invoke")
        self.assertEqual(call_args[1]['add_prefix'], False)

        messages_passed = call_args[1]['messages']
        self.assertIsInstance(messages_passed, list)
        self.assertTrue(len(messages_passed) >= 1)


class TestClarifierWithMockedPrompt(unittest.TestCase):

    @patch('openjiuwen.agent_builder.nl_to_agent.llm_agent_builder.clarifier.clarifier.PROMPT', "Mocked PROMPT content")
    def test_clarify_with_mocked_prompt(self):
        mock_model = Mock()
        mock_model.chat.return_value = "LLM response"

        clarifier = Clarifier(mock_model)
        result = clarifier.clarify(
            [{"role": "user", "content": "Test"}],
            {
                "plugins": [
                    {"name": "Test plugin", "description": "Test description"}
                ]
            }
        )

        self.assertIn("LLM response", result)
        self.assertIn("<可用外部资源>", result)
        self.assertIn("plugins", result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
