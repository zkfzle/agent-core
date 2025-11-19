#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import unittest
from unittest.mock import patch, MagicMock
import json

from openjiuwen.agent_builder.nl_to_agent.common.resource.resource_retrieve import ResourceRetriever

LLM_CHAT_RESULT = """{
    "plugin_id_list": ["mock_tool_id_1", "mock_tool_id_2"]
}"""


class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock(name='llm')
        self.llm.chat.return_value = LLM_CHAT_RESULT

        self.retriever = ResourceRetriever(self.llm)

    @patch.object(ResourceRetriever, '_load_resources', return_value={"raw_plugins": "plugin_data"})
    @patch.object(ResourceRetriever, '_preprocess_plugin_info', return_value=("plugin_dict", "tool_plugin_id_map"))
    @patch.object(ResourceRetriever, '_format_plugin_info_list', return_value="plugin_info_list")
    @patch.object(ResourceRetriever, '_llm_retrieve', return_value=json.loads(LLM_CHAT_RESULT))
    @patch.object(ResourceRetriever, '_get_retrieved_info', return_value="retrieved_info")
    def test_retrieve(self,
                      mock_get_retrieved_info,
                      mock_llm_retrieve,
                      mock_format_plugin_info_list,
                      mock_preprocess_plugin_info,
                      mock_load_resources):
        query = "test query"
        dialog_history = "test dialog history"
        result = self.retriever.retrieve(query, dialog_history)
        mock_load_resources.assert_called_once()
        mock_preprocess_plugin_info.assert_called_once_with("plugin_data")
        mock_format_plugin_info_list.assert_called_once_with("plugin_dict")
        mock_llm_retrieve.assert_called_once()
        mock_get_retrieved_info.assert_called_once_with(
            json.loads(LLM_CHAT_RESULT), "plugin_dict", "tool_plugin_id_map"
        )
        self.assertEqual(result, "retrieved_info")


    def test_preprocess_plugin_info(self):
        raw_plugins = [
            {
                "plugin_id": "plugin_1",
                "plugin_name": "Plugin One",
                "plugin_desc": "Description of Plugin One",
                "tools": [
                    {
                        "tool_id": "tool_1",
                        "tool_name": "Tool One",
                        "desc": "Description of Tool One",
                        "input_parameters": [{"name": "input1", "description": "Input 1 desc", "others": "mock_value"}],
                        "output_parameters": [{"name": "output1", "description": "Output 1 desc", "others": "mock_value"}]
                    }
                ]
            }
        ]
        plugin_dict, tool_plugin_id_map = self.retriever._preprocess_plugin_info(raw_plugins)
        self.assertIn("plugin_1", plugin_dict)
        self.assertIn("tool_1", plugin_dict["plugin_1"]["tools"])
        self.assertEqual(
            raw_plugins[0]["tools"][0]["input_parameters"][0],
            plugin_dict["plugin_1"]["tools"]["tool_1"]["ori_inputs"][0]
        )
        self.assertEqual(
            {"name": "input1", "desc": "Input 1 desc"},
            plugin_dict["plugin_1"]["tools"]["tool_1"]["inputs_for_dl_gen"][0]
        )
        self.assertEqual(tool_plugin_id_map["tool_1"], "plugin_1")

    def test_format_plugin_info_list(self):
        plugin_dict = {
            "plugin_1": {
                "plugin_id": "plugin_1",
                "plugin_name": "Plugin One",
                "plugin_desc": "Description of Plugin One",
                "tools": {
                    "tool_1": {
                        "tool_id": "tool_1",
                        "tool_name": "Tool One",
                        "tool_desc": "Description of Tool One",
                        "ori_inputs": [],
                        "ori_outputs": [],
                        "inputs_for_dl_gen": [],
                        "outputs_for_dl_gen": []
                    }
                }
            }
        }
        plugin_info_list = self.retriever._format_plugin_info_list(plugin_dict)
        expected = [
            {
                'plugin_name': 'Plugin One',
                'plugin_id': 'plugin_1',
                'plugin_desc': 'Description of Plugin One',
                'tools': [
                    {
                        'tool_name': 'Tool One',
                        'tool_id': 'tool_1',
                        'tool_desc': 'Description of Tool One'
                    }
                ]
            }
        ]
        self.assertEqual(plugin_info_list, expected)
        
    def test_get_retrieved_info(self):
        data = json.loads(LLM_CHAT_RESULT)
        plugin_dict = {
            "plugin_1": {
                "plugin_id": "plugin_1",
                "plugin_name": "Plugin One",
                "plugin_desc": "Description of Plugin One",
                "tools": {
                    "mock_tool_id_1": {
                        "tool_id": "mock_tool_id_1",
                        "tool_name": "Mock Tool 1",
                        "tool_desc": "Description of Mock Tool 1",
                        "ori_inputs": ["ori_input_1"],
                        "ori_outputs": ["ori_output_1"],
                        "inputs_for_dl_gen": ["input_for_dl_gen_1"],
                        "outputs_for_dl_gen": ["output_for_dl_gen_1"]
                    },
                    "mock_tool_id_2": {
                        "tool_id": "mock_tool_id_2",
                        "tool_name": "Mock Tool 2",
                        "tool_desc": "Description of Mock Tool 2",
                        "ori_inputs": ["ori_input_2"],
                        "ori_outputs": ["ori_output_2"],
                        "inputs_for_dl_gen": ["input_for_dl_gen_2"],
                        "outputs_for_dl_gen": ["output_for_dl_gen_2"]
                    }
                }
            },
            "plugin_2": {
                "plugin_id": "plugin_2",
                "plugin_name": "Plugin Two",
                "plugin_desc": "Description of Plugin Two",
                "tools": {
                    "mock_tool_id_3": {
                        "tool_id": "mock_tool_id_3",
                        "tool_name": "Mock Tool 3",
                        "tool_desc": "Description of Mock Tool 3",
                        "ori_inputs": ["ori_input_3"],
                        "ori_outputs": ["ori_output_3"],
                        "inputs_for_dl_gen": ["input_for_dl_gen_3"],
                        "outputs_for_dl_gen": ["output_for_dl_gen_3"]
                    }
                }
            }
        }
        tool_plugin_id_map = {
            "mock_tool_id_1": "plugin_1",
            "mock_tool_id_2": "plugin_1",
            "mock_tool_id_3": "plugin_2"
        }
        result = self.retriever._get_retrieved_info(data, plugin_dict, tool_plugin_id_map)
        expected_tools = [
            {
                "tool_id": "mock_tool_id_1",
                "tool_name": "Mock Tool 1",
                "tool_desc": "Description of Mock Tool 1",
                "inputs": ["input_for_dl_gen_1"],
                "outputs": ["output_for_dl_gen_1"]
            },
            {
                "tool_id": "mock_tool_id_2",
                "tool_name": "Mock Tool 2",
                "tool_desc": "Description of Mock Tool 2",
                "inputs": ["input_for_dl_gen_2"],
                "outputs": ["output_for_dl_gen_2"]
            }
        ]
        self.assertEqual(result["tool_detail_list"], expected_tools)
        self.assertEqual(result["plugin_dict"], {"plugin_1": plugin_dict["plugin_1"]})
        self.assertEqual(
            result["tool_plugin_id_map"],
            {"mock_tool_id_1": "plugin_1", "mock_tool_id_2": "plugin_1"}
        )


if __name__ == "__main__":
    unittest.main()
