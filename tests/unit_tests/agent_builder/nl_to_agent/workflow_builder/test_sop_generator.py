#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
import unittest
from unittest.mock import MagicMock, Mock

from jiuwen.core.utils.llm.messages import SystemMessage, HumanMessage
import jiuwen.agent_builder.nl_to_agent.agent_builder.workflow_builder.sop_generator.sop_generator as sg
from jiuwen.agent_builder.nl_to_agent.agent_builder.workflow_builder.sop_generator.sop_generator import SopGenerator


LLM_CHAT_SOP_RESULT = "sop_result"


class TestSOPGenrator(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock(name="llm")
        self.llm.chat.return_value = LLM_CHAT_SOP_RESULT
        self.sop_generator = SopGenerator(self.llm)

    def test_format_resource_info(self):
        resource_info = self.sop_generator._format_resource_info(None)
        self.assertEqual(resource_info, "无可用工具/资源/外部接口。")

        # Invalid resource info format
        resource_info = self.sop_generator._format_resource_info({
            "plugin": [
                {"name": "", "description": "test_plugin_desc_1"},
                {"name": None, "description": "test_plugin_desc_2"},
                {"description": "test_plugin_desc_3"}
            ],
        })
        self.assertEqual(resource_info, "plugin:")

        # Valid resource info format
        resource_info = self.sop_generator._format_resource_info({
            "plugin": [
                {"name": "test_plugin_1", "description": "test_plugin_desc_1"},
                {"name": "test_plugin_2", "description": "test_plugin_desc_2"}
            ],
        })
        self.assertEqual(
            resource_info,
            "plugin:\n- test_plugin_1: test_plugin_desc_1\n- test_plugin_2: test_plugin_desc_2"
        )

    def test_execute(self):
        self.sop_generator._execute("test_query", "test_system_prompt")
        self.llm.chat.assert_called_with(
            [SystemMessage(content="test_system_prompt"), HumanMessage(content="test_query")]
        )

    def test_transform(self):
        self.sop_generator._execute = Mock()
        query = "test_query"
        self.sop_generator.transform(query)
        self.sop_generator._execute.assert_called_with(query, sg.transform_system_prompt)

    def test_generate_without_resource(self):
        self.sop_generator._execute = Mock()
        query = "test_query"
        system_prompt = sg.generate_system_prompt.replace("{{resource_info}}", "无可用工具/资源/外部接口。")
        self.sop_generator.generate(query, None)
        self.sop_generator._execute.assert_called_with(sg.SOP_GENERATE_PROMPT + query, system_prompt)

    def test_generate_with_resource(self):
        self.sop_generator._execute = Mock()
        query = "test_query"
        resource = {
            "plugin": [
                {"name": "test_plugin_1", "description": "test_plugin_desc_1"},
                {"name": "test_plugin_2", "description": "test_plugin_desc_2"}
            ]
        }
        system_prompt = sg.generate_system_prompt.replace(
            "{{resource_info}}", "plugin:\n- test_plugin_1: test_plugin_desc_1\n- test_plugin_2: test_plugin_desc_2"
        )
        self.sop_generator.generate(query, resource)
        self.sop_generator._execute.assert_called_with(sg.SOP_GENERATE_PROMPT + query, system_prompt)


if __name__ == "__main__":
    unittest.main()