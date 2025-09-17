#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import unittest
from unittest import mock
from unittest.mock import MagicMock

from jiuwen.core.utils.llm.messages import ToolInfo, Function, Parameters
from jiuwen.core.utils.tool.service_api.param import Param
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi


class TestRestFulApi(unittest.TestCase):
    def setUp(self):
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.text = "{}"
        response_mock.content = b"{}"
        self.mocked_functions = mock.patch.multiple(
            "requests",
            request=mock.MagicMock(return_value=response_mock)
        )
        self.mocked_functions.start()

    def tearDown(self):
        self.mocked_functions.stop()

    def test_invoke(self):
        mock_data = RestfulApi(
            name="test",
            description="test",
            params=[],
            path="http://127.0.0.1:8000",
            headers={},
            method="GET",
            response=[],
        )
        mock_data.invoke({})
        self.assertEqual(mock_data.headers, {})

    def test_get_tool_info(self):
        mock_data = RestfulApi(
            name="test",
            description="test",
            params=[Param("test", "test", param_type="string", default_value="123")],
            path="http://127.0.0.1:8000",
            headers={},
            method="GET",
            response=[Param("results", "test", param_type="string", default_value="456")],
        )
        res = mock_data.get_tool_info()
        too_info = ToolInfo(
            function=Function(
                name='test', description='test',
                parameters=Parameters(
                    type='object',
                    properties={'test': {'description': 'test', 'type': 'string'}, 'required': ['test']},
                    required=['test']
                )
            )
        )
        self.assertEqual(res, too_info)