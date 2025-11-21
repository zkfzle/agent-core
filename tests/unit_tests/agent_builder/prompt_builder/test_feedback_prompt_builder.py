#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import unittest
from unittest.mock import patch
from typing import List, Any, Dict, Iterator, AsyncIterator

from openjiuwen.agent_builder.prompt_builder.builder.feedback_prompt_builder import FeedbackPromptBuilder
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.prompt.template.template import Template
from openjiuwen.core.utils.prompt.template.template_manager import TemplateManager
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.llm.messages import AIMessage
from openjiuwen.agent_builder.prompt_builder.builder.meta_template_builder import (MetaTemplateBuilder,
                                                                                   META_TEMPLATE_NAME_PREFIX)
import openjiuwen.agent_builder.prompt_builder.builder.utils as TEMPLATE

Mock_intent = '''```json{"intent": "true",\n"optimized_feedback": "[优化后的反馈信息]",
                            "optimization_directions": "[联想并提示其他优化方向的建议]"}```'''


class MockLLMModel(BaseModelClient):
    """Mock 大模型，返回预定义的响应"""

    def __init__(self, api_key: str, api_base: str, **kwargs):
        super().__init__(api_key=api_key, api_base=api_base)

    def _get_next_response(self, messages) -> AIMessage:
        """获取下一个响应"""
        return AIMessage(content="".join(Mock_intent+msg.get("content") for msg in messages))

    def _invoke(
            self,
            model_name: str,
            messages: List[Dict],
            tools: List[Dict] = None,
            temperature: float = 0.1,
            top_p: float = 0.1,
            **kwargs: Any
    ) -> AIMessage:
        """同步调用"""
        return self._get_next_response(messages)

    async def _ainvoke(
            self,
            model_name: str,
            messages: List[Dict],
            tools: List[Dict] = None,
            temperature: float = 0.1,
            top_p: float = 0.1,
            **kwargs: Any
    ) -> AIMessage:
        """异步调用"""
        return self._get_next_response()

    def _stream(
            self,
            model_name: str,
            messages: List[Dict],
            tools: List[Dict] = None,
            temperature: float = 0.1,
            top_p: float = 0.1,
            **kwargs: Any
    ) -> Iterator[Any]:
        """流式返回"""
        result = self._get_next_response()
        yield result

    async def _astream(
            self,
            model_name: str,
            messages: List[Dict],
            tools: List[Dict] = None,
            temperature: float = 0.1,
            top_p: float = 0.1,
            **kwargs: Any
    ) -> AsyncIterator[Any]:
        """异步流式返回"""
        result = self._get_next_response()
        yield result


class TestFeedbackPromptBuilder(unittest.TestCase):
    def setUp(self):
        pass

    def test_feedback_prompt_builder_general(self):
        mock_llm = MockLLMModel(api_key="mock_key", api_base="https://api.openai.com")
        with patch('openjiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model') as mock_get_model:
            mock_get_model.return_value = mock_llm
            config = ModelConfig(model_provider="", model_info=BaseModelInfo())
            builder = FeedbackPromptBuilder(config)
            prompt = "你是一个旅行助手"
            feedback = "丰富一下"
            response = builder.build(prompt=prompt, feedback=feedback, mode = "general")
            self.assertEqual(response,
                             Mock_intent+TEMPLATE.PROMPT_FEEDBACK_GENERAL_TEMPLATE.format(
                             dict(original_prompt=prompt, suggestion=feedback)).content[0].content)

    def test_feedback_prompt_builder_insert(self):
        mock_llm = MockLLMModel(api_key="mock_key", api_base="https://api.openai.com")
        with patch('openjiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model') as mock_get_model:
            mock_get_model.return_value = mock_llm
            config = ModelConfig(model_provider="", model_info=BaseModelInfo())
            builder = FeedbackPromptBuilder(config)
            INSERT_TAG = "[用户要插入的位置]"
            prompt = "你是一个旅行助手"
            feedback = "丰富一下"
            response = builder.build(prompt=prompt, feedback=feedback, mode="insert", start_pos=3)
            self.assertEqual(response,
                             Mock_intent+TEMPLATE.PROMPT_FEEDBACK_INSERT_TEMPLATE.format(
                             dict(original_prompt=prompt[:3]+INSERT_TAG+prompt[3:],
                                  suggestion="[优化后的反馈信息]")).content[0].content)

    def test_feedback_prompt_builder_select(self):
        mock_llm = MockLLMModel(api_key="mock_key", api_base="https://api.openai.com")
        with patch('openjiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model') as mock_get_model:
            mock_get_model.return_value = mock_llm
            config = ModelConfig(model_provider="", model_info=BaseModelInfo())
            builder = FeedbackPromptBuilder(config)

            prompt = "你是一个旅行助手"
            feedback = "丰富一下"
            response = builder.build(prompt=prompt+Mock_intent,
                                     feedback=feedback,
                                     mode="select",
                                     start_pos=0,
                                     end_pos=3)
            self.assertEqual(response, Mock_intent+TEMPLATE.PROMPT_FEEDBACK_SELECT_TEMPLATE.format(
                                 dict(original_prompt=prompt+Mock_intent,
                                      suggestion="[优化后的反馈信息]",
                                      pending_optimized_prompt=prompt[0:3])).content[0].content)


if __name__ == "__main__":
    unittest.main()
