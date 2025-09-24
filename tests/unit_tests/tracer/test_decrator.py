import asyncio
import unittest
from typing import Dict, List, Any, Union
from unittest.mock import MagicMock

from jiuwen.core.common.logging import logger
from jiuwen.core.context_engine.base import Context
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.stream.base import StreamMode, BaseStreamMode
from jiuwen.core.tracer.decorator import decrate_tool_with_trace, decrate_workflow_with_trace, decrate_model_with_trace
from jiuwen.core.utils.llm.messages import ToolInfo
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.tool.constant import Input, Output


class MockTool(Tool):
    async def ainvoke(self, inputs: Input, **kwargs) -> Output:
        logger.info(inputs)
        logger.info(f"begin to ainvoke , inputs={inputs}")
        return {}

    def invoke(self, inputs: Input, **kwargs) -> Output:
        logger.info(f"begin to invoke, inputs={inputs}")
        return inputs

    def get_tool_info(self) -> ToolInfo:
        pass


class MockWorkflow:
    async def invoke(self, inputs: Input, runtime: BaseRuntime, context: Context = None):
        logger.info(inputs)
        async for item in self.stream(inputs, runtime, context=context, stream_modes=[BaseStreamMode.CUSTOM]):
            continue
        return inputs

    async def stream(self, inputs: Input, runtime: BaseRuntime, context: Context = None,
                     stream_modes: list[StreamMode] = None):
        logger.info(inputs)
        logger.info(f"begin to ainvoke , inputs={inputs}")
        yield inputs


class MockModel:
    def invoke(self, model_name: str, messages: Union[List[str], List[Dict], str],
               tools: Union[List[ToolInfo], List[Dict]] = None, temperature: float = 0.1,
               top_p: float = 0.1, **kwargs: Any):
        logger.info(f"begin to invoke, inputs={messages}")
        return messages

    def stream(self, model_name: str, messages: Union[List[str], List[Dict], str],
               tools: Union[List[ToolInfo], List[Dict]] = None, temperature: float = 0.1,
               top_p: float = 0.1, **kwargs: Any):
        logger.info(messages)
        logger.info(f"begin to ainvoke , inputs={messages}")
        yield messages

    async def ainvoke(self, model_name: str, messages: Union[List[str], List[Dict], str],
                      tools: Union[List[ToolInfo], List[Dict]] = None, temperature: float = 0.1,
                      top_p: float = 0.1, **kwargs: Any):
        logger.info(messages)
        logger.info(f"begin to ainvoke , inputs={messages}")
        return messages


class TestDecator(unittest.TestCase):

    def test_decrate_tool(self):
        tool = MockTool()
        results = []

        async def mock_trigger(handler_class_name: str, event_name: str, **kwargs):
            results.append([handler_class_name, event_name, kwargs])

        mock_tracer = MagicMock()
        mock_tracer.trigger = mock_trigger
        mock_trigger.side_effect = mock_trigger

        mock_agent_span_manager = MagicMock()
        mock_agent_span_manager.create_agent_span = {}
        mock_trigger.tracer_agent_span_manager = mock_agent_span_manager

        decrate_tool_with_trace(tool, mock_tracer, None)
        tool.invoke({"a": "a"}, context=3)
        for item in results:
            print(item)
        assert len(results) == 2

        results.clear()
        asyncio.get_event_loop().run_until_complete(tool.ainvoke({"a": "a"}, context=3))
        for item in results:
            print(item)
        assert len(results) == 2

    def test_decrate_workflow(self):
        workflow = MockWorkflow()
        results = []

        async def mock_trigger(handler_class_name: str, event_name: str, **kwargs):
            results.append([handler_class_name, event_name, kwargs])

        mock_tracer = MagicMock()
        mock_tracer.trigger = mock_trigger
        mock_trigger.side_effect = mock_trigger

        mock_agent_span_manager = MagicMock()
        mock_agent_span_manager.create_agent_span = {}
        mock_trigger.tracer_agent_span_manager = mock_agent_span_manager
        decrate_workflow_with_trace(workflow, mock_tracer, None)

        asyncio.get_event_loop().run_until_complete(workflow.invoke({"a": "a"}, MagicMock(), context=None))

        for item in results:
            print(item)
        assert len(results) == 4

    def test_decrate_model(self):
        model = MockModel()
        results = []

        async def mock_trigger(handler_class_name: str, event_name: str, **kwargs):
            results.append([handler_class_name, event_name, kwargs])

        mock_tracer = MagicMock()
        mock_tracer.trigger = mock_trigger
        mock_trigger.side_effect = mock_trigger

        mock_agent_span_manager = MagicMock()
        mock_agent_span_manager.create_agent_span = {}
        mock_trigger.tracer_agent_span_manager = mock_agent_span_manager
        decrate_model_with_trace(model, mock_tracer, None)

        model.invoke("a", "messages")

        for item in results:
            print(item)
        assert len(results) == 2

        results.clear()
        for item in model.stream("a", "messages"):
            print(item)

        for item in results:
            print(item)
        assert len(results) == 2

        results.clear()
        asyncio.get_event_loop().run_until_complete(model.ainvoke("a", "messages"))

        for item in results:
            print(item)
        assert len(results) == 2
