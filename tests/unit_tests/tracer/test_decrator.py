import unittest
from unittest.mock import MagicMock

from jiuwen.core.common.logging import logger
from jiuwen.core.tracer.decorator import decrate_tool_with_trace
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


class TestDecator(unittest.TestCase):

    def test_demo(self):
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
        tool.invoke(inputs={"a": "a"})
        assert len(results) == 2

