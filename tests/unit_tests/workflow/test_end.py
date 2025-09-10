import asyncio
import unittest
from collections.abc import Callable

from jiuwen.core.common.logging import logger
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.start_comp import Start
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.runtime import BaseRuntime, WorkflowRuntime
from jiuwen.core.runtime.state import InMemoryState
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import ComponentAbility
from tests.unit_tests.workflow.test_mock_node import Node1, StreamCompNode


class EndNodeTest(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def invoke_workflow(self, inputs: dict, context: BaseRuntime, flow: Workflow):
        feature = asyncio.ensure_future(flow.invoke(inputs=inputs, context=context))
        self.loop.run_until_complete(feature)
        return feature.result()

    def assert_workflow_invoke(self, inputs: dict, context: BaseRuntime, flow: Workflow, expect_results: dict = None,
                               checker: Callable = None):
        if expect_results is not None:
            assert self.invoke_workflow(inputs, context, flow) == expect_results
        elif checker is not None:
            checker(self.invoke_workflow(inputs, context, flow))

    def test_simple_template_workflow(self):
        # flow1: start -> a -> end
        flow = Workflow()
        flow.set_start_comp("start", Start(
            {"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]}),
                            inputs_schema={
                                "query": "${a}",
                                "response_node": "${response_mode}",
                                "d": "${b}"})
        flow.add_workflow_comp("a", Node1("a"),
                               inputs_schema={
                                   "aa": "${start.d}",
                                   "ac": "${start.d}"})
        flow.set_end_comp("end", End({"responseTemplate": "hello:{{end_input}}"}),
                          inputs_schema={
                              "end_input": "${start.d}",
                              "response_mode": "${start.response_node}"})
        flow.add_connection("start", "a")
        flow.add_connection("a", "end")
        self.assert_workflow_invoke({"a": 1, "b": "haha"}, WorkflowRuntime(), flow, expect_results={'output': {}, 'responseContent': 'hello:haha'})


    def test_simple_output_schema_workflow(self):
        # flow1: start -> a -> end
        flow = Workflow()
        flow.set_start_comp("start", Start(
            {"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]}),
                            inputs_schema={
                                "query": "${a}",
                                "response_node": "${response_mode}",
                                "d": "${b}"})
        flow.add_workflow_comp("a", Node1("a"),
                               inputs_schema={
                                   "aa": "${start.d}",
                                   "ac": "${start.d}"})
        flow.set_end_comp("end", End(),
                          inputs_schema={
                              "end_input": "${start.d}",
                              "response_mode": "${start.response_node}"},
                          )
        flow.add_connection("start", "a")
        flow.add_connection("a", "end")
        self.assert_workflow_invoke({"a": 1, "b": "haha"}, WorkflowRuntime(), flow, expect_results={'output': {'end_input': 'haha'}, 'responseContent': ''})

    def test_end_stream_workflow(self):
        async def stream_workflow():
            flow = Workflow()
            start = Start({"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]})
            flow.set_start_comp("start", start,
                                inputs_schema={
                                    "query": "${a}",
                                    "response_node": "${response_mode}",
                                    "d": "${a}"})

            flow.add_workflow_comp("a", StreamCompNode("a"), inputs_schema={"value": "${a}"},
                                   comp_ability=[ComponentAbility.STREAM], wait_for_all=True)

            flow.set_end_comp("end", End({"responseTemplate": "hello:{{end_input}}"}),
                              inputs_schema={"end_input": "${start.d}"}, response_mode="streaming")
            flow.add_connection("start", "a")
            flow.add_stream_connection("a", "end")

            index = 0
            async for chunk in flow.stream({"a": 1, "b": "haha"},
                                           WorkflowRuntime(config=Config(), state=InMemoryState(), store=None)):
                logger.info("stream chunk: {%s}", chunk)
                index += 1

        self.loop.run_until_complete(stream_workflow())



    def test_end_batch_stream_workflow(self):

        async def stream_workflow():
            flow = Workflow()
            start = Start({"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]})
            input_schema = {
                "query": "${a}",
                "response_node": "${response_mode}",
                "d": "${a}"
            }
            flow.set_start_comp("start", start, inputs_schema=input_schema)

            flow.add_workflow_comp("a", StreamCompNode("a"), inputs_schema={"value": "${a}"},
                                   comp_ability=[ComponentAbility.STREAM], wait_for_all=True)

            flow.set_end_comp("end", End({"responseTemplate": "hello:{{value}}"}),
                              stream_inputs_schema={"value": "${a.value}"}, inputs_schema={"value": "${a.value}"},
                              response_mode="streaming")
            flow.add_connection("start", "a")
            flow.add_stream_connection("a", "end")

            index = 0
            async for chunk in flow.stream({"a": 1, "b": "haha"},
                                           WorkflowRuntime(config=Config(), state=InMemoryState(), store=None)):
                logger.info("stream chunk: {%s}", chunk)
                index += 1

        self.loop.run_until_complete(stream_workflow())