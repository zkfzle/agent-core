import asyncio
import unittest
from collections.abc import Callable

from jiuwen.core.common.logging import logger
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.start_comp import Start



from jiuwen.core.context.config import Config
from jiuwen.core.context.context import Context, WorkflowContext
from jiuwen.core.context.state import InMemoryState, ReadableStateLike
from jiuwen.core.graph.base import Graph
from jiuwen.core.workflow.base import WorkflowConfig, Workflow
from jiuwen.core.workflow.workflow_config import ComponentAbility
from jiuwen.graph.pregel.graph import PregelGraph
from tests.unit_tests.tracer.test_workflow import create_context_with_tracer
from tests.unit_tests.workflow.test_mock_node import MockStartNode, MockEndNode, Node1, StreamNode, StreamCompNode


def create_context() -> Context:
    return WorkflowContext(config=Config(), state=InMemoryState(), store=None)


def create_graph() -> Graph:
    return PregelGraph()


def create_flow() -> Workflow:
    return Workflow(workflow_config=DEFAULT_WORKFLOW_CONFIG, graph=create_graph())


DEFAULT_WORKFLOW_CONFIG = WorkflowConfig(metadata={})


class EndNodeTest(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def invoke_workflow(self, inputs: dict, context: Context, flow: Workflow):
        feature = asyncio.ensure_future(flow.invoke(inputs=inputs, context=context))
        self.loop.run_until_complete(feature)
        return feature.result()

    def assert_workflow_invoke(self, inputs: dict, context: Context, flow: Workflow, expect_results: dict = None,
                               checker: Callable = None):
        if expect_results is not None:
            assert self.invoke_workflow(inputs, context, flow) == expect_results
        elif checker is not None:
            checker(self.invoke_workflow(inputs, context, flow))

    def test_simple_template_workflow(self):
        # flow1: start -> a -> end
        flow = create_flow()
        flow.set_start_comp("start", Start("start",{"userFields":{"inputs":[],"outputs":[]},"systemFields":{"input":[{"id":"query","type":"String","required":"true","sourceType":"ref"}]}}),
                            inputs_schema={
                                "systemFields": {"query": "${a}"},
                                "userFields": {},
                                "response_node": "${response_mode}",
                                "d": "${b}"})
        flow.add_workflow_comp("a", Node1("a"),
                               inputs_schema={
                                   "aa": "${start.d}",
                                   "ac": "${start.d}"})
        flow.set_end_comp("end", End("end", "end", {"responseTemplate": "hello:{{end_input}}"}),
                          inputs_schema={
                              "userFields": {"end_input": "${start.userFields.d}"},
                              "response_mode": "${start.userFields.response_node}"})
        flow.add_connection("start", "a")
        flow.add_connection("a", "end")
        self.assert_workflow_invoke({"a": 1, "b": "haha"}, create_context(), flow, expect_results={'output': {}, 'responseContent': 'hello:haha'})


    def test_simple_output_schema_workflow(self):
        # flow1: start -> a -> end
        flow = create_flow()
        flow.set_start_comp("start", Start("start",{"userFields":{"inputs":[],"outputs":[]},"systemFields":{"input":[{"id":"query","type":"String","required":"true","sourceType":"ref"}]}}),
                            inputs_schema={
                                "systemFields": {"query": "${a}"},
                                "userFields": {},
                                "response_node": "${response_mode}",
                                "d": "${b}"})
        flow.add_workflow_comp("a", Node1("a"),
                               inputs_schema={
                                   "aa": "${start.d}",
                                   "ac": "${start.d}"})
        flow.set_end_comp("end", End("end", "end",{}),
                          inputs_schema={
                              "userFields": {"end_input": "${start.userFields.d}"},
                              "response_mode": "${start.userFields.response_node}"},
                        )
        flow.add_connection("start", "a")
        flow.add_connection("a", "end")
        self.assert_workflow_invoke({"a": 1, "b": "haha"}, create_context(), flow, expect_results={'output': {'end_input': 'haha'}, 'responseContent': ''})

    def test_end_stream_workflow(self):
        async def stream_workflow():
            flow = create_flow()
            flow.set_start_comp("start", Start("start", {"userFields": {"inputs": [], "outputs": []}, "systemFields": {
                "input": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]}}),
                                inputs_schema={
                                    "systemFields": {"query": "${a}"},
                                    "userFields": {"d": "${a}"},
                                    "response_node": "${response_mode}",
                                    "d": "${a}"})

            flow.add_workflow_comp("a", StreamCompNode("a"), inputs_schema={"value": "${a}"},
                                   comp_ability=[ComponentAbility.STREAM], wait_for_all=True)


            flow.set_end_comp("end", End("end", "end", {"responseTemplate": "hello:{{end_input}}"}),
                              inputs_schema={
                                  "userFields": {"end_input": "${start.userFields.d}"}},response_mode="streaming")
            flow.add_connection("start", "a")
            flow.add_stream_connection("a", "end")

            index = 0
            async for chunk in flow.stream({"a": 1, "b": "haha"}, create_context_with_tracer()):
                logger.info("stream chunk: {%s}", chunk)
                index += 1

        self.loop.run_until_complete(stream_workflow())



    def test_end_batch_stream_workflow(self):

        async def stream_workflow():
            flow = create_flow()
            flow.set_start_comp("start", Start("start", {"userFields": {"inputs": [], "outputs": []}, "systemFields": {
                "input": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]}}),
                                inputs_schema={
                                    "systemFields": {"query": "${a}"},
                                    "userFields": {"d": "${a}"},
                                    "response_node": "${response_mode}",
                                    "d": "${a}"})

            flow.add_workflow_comp("a", StreamCompNode("a"), inputs_schema={"value": "${a}"},
                                   comp_ability=[ComponentAbility.STREAM], wait_for_all=True)

            flow.set_end_comp("end", End("end", "end", {"responseTemplate": "hello:{{value}}"}),
                              stream_inputs_schema={"value": "${a.value}"},inputs_schema={"value": "${a.value}"},response_mode="streaming")
            flow.add_connection("start", "a")
            flow.add_stream_connection("a", "end")

            index = 0
            async for chunk in flow.stream({"a": 1, "b": "haha"}, create_context_with_tracer()):
                logger.info("stream chunk: {%s}", chunk)
                index += 1

        self.loop.run_until_complete(stream_workflow())