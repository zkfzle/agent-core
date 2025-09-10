import sys
import time
import types
import uuid
from unittest.mock import Mock

from jiuwen.core.common.constants.constant import INTERACTION
from jiuwen.core.component.workflow_comp import SubWorkflowComponent
from jiuwen.core.graph.executable import Input
from jiuwen.core.graph.interrupt.interactive_input import InteractiveInput
from jiuwen.core.stream.base import BaseStreamMode

fake_base = types.ModuleType("base")
fake_base.logger = Mock()

fake_exception_module = types.ModuleType("base")
fake_exception_module.JiuWenBaseException = Mock()

sys.modules["jiuwen.core.common.logging.base"] = fake_base
sys.modules["jiuwen.core.common.exception.base"] = fake_exception_module

import asyncio
import unittest
from collections.abc import Callable

from jiuwen.core.component.condition.array import ArrayCondition
from jiuwen.core.component.loop_callback.intermediate_loop_var import IntermediateLoopVarCallback
from jiuwen.core.component.loop_callback.output import OutputCallback
from jiuwen.core.component.loop_comp import LoopGroup, LoopComponent
from jiuwen.core.component.set_variable_comp import SetVariableComponent
from jiuwen.core.context.config import Config
from jiuwen.core.context.context import Context, WorkflowContext
from jiuwen.core.context.state import InMemoryState
from jiuwen.core.graph.base import Graph
from jiuwen.core.workflow.base import WorkflowConfig, Workflow
from jiuwen.graph.pregel.graph import PregelGraph
from test_node import AddTenNode, CommonNode
from tests.unit_tests.workflow.test_mock_node import InteractiveNode4StreamCp, MockStartNode, MockEndNode, Node4Cp, \
    MockStartNode4Cp, InteractiveNode4Cp, AddTenNode4Cp


def create_context(session_id: str = None) -> Context:
    return WorkflowContext(config=Config(), state=InMemoryState(), store=None, session_id=session_id)


def create_graph() -> Graph:
    return PregelGraph()


def create_flow() -> Workflow:
    return Workflow(workflow_config=WorkflowConfig(), graph=create_graph())


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def invoke_workflow(self, inputs: Input, context: Context, flow: Workflow):
        return asyncio.run(flow.invoke(inputs=inputs, context=context))

    async def collect_reuslt(self, inputs: Input, context: Context, flow: Workflow):
        return [result async for result in
                flow.stream(inputs=inputs, context=context, stream_modes=[BaseStreamMode.OUTPUT])]

    def stream_workflow(self, inputs: Input, context: Context, flow: Workflow):
        return asyncio.run(self.collect_reuslt(inputs, context, flow))

    def assert_workflow_invoke(self, inputs: Input, context: Context, flow: Workflow, expect_results: dict = None,
                               checker: Callable = None):
        if expect_results is not None:
            assert self.invoke_workflow(inputs, context, flow) == expect_results
        elif checker is not None:
            checker(self.invoke_workflow(inputs, context, flow))

    def test_simple_workflow(self):
        """
        graph : start->a->end
        """
        mock_start = MockStartNode4Cp("start")
        mock_node = Node4Cp("a")
        flow = create_flow()
        flow.set_start_comp("start", mock_start,
                            inputs_schema={
                                "a": "${user.inputs.a}",
                                "b": "${user.inputs.b}",
                                "c": 1,
                                "d": [1, 2, 3]})
        flow.add_workflow_comp("a", mock_node,
                               inputs_schema={
                                   "aa": "${start.a}",
                                   "ac": "${start.c}"})
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={
                              "result": "${a.aa}"})
        flow.add_connection("start", "a")
        flow.add_connection("a", "end")
        session_id = uuid.uuid4().hex
        try:
            self.invoke_workflow({"a": 1, "b": "haha"}, create_context(session_id=session_id), flow)
        except Exception as e:
            assert str(e) == 'value < 20'
        assert mock_start.runtime == 1
        assert mock_node.runtime == 1

        try:
            self.invoke_workflow(InteractiveInput(), create_context(session_id=session_id), flow)
        except Exception as e:
            assert str(e) == 'value < 20'
        assert mock_start.runtime == 1
        assert mock_node.runtime == 2

    def test_workflow_comp(self):
        """
        graph : start->a(start->a1->a2->a3->end)->end
        """
        mock_start = MockStartNode4Cp("a1")
        mock_node = Node4Cp("a2")
        subflow = create_flow()
        subflow.set_start_comp("a1", mock_start,
                               inputs_schema={
                                   "a": "${user.inputs.a}",
                                   "b": "${user.inputs.b}",
                                   "c": 1,
                                   "d": [1, 2, 3]})
        subflow.add_workflow_comp("a2", mock_node,
                                  inputs_schema={
                                      "aa": "${start.a}",
                                      "ac": "${start.c}"})
        subflow.set_end_comp("a3", MockEndNode("a3"),
                             inputs_schema={
                                 "result": "${a.aa}"})
        subflow.add_connection("a1", "a2")
        subflow.add_connection("a2", "a3")

        flow = create_flow()
        flow.set_start_comp("start", MockStartNode("start"),
                            inputs_schema={
                                "a": "${user.inputs.a}",
                                "b": "${user.inputs.b}",
                                "c": 1,
                                "d": [1, 2, 3]})
        flow.add_workflow_comp("a", SubWorkflowComponent(subflow),
                               inputs_schema={
                                   "aa": "${start.a}",
                                   "ac": "${start.c}"})
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={
                              "result": "${a.aa}"})
        flow.add_connection("start", "a")
        flow.add_connection("a", "end")
        session_id = uuid.uuid4().hex
        try:
            self.invoke_workflow({"a": 1, "b": "haha"}, create_context(session_id=session_id), flow)
        except Exception as e:
            assert str(e) == 'value < 20'
        assert mock_start.runtime == 1
        assert mock_node.runtime == 1

        time.sleep(0.1)
        try:
            self.invoke_workflow(InteractiveInput(), create_context(session_id=session_id), flow)
        except Exception as e:
            assert str(e) == 'value < 20'
        assert mock_start.runtime == 1
        assert mock_node.runtime == 2

    def test_workflow_with_loop(self):
        flow = create_flow()
        flow.set_start_comp("s", MockStartNode("s"))
        flow.set_end_comp("e", MockEndNode("e"),
                          inputs_schema={"array_result": "${b.array_result}", "user_var": "${b.user_var}"})
        flow.add_workflow_comp("a", CommonNode("a"),
                               inputs_schema={"array": "${input_array}"})
        flow.add_workflow_comp("b", CommonNode("b"),
                               inputs_schema={"array_result": "${l.results}", "user_var": "${l.user_var}"})

        # create  loop: (1->2->3)
        loop_group = LoopGroup(WorkflowConfig(), PregelGraph())
        loop_group.add_workflow_comp("1", AddTenNode("1"), inputs_schema={"source": "${arrLoopVar.item}"})
        loop_group.add_workflow_comp("2", AddTenNode4Cp("2"),
                                     inputs_schema={"source": "${intermediateLoopVar.user_var}"})
        loop_group.add_workflow_comp("3", SetVariableComponent(
                                                      {"${intermediateLoopVar.user_var}": "${2.result}"}))
        loop_group.start_comp("1")
        loop_group.end_comp("3")
        loop_group.add_connection("1", "2")
        loop_group.add_connection("2", "3")
        output_callback = OutputCallback(
                                         {"results": "${1.result}", "user_var": "${intermediateLoopVar.user_var}"})
        intermediate_callback = IntermediateLoopVarCallback({"user_var": "${input_number}"})

        loop = LoopComponent("l", loop_group, PregelGraph(), ArrayCondition("arrLoopVar", {"item": "${a.array}"}),
                             callbacks=[output_callback, intermediate_callback])

        flow.add_workflow_comp("l", loop, inputs_schema={"input_number": "${input_number}"})

        # s->a->(1->2->3)->b->e
        flow.add_connection("s", "a")
        flow.add_connection("a", "l")
        flow.add_connection("l", "b")
        flow.add_connection("b", "e")

        session_id = uuid.uuid4().hex
        try:
            expect_e = Exception()
            result = self.invoke_workflow({"input_array": [1, 2, 3], "input_number": 1},
                                          create_context(session_id=session_id), flow)
        except Exception as e:
            expect_e = e
        assert str(expect_e) == "inner error: 1"
        try:
            expect_e = Exception()
            result = self.invoke_workflow(InteractiveInput(), create_context(session_id=session_id), flow)
        except Exception as e:
            expect_e = e
        assert str(expect_e) == "inner error: 11"
        try:
            expect_e = Exception()
            result = self.invoke_workflow(InteractiveInput(), create_context(session_id=session_id), flow)
        except Exception as e:
            expect_e = e
        assert str(expect_e) == "inner error: 21"
        try:
            expect_e = Exception()
            result = self.invoke_workflow(InteractiveInput(), create_context(session_id=session_id), flow)
            assert result == {"array_result": [11, 12, 13], "user_var": 31}
        except Exception as e:
            assert True
            expect_e = e
        assert str(expect_e) == ""

        try:
            expect_e = Exception()
            result = self.invoke_workflow({"input_array": [4, 5], "input_number": 2},
                                          create_context(session_id=session_id), flow)
        except Exception as e:
            expect_e = e
        assert str(expect_e) == "inner error: 2"
        try:
            expect_e = Exception()
            result = self.invoke_workflow(InteractiveInput(), create_context(session_id=session_id), flow)
        except Exception as e:
            expect_e = e
        assert str(expect_e) == "inner error: 12"
        try:
            expect_e = Exception()
            result = self.invoke_workflow(InteractiveInput(), create_context(session_id=session_id), flow)
            assert result == {"array_result": [14, 15], "user_var": 22}
        except Exception as e:
            assert True
            expect_e = e
        assert str(expect_e) == ""

    def test_workflow_with_loop_interactive(self):
        flow = create_flow()
        flow.set_start_comp("s", MockStartNode("s"))
        flow.set_end_comp("e", MockEndNode("e"),
                          inputs_schema={"array_result": "${b.array_result}", "user_var": "${b.user_var}"})
        flow.add_workflow_comp("a", CommonNode("a"),
                               inputs_schema={"array": "${input_array}"})
        flow.add_workflow_comp("b", CommonNode("b"),
                               inputs_schema={"array_result": "${l.results}", "user_var": "${l.user_var}"})

        # create  loop: (1->2->3)
        loop_group = LoopGroup(WorkflowConfig(), PregelGraph())
        loop_group.add_workflow_comp("1", AddTenNode("1"), inputs_schema={"source": "${arrLoopVar.item}"})
        loop_group.add_workflow_comp("2", InteractiveNode4Cp("2"),
                                     inputs_schema={"source": "${intermediateLoopVar.user_var}"})
        loop_group.add_workflow_comp("3", SetVariableComponent(
                                                      {"${intermediateLoopVar.user_var}": "${2.result}"}))
        loop_group.start_comp("1")
        loop_group.end_comp("3")
        loop_group.add_connection("1", "2")
        loop_group.add_connection("2", "3")
        output_callback = OutputCallback({"results": "${1.result}", "user_var": "${intermediateLoopVar.user_var}"})
        intermediate_callback = IntermediateLoopVarCallback({"user_var": "${input_number}"})

        loop = LoopComponent("l", loop_group, PregelGraph(), ArrayCondition("arrLoopVar", {"item": "${a.array}"}),
                             callbacks=[output_callback, intermediate_callback])

        flow.add_workflow_comp("l", loop, inputs_schema={"input_number": "${input_number}"})

        # s->a->(1->2->3)->b->e
        flow.add_connection("s", "a")
        flow.add_connection("a", "l")
        flow.add_connection("l", "b")
        flow.add_connection("b", "e")

        session_id = uuid.uuid4().hex

        # 每次节点2有两个等待用户输入，索引为：0、1，循环三次，共6个输入
        res = self.invoke_workflow({"input_array": [1, 2, 3], "input_number": 1}, create_context(session_id=session_id),
                                   flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 0, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 1, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 0, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 1, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 0, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 1, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, {"array_result": [11, 12, 13], "user_var": None})

        # 重复执行
        res = self.invoke_workflow({"input_array": [4, 5], "input_number": 2}, create_context(session_id=session_id),
                                   flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 0, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 1, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 0, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 1, 'payload': ('l.2', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})

        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, {"array_result": [14, 15], "user_var": None})

    def test_simple_interactive_workflow(self):
        """
        graph : start->a->end
        """
        start_node = MockStartNode4Cp("start")
        flow = create_flow()
        flow.set_start_comp("start", start_node,
                            inputs_schema={
                                "a": "${user.inputs.a}",
                                "b": "${user.inputs.b}",
                                "c": 1,
                                "d": [1, 2, 3]})
        flow.add_workflow_comp("a", InteractiveNode4Cp("a"),
                               inputs_schema={
                                   "aa": "${start.a}",
                                   "ac": "${start.c}"})
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={
                              "result": "${a.aa}"})
        flow.add_connection("start", "a")
        flow.add_connection("a", "end")

        session_id = uuid.uuid4().hex

        res = self.invoke_workflow({"a": 1, "b": "haha"}, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'type': '__interaction__', 'index': 0, 'payload': ('a', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update(res[0].get("payload")[0], {"aa": "any key"})
        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, [{'index': 1,
                               'payload': ('a', 'Please enter any key'),
                               'type': '__interaction__'}])
        self.assertEqual(start_node.runtime, 1)

    def test_simple_stream_interactive_workflow(self):
        """
        graph : start->a->end
        """
        start_node = MockStartNode4Cp("start")
        flow = create_flow()
        flow.set_start_comp("start", start_node,
                            inputs_schema={
                                "a": "${user.inputs.a}",
                                "b": "${user.inputs.b}",
                                "c": 1,
                                "d": [1, 2, 3]})
        flow.add_workflow_comp("a", InteractiveNode4StreamCp("a"),
                               inputs_schema={
                                   "aa": "${start.a}",
                                   "ac": "${start.c}"})
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={
                              "result": "${a.aa}"})
        flow.add_connection("start", "a")
        flow.add_connection("a", "end")
        interaction_node = None
        interaction_msg = None

        session_id = uuid.uuid4().hex

        for res in self.stream_workflow({"a": 1, "b": "haha"}, create_context(session_id=session_id), flow):
            if res.type == INTERACTION:
                interaction_node = res.payload[0]
                interaction_msg = res.payload[1]
        self.assertEqual(interaction_node, "a")
        self.assertEqual(interaction_msg, "Please enter any key")
        user_input = InteractiveInput()
        user_input.update(interaction_node, {"aa": "any key"})
        result = None
        for res in self.stream_workflow(user_input, create_context(session_id=session_id), flow):
            if res.type == "output":
                self.assertEqual(res.type, "output")
                self.assertEqual(res.payload[0], "a")
                result = res.payload[1]
        self.assertEqual(result, {"aa": "any key"})
        self.assertEqual(start_node.runtime, 1)

    def test_simple_concurrent_interactive_workflow(self):
        """
        graph : start->a->end
                     ->b->end
        """
        start_node = MockStartNode4Cp("start")
        flow = create_flow()
        flow.set_start_comp("start", start_node,
                            inputs_schema={
                                "a": "${user.inputs.a}",
                                "b": "${user.inputs.b}",
                                "c": 1,
                                "d": [1, 2, 3]})
        flow.add_workflow_comp("a", InteractiveNode4Cp("a"),
                               inputs_schema={
                                   "aa": "${start.a}",
                                   "ac": "${start.c}"})
        flow.add_workflow_comp("b", InteractiveNode4Cp("b"),
                               inputs_schema={
                                   "aa": "${start.a}",
                                   "ac": "${start.c}"})
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={
                              "result": ["${a.aa}", "${b.aa}"]})
        flow.add_connection("start", "a")
        flow.add_connection("start", "b")
        flow.add_connection("a", "end")
        flow.add_connection("b", "end")

        session_id = uuid.uuid4().hex

        res = self.invoke_workflow({"a": 1, "b": "haha"}, create_context(session_id=session_id), flow)
        self.assertCountEqual(res, [{'type': '__interaction__', 'index': 0, 'payload': ('a', 'Please enter any key')},
                                    {'type': '__interaction__', 'index': 0, 'payload': ('b', 'Please enter any key')}])
        user_input = InteractiveInput()
        user_input.update("a", {"aa": "any key a"})
        user_input.update("b", {"aa": "any key b"})
        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertCountEqual(res, [{'index': 1,
                               'payload': ('a', 'Please enter any key'),
                               'type': '__interaction__'}, {'index': 1,
                               'payload': ('b', 'Please enter any key'),
                               'type': '__interaction__'}])
        self.assertEqual(start_node.runtime, 1)
        user_input = InteractiveInput()
        user_input.update("a", {"aa": "any key a"})
        user_input.update("b", {"aa": "any key b"})
        res = self.invoke_workflow(user_input, create_context(session_id=session_id), flow)
        self.assertEqual(res, {"result": ["any key a", "any key b"]})

