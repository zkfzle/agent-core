import os
import unittest
from typing import Literal
from unittest.mock import patch

from jiuwen.core.common.configs.env_constant import WORKFLOW_DRAWABLE
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.component.branch_comp import BranchComponent
from jiuwen.core.component.branch_router import BranchRouter
from jiuwen.core.component.condition.number import NumberCondition
from jiuwen.core.component.loop_callback.intermediate_loop_var import IntermediateLoopVarCallback
from jiuwen.core.component.loop_callback.output import OutputCallback
from jiuwen.core.component.loop_comp import AdvancedLoopComponent, LoopGroup
from jiuwen.core.component.set_variable_comp import SetVariableComponent
from jiuwen.core.component.workflow_comp import SubWorkflowComponent
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import ComponentAbility
from jiuwen.graph.visualization.drawable import Drawable
from tests.unit_tests.workflow.test_mock_node import MockStartNode, Node1, MockEndNode, StreamCompNode, CollectCompNode
from tests.unit_tests.workflow.test_node import CommonNode, AddTenNode


class WorkflowTest(unittest.TestCase):
    @patch.dict(os.environ, {WORKFLOW_DRAWABLE: "true"})
    def test_visualize_simple_workflow(self):
        # flow: start -> a -> end
        flow = Workflow()
        flow.set_start_comp("start", MockStartNode("start"),
                            inputs_schema={
                                "a": "${a}",
                                "b": "${b}",
                                "c": 1,
                                "d": [1, 2, 3]})
        flow.add_workflow_comp("a", Node1("a"),
                               inputs_schema={
                                   "aa": "${start.a}",
                                   "ac": "${start.c}"})
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={
                              "result": "${a.aa}"})
        flow.add_connection("start", "a")
        flow.add_connection("a", "end")
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("start")
\tnode_2["a"]
\tnode_3("end")
\tnode_1 --> node_2
\tnode_2 --> node_3
"""
        self.assertEqual(flow.to_mermaid("jiuwen workflow"), mermaid_script)

    @patch.dict(os.environ, {WORKFLOW_DRAWABLE: "true"})
    def test_visualize_simple_stream_workflow(self):
        # flow: start -> a ---> b -> end
        flow = Workflow()
        flow.set_start_comp("start", MockStartNode("start"), inputs_schema={"a": "${a}"})
        flow.add_workflow_comp("a", StreamCompNode("a"), inputs_schema={"value": "${start.a}"},
                               comp_ability=[ComponentAbility.STREAM], wait_for_all=True)
        flow.add_workflow_comp("b", CollectCompNode("b"), inputs_schema={"value": "${a.value}"},
                               stream_inputs_schema={"value": "${a.value}"}, comp_ability=[ComponentAbility.COLLECT],
                               wait_for_all=True)
        flow.set_end_comp("end", MockEndNode("end"), inputs_schema={"result1": "${b.value}"})
        flow.add_connection("start", "a")
        flow.add_stream_connection("a", "b")
        flow.add_connection("b", "end")
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("start")
\tnode_2["a"]
\tnode_3["b"]
\tnode_4("end")
\tnode_1 --> node_2
\tnode_2 -->|stream| node_3
\tnode_3 --> node_4
"""
        self.assertEqual(flow.to_mermaid("jiuwen workflow"), mermaid_script)

    @patch.dict(os.environ, {WORKFLOW_DRAWABLE: "true"})
    def test_visualize_workflow_with_branch_comp(self):
        # flow: start -> sw[a,b] -> end
        flow = Workflow()
        flow.set_start_comp("start", MockStartNode("start"))
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={"a": "${a.result}", "b": "${b.result}"})

        sw = BranchComponent()
        sw.add_branch("${a} <= 10", ["b"], "1")
        sw.add_branch("${a} > 10", ["a"], "2")

        flow.add_workflow_comp("sw", sw)

        flow.add_workflow_comp("a", CommonNode("a"),
                               inputs_schema={"result": "${a}"})

        flow.add_workflow_comp("b", AddTenNode("b"),
                               inputs_schema={"source": "${a}"})

        flow.add_connection("start", "sw")
        flow.add_connection("a", "end")
        flow.add_connection("b", "end")

        mermaid_script = """---
title: 
---
flowchart TB
\tnode_1("start")
\tnode_2("end")
\tnode_3["sw"]
\tnode_4["a"]
\tnode_5["b"]
\tnode_3 -.->|"${a} <= 10"| node_5
\tnode_3 -.->|"${a} > 10"| node_4
\tnode_1 --> node_3
\tnode_4 --> node_2
\tnode_5 --> node_2
"""
        self.assertEqual(flow.to_mermaid(), mermaid_script)

    @patch.dict(os.environ, {WORKFLOW_DRAWABLE: "true"})
    def test_visualize_workflow_with_branch_router(self):
        """
        flow: start -> condition[a,b] -> end
        """
        flow = Workflow()
        flow.set_start_comp("start", MockStartNode("start"),
                            inputs_schema={"a": "${a}",
                                           "b": "${b}",
                                           "c": 1,
                                           "d": [1, 2, 3]})

        router = BranchRouter()
        router.add_branch("${start.a} is not None", "a")
        router.add_branch("${start.b} is not None", "b")

        flow.add_conditional_connection("start", router=router)
        flow.add_workflow_comp("a", Node1("a"), inputs_schema={"a": "${start.a}", "b": "${start.c}"})
        flow.add_workflow_comp("b", Node1("b"), inputs_schema={"b": "${start.b}"})
        flow.set_end_comp("end", MockEndNode("end"), {"result1": "${a.a}", "result2": "${b.b}"})
        flow.add_connection("a", "end")
        flow.add_connection("b", "end")
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("start")
\tnode_2["a"]
\tnode_3["b"]
\tnode_4("end")
\tnode_1 -.->|"${start.a} is not None"| node_2
\tnode_1 -.->|"${start.b} is not None"| node_3
\tnode_2 --> node_4
\tnode_3 --> node_4
"""
        self.assertEqual(flow.to_mermaid("jiuwen workflow"), mermaid_script)

    @patch.dict(os.environ, {WORKFLOW_DRAWABLE: "true"})
    def test_visualize_workflow_with_condition(self):
        """
        start -> condition[a,b] -> end
        """
        flow = Workflow()
        flow.set_start_comp("start", MockStartNode("start"),
                            inputs_schema={"a": "${a}",
                                           "b": "${b}",
                                           "c": 1,
                                           "d": [1, 2, 3]})

        # Literal is for visualization
        def router(runtime: BaseRuntime) -> Literal["a", "b"]:
            val = runtime.state().get_global("start.a")
            if val is not None:
                return "a"
            val = runtime.state().get_global("start.b")
            if val is not None:
                return "b"
            return "a"

        flow.add_conditional_connection("start", router=router)
        flow.add_workflow_comp("a", Node1("a"), inputs_schema={"a": "${start.a}", "b": "${start.c}"})
        flow.add_workflow_comp("b", Node1("b"), inputs_schema={"b": "${start.b}"})
        flow.set_end_comp("end", MockEndNode("end"), {"result1": "${a.a}", "result2": "${b.b}"})
        flow.add_connection("a", "end")
        flow.add_connection("b", "end")

        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("start")
\tnode_2["a"]
\tnode_3["b"]
\tnode_4("end")
\tnode_1 -.-> node_2
\tnode_1 -.-> node_3
\tnode_2 --> node_4
\tnode_3 --> node_4
"""
        self.assertEqual(flow.to_mermaid("jiuwen workflow"), mermaid_script)

    @patch.dict(os.environ, {WORKFLOW_DRAWABLE: "true"})
    def test_visualize_sub_workflow(self):
        # flow: start -> a -> (sub_start -> sub_a -> sub_end) -> end
        sub_flow = Workflow()
        sub_flow.set_start_comp("sub_start", MockStartNode("start"),
                                inputs_schema={
                                    "a": "${a}",
                                    "b": "${b}",
                                    "c": 1,
                                    "d": [1, 2, 3]})
        sub_flow.add_workflow_comp("sub_a", Node1("a"),
                                   inputs_schema={
                                       "aa": "${start.a}",
                                       "ac": "${start.c}"})
        sub_flow.set_end_comp("sub_end", MockEndNode("end"),
                              inputs_schema={
                                  "result": "${a.aa}"})
        sub_flow.add_connection("sub_start", "sub_a")
        sub_flow.add_connection("sub_a", "sub_end")

        flow = Workflow()
        flow.set_start_comp("start", MockStartNode("start"),
                            inputs_schema={
                                "a": "${a}",
                                "b": "${b}",
                                "c": 1,
                                "d": [1, 2, 3]})
        flow.add_workflow_comp("a", Node1("a"),
                               inputs_schema={
                                   "aa": "${start.a}",
                                   "ac": "${start.c}"})
        flow.add_workflow_comp("sub_flow", SubWorkflowComponent(sub_flow))
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={
                              "result": "${a.aa}"})
        flow.add_connection("start", "a")
        flow.add_connection("a", "sub_flow")
        flow.add_connection("sub_flow", "end")

        # no expand sub graph
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("start")
\tnode_2["a"]
\tnode_3["sub_flow"]
\tnode_4("end")
\tnode_1 --> node_2
\tnode_2 --> node_3
\tnode_3 --> node_4
"""
        self.assertEqual(flow.to_mermaid("jiuwen workflow"), mermaid_script)

        # expand sub graph
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("start")
\tnode_2["a"]
\tnode_7("end")
\tsubgraph node_6 ["sub_flow"]
\tdirection TB
\tnode_3("sub_start")
\tnode_4["sub_a"]
\tnode_5("sub_end")
end
\tnode_1 --> node_2
\tnode_2 --> node_3
\tnode_5 --> node_7
\tnode_3 --> node_4
\tnode_4 --> node_5
"""
        self.assertEqual(flow.to_mermaid(title="jiuwen workflow", expand_subgraph=True), mermaid_script)

    @patch.dict(os.environ, {WORKFLOW_DRAWABLE: "true"})
    def test_visualize_multi_layer_sub_workflow(self):
        # flow: start -> a -> (sub_start -> sub_a -> (sub_sub_start -> sub_sub_a -> sub_sub_end) -> sub_end) -> end
        sub_sub_flow = Workflow()
        sub_sub_flow.set_start_comp("sub_sub_start", MockStartNode("start"),
                                inputs_schema={
                                    "a": "${a}",
                                    "b": "${b}",
                                    "c": 1,
                                    "d": [1, 2, 3]})
        sub_sub_flow.add_workflow_comp("sub_sub_a", Node1("a"),
                                   inputs_schema={
                                       "aa": "${start.a}",
                                       "ac": "${start.c}"})
        sub_sub_flow.set_end_comp("sub_sub_end", MockEndNode("end"),
                              inputs_schema={
                                  "result": "${a.aa}"})
        sub_sub_flow.add_connection("sub_sub_start", "sub_sub_a")
        sub_sub_flow.add_connection("sub_sub_a", "sub_sub_end")


        sub_flow = Workflow()
        sub_flow.set_start_comp("sub_start", MockStartNode("start"),
                                inputs_schema={
                                    "a": "${a}",
                                    "b": "${b}",
                                    "c": 1,
                                    "d": [1, 2, 3]})
        sub_flow.add_workflow_comp("sub_a", Node1("a"),
                                   inputs_schema={
                                       "aa": "${start.a}",
                                       "ac": "${start.c}"})
        sub_flow.add_workflow_comp("sub_sub_flow", SubWorkflowComponent(sub_sub_flow))
        sub_flow.set_end_comp("sub_end", MockEndNode("end"),
                              inputs_schema={
                                  "result": "${a.aa}"})
        sub_flow.add_connection("sub_start", "sub_a")
        sub_flow.add_connection("sub_a", "sub_sub_flow")
        sub_flow.add_connection("sub_sub_flow", "sub_end")

        flow = Workflow()
        flow.set_start_comp("start", MockStartNode("start"),
                            inputs_schema={
                                "a": "${a}",
                                "b": "${b}",
                                "c": 1,
                                "d": [1, 2, 3]})
        flow.add_workflow_comp("a", Node1("a"),
                               inputs_schema={
                                   "aa": "${start.a}",
                                   "ac": "${start.c}"})
        flow.add_workflow_comp("sub_flow", SubWorkflowComponent(sub_flow))
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={
                              "result": "${a.aa}"})
        flow.add_connection("start", "a")
        flow.add_connection("a", "sub_flow")
        flow.add_connection("sub_flow", "end")

        # no expand sub graph
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("start")
\tnode_2["a"]
\tnode_3["sub_flow"]
\tnode_4("end")
\tnode_1 --> node_2
\tnode_2 --> node_3
\tnode_3 --> node_4
"""
        self.assertEqual(flow.to_mermaid("jiuwen workflow"), mermaid_script)

        # expand first layer sub graph
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("start")
\tnode_2["a"]
\tnode_8("end")
\tsubgraph node_7 ["sub_flow"]
\tdirection TB
\tnode_3("sub_start")
\tnode_4["sub_a"]
\tnode_5["sub_sub_flow"]
\tnode_6("sub_end")
end
\tnode_1 --> node_2
\tnode_2 --> node_3
\tnode_6 --> node_8
\tnode_3 --> node_4
\tnode_4 --> node_5
\tnode_5 --> node_6
"""
        self.assertEqual(flow.to_mermaid(title="jiuwen workflow", expand_subgraph=1), mermaid_script)

        # expand second layer sub graph
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("start")
\tnode_2["a"]
\tnode_11("end")
\tsubgraph node_10 ["sub_flow"]
\tdirection TB
\tnode_3("sub_start")
\tnode_4["sub_a"]
\tnode_9("sub_end")
\tsubgraph node_8 ["sub_sub_flow"]
\tdirection TB
\tnode_5("sub_sub_start")
\tnode_6["sub_sub_a"]
\tnode_7("sub_sub_end")
end
end
\tnode_1 --> node_2
\tnode_2 --> node_3
\tnode_9 --> node_11
\tnode_3 --> node_4
\tnode_4 --> node_5
\tnode_7 --> node_9
\tnode_5 --> node_6
\tnode_6 --> node_7
"""
        self.assertEqual(flow.to_mermaid(title="jiuwen workflow", expand_subgraph=2), mermaid_script)

        # expand all layer sub graph
        self.assertEqual(flow.to_mermaid(title="jiuwen workflow", expand_subgraph=True), mermaid_script)

    @patch.dict(os.environ, {WORKFLOW_DRAWABLE: "true"})
    def test_visualize_workflow_with_loop(self):
        flow = Workflow()
        flow.set_start_comp("s", MockStartNode("s"))
        flow.add_workflow_comp("a", CommonNode("a"))
        # create  loop: (1->2->3)
        loop_group = LoopGroup()
        loop_group.add_workflow_comp("1", AddTenNode("1"), inputs_schema={"source": "${l.index}"})
        loop_group.add_workflow_comp("2", AddTenNode("2"),
                                     inputs_schema={"source": "${l.intermediate_loop_var.user_var}"})
        set_variable_component = SetVariableComponent({"${l.intermediate_loop_var.user_var}": "${2.result}"})
        loop_group.add_workflow_comp("3", set_variable_component)
        loop_group.start_nodes(["1"])
        loop_group.end_nodes(["3"])
        loop_group.add_connection("1", "2")
        loop_group.add_connection("2", "3")
        output_callback = OutputCallback({"results": "${1.result}", "user_var": "${l.intermediate_loop_var.user_var}"})
        intermediate_callback = IntermediateLoopVarCallback({"user_var": "${input_number}"}, "intermediate_loop_var")
        loop = AdvancedLoopComponent(loop_group, NumberCondition("${loop_number}"),
                                     callbacks=[output_callback, intermediate_callback])
        flow.add_workflow_comp("l", loop, inputs_schema={"input_number": "${input_number}"})
        flow.add_workflow_comp("b", CommonNode("b"),
                               inputs_schema={"array_result": "${l.results}", "user_var": "${l.user_var}"})
        flow.set_end_comp("e", MockEndNode("e"),
                          inputs_schema={"array_result": "${b.array_result}", "user_var": "${b.user_var}"})
        # s->a->(1->2->3)->b->e
        flow.add_connection("s", "a")
        flow.add_connection("a", "l")
        flow.add_connection("l", "b")
        flow.add_connection("b", "e")

        # no expand loop
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("s")
\tnode_2["a"]
\tnode_3["l"]
\tnode_4["b"]
\tnode_5("e")
\tnode_3 -.-> node_3
\tnode_1 --> node_2
\tnode_2 --> node_3
\tnode_3 -.-> node_4
\tnode_4 --> node_5
"""
        self.assertEqual(flow.to_mermaid("jiuwen workflow"), mermaid_script)

        # expand loop
        mermaid_script = """---
title: jiuwen workflow
---
flowchart TB
\tnode_1("s")
\tnode_2["a"]
\tnode_7["b"]
\tnode_8("e")
\tsubgraph node_6 ["l"]
\tdirection TB
\tnode_3("1")
\tnode_4["2"]
\tnode_5("3")
end
\tnode_5 -.-> node_3
\tnode_1 --> node_2
\tnode_2 --> node_3
\tnode_5 -.-> node_7
\tnode_7 --> node_8
\tnode_3 --> node_4
\tnode_4 --> node_5
"""
        self.assertEqual(flow.to_mermaid("jiuwen workflow", expand_subgraph=True), mermaid_script)

    def test_drawable_exception(self):
        drawable = Drawable()
        # set start node failed
        node_id = "start"
        with self.assertRaises(JiuWenBaseException) as cm:
            drawable.set_start_node(node_id)
        self.assertEqual(cm.exception.error_code, StatusCode.DRAWABLE_GRAPH_SET_START_NODE_FAILED.code)
        self.assertEqual(cm.exception.message, StatusCode.DRAWABLE_GRAPH_SET_START_NODE_FAILED.errmsg.format(
            node_id=node_id))

        # set end node failed
        node_id = "end"
        with self.assertRaises(JiuWenBaseException) as cm:
            drawable.set_end_node(node_id)
        self.assertEqual(cm.exception.error_code, StatusCode.DRAWABLE_GRAPH_SET_END_NODE_FAILED.code)
        self.assertEqual(cm.exception.message, StatusCode.DRAWABLE_GRAPH_SET_END_NODE_FAILED.errmsg.format(
            node_id=node_id))

        # set end node failed
        node_id = "break"
        with self.assertRaises(JiuWenBaseException) as cm:
            drawable.set_break_node(node_id)
        self.assertEqual(cm.exception.error_code, StatusCode.DRAWABLE_GRAPH_SET_BREAK_NODE_FAILED.code)
        self.assertEqual(cm.exception.message, StatusCode.DRAWABLE_GRAPH_SET_BREAK_NODE_FAILED.errmsg.format(
            node_id=node_id))

        # to mermaid failed
        with self.assertRaises(JiuWenBaseException) as cm:
            drawable.to_mermaid(expand_subgraph=-1)
        self.assertEqual(cm.exception.error_code, StatusCode.DRAWABLE_GRAPH_INVALID_EXPAND_SUBGRAPH.code)
        self.assertEqual(cm.exception.message, StatusCode.DRAWABLE_GRAPH_INVALID_EXPAND_SUBGRAPH.errmsg)
