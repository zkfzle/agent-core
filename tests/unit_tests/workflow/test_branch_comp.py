import asyncio
import unittest

from jiuwen.core.common.constants.component import SUB_WORKFLOW_COMPONENT
from jiuwen.core.common.constants.constant import INPUTS_KEY, CONFIG_KEY
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.component.branch_comp import BranchComponent
from jiuwen.core.component.branch_router import BranchRouter
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.start_comp import Start
from jiuwen.core.context_engine.base import Context
from jiuwen.core.runtime.base import Input, Output, ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.runtime.workflow import WorkflowRuntime
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import WorkflowConfig
from test_mock_node import MockStartNode, Node1, MockEndNode


class MockSubWorkflowComponent(WorkflowComponent, ComponentExecutable):
    def __init__(self):
        super().__init__()

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        results = []
        for i in range (0,8):
            workflow = self.sub_workflow()
            results.append(await workflow.sub_invoke({"a": "1", "b": 2}, runtime.base(), inputs.get(CONFIG_KEY)))
        output = {"results": results}
        print(output)
        return output


    def graph_invoker(self) -> bool:
        return True

    def component_type(self) -> str:
        return SUB_WORKFLOW_COMPONENT

    def sub_workflow(self) -> Workflow:
        flow = Workflow(workflow_config=WorkflowConfig(stream_timeout=10))
        flow.set_start_comp("start", MockStartNode("start"),
                            inputs_schema={"a": "${a}",
                                           "b": "${b}",
                                           "c": 1,
                                           "d": [1, 2, 3]})

        router = BranchRouter()
        router.add_branch("len(${start.d}) > 2", "a")
        router.add_branch("len(${start.d}) < 2", "b")

        flow.add_conditional_connection("start", router=router)
        flow.add_workflow_comp("a", Node1("a"), inputs_schema={"a": "${start.a}"})
        flow.add_workflow_comp("b", Node1("b"), inputs_schema={"b": "${start.b}"})
        flow.set_end_comp("end", End(), {"result1": "${a.a}", "result2": "${b.b}"})
        flow.add_connection("a", "end")
        flow.add_connection("b", "end")
        return flow


class TestBranchComponent(unittest.TestCase):
    def test_sub_workflow_with_branch(self):
        workflow = Workflow()
        workflow.set_start_comp("s", Start(), inputs_schema={"input": "${data}"})
        workflow.add_workflow_comp("sub_workflow", MockSubWorkflowComponent())
        workflow.set_end_comp("e", End(), inputs_schema={"end_out": "${print_inputs}"})

        workflow.add_connection("s", "sub_workflow")
        workflow.add_connection("sub_workflow", "e")

        inputs = {"data":'aaa'}

        async def run_workflow():
            return await workflow.invoke(inputs, WorkflowRuntime())
        results = asyncio.get_event_loop().run_until_complete(run_workflow())
        print(results)

    def test_add_branch_error(self):
        branch = BranchComponent()
        with self.assertRaises(JiuWenBaseException):
            branch.add_branch(condition=None, target="a", branch_id='')
        with self.assertRaises(JiuWenBaseException):
            branch.add_branch(condition="sss", target='', branch_id='')
        with self.assertRaises(JiuWenBaseException):
            branch.add_branch(condition="sss", target=None, branch_id='')
        with self.assertRaises(JiuWenBaseException):
            branch.add_branch(condition="sss", target=['', "xxx"], branch_id='')
        with self.assertRaises(JiuWenBaseException):
            branch.add_branch(condition="sss", target=["xxx", None], branch_id='')


    def runtime_with_expression(self, expression, value):
        workflow = Workflow()
        workflow.set_start_comp("start", Start(), inputs_schema={"input": "${data}"})
        branch_comp = BranchComponent()
        branch_comp.add_branch(condition=expression, target=["print_inputs"])
        workflow.add_workflow_comp("branch_component", branch_comp)
        workflow.add_workflow_comp("print_inputs", Start(), inputs_schema={"data": "${start}"})
        workflow.set_end_comp("end", End(), inputs_schema={"end_out": "${print_inputs}"})

        workflow.add_connection("start", "branch_component")
        workflow.add_connection("print_inputs", "end")

        inputs = {"data": value}

        async def run_workflow():
            return await workflow.invoke(inputs, WorkflowRuntime())

        print(asyncio.get_event_loop().run_until_complete(run_workflow()))

    def test_expression_is_empty(self):
       self.runtime_with_expression("is_empty(${start.input})", None)
       self.runtime_with_expression("is_empty(${start.input})", [])
       self.runtime_with_expression("is_empty(${start.input})", '')
       self.runtime_with_expression("is_empty(${start.input})", {})
       with self.assertRaises(JiuWenBaseException) as error:
           self.runtime_with_expression("is_empty(${start.input})", 0)
       assert error.exception.error_code == StatusCode.EXPRESSION_CONDITION_EVAL_ERROR.code
       print(error.exception)

       with self.assertRaises(JiuWenBaseException) as error:
           self.runtime_with_expression("is_not_empty(${start.input})", 1.2)
       assert error.exception.error_code == StatusCode.EXPRESSION_CONDITION_EVAL_ERROR.code
       print(error.exception)

       self.runtime_with_expression("is_empty(${start.input}[0])", [None, 'y'])
       self.runtime_with_expression("is_empty(${start.input}['x'])", {'x': None})
       self.runtime_with_expression("is_empty(${start.input}['x'][0])", {'x': [None]})

    def test_expression_is_not_empty(self):
        self.runtime_with_expression("is_not_empty(${start.input})", 'x')
        self.runtime_with_expression("is_not_empty(${start.input})", {'a':'a'})
        self.runtime_with_expression("is_not_empty(${start.input})", ['a'])
        self.runtime_with_expression("is_not_empty(${start.input})", (1,2))
        with self.assertRaises(JiuWenBaseException) as error:
            self.runtime_with_expression("is_not_empty(${start.input})", None)
        print(error.exception)
        assert error.exception.error_code == StatusCode.BRANCH_COMPONENT_BRANCH_NOT_FOUND_ERROR.code

        with self.assertRaises(JiuWenBaseException) as error:
            self.runtime_with_expression("is_not_empty(${start.input})", 1.2)
        assert error.exception.error_code == StatusCode.EXPRESSION_CONDITION_EVAL_ERROR.code
        print(error.exception)

        self.runtime_with_expression("is_not_empty(${start.input}[0])", ['x', 'y'])
        self.runtime_with_expression("is_not_empty(${start.input}['x'])", {'x' : 'x'})
        self.runtime_with_expression("is_not_empty(${start.input}['x'][0])", {'x' : ['x']})

    def test_expression_length(self):
        with self.assertRaises(JiuWenBaseException) as error:
            self.runtime_with_expression("length(${start.input}) == 0", 0)
        assert error.exception.error_code == StatusCode.EXPRESSION_CONDITION_EVAL_ERROR.code
        print(error.exception)
        self.runtime_with_expression("length(${start.input}) == 0", {})
        self.runtime_with_expression("length(${start.input}) == 0", [])
        self.runtime_with_expression("length(${start.input}) == 0", '')
        self.runtime_with_expression("length(${start.input}) == 0", ())