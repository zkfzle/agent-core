import asyncio
import unittest

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.start_comp import Start
from jiuwen.core.component.workflow_comp import SubWorkflowComponent
from jiuwen.core.runtime.workflow import WorkflowRuntime
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import WorkflowConfig


class TestSubWorkflowComp(unittest.TestCase):
    def test_add_component(self):
        main_workflow = Workflow()
        with self.assertRaises(JiuWenBaseException) as error:
            main_workflow.add_workflow_comp("fick_comp", SubWorkflowComponent(main_workflow))
            assert error.error_code == StatusCode.SUB_WORKFLOW_COMPONENT_RUNNING_ERROR.code
            assert ('sub_workflow can not be main workflow' in error.message) == True
        sub_workflow = Workflow()
        main_workflow.add_workflow_comp("fick_comp", SubWorkflowComponent(sub_workflow))

    def create_nesting_workflow(self, sub_workflow_depth=0, workflow_config=None):
        workflow = Workflow(workflow_config)
        workflow.set_start_comp("start", Start())
        if sub_workflow_depth > 0:
            workflow.add_workflow_comp(f'sub{sub_workflow_depth}',
                                       SubWorkflowComponent(self.create_nesting_workflow(sub_workflow_depth - 1)))
        workflow.set_end_comp("end", End())
        if sub_workflow_depth > 0:
            workflow.add_connection("start", f'sub{sub_workflow_depth}')
            workflow.add_connection(f'sub{sub_workflow_depth}', "end")
        else:
            workflow.add_connection("start", "end")
        return workflow

    def test_sub_invoke(self):
        with self.assertRaises(JiuWenBaseException) as err:
            workflow_config = WorkflowConfig(workflow_max_nesting_depth=1)
            main_workflow = self.create_nesting_workflow(3, workflow_config)

            async def run():
                return await main_workflow.invoke(inputs={}, runtime=WorkflowRuntime())

            print(asyncio.get_event_loop().run_until_complete(run()))
            assert err.msg == "failed to invoke, caused by failed to invoke, caused by Sub workflow component running error, detail: workflow nesting hierarchy is too big, must <= 1"

        workflow_config = WorkflowConfig(workflow_max_nesting_depth=3)
        main_workflow = self.create_nesting_workflow(3, workflow_config)

        async def run():
            return await main_workflow.invoke(inputs={}, runtime=WorkflowRuntime())

        print(asyncio.get_event_loop().run_until_complete(run()))

        workflow_config = WorkflowConfig(workflow_max_nesting_depth=0)
        main_workflow = self.create_nesting_workflow(0, workflow_config)

        async def run():
            return await main_workflow.invoke(inputs={}, runtime=WorkflowRuntime())

        print(asyncio.get_event_loop().run_until_complete(run()))


