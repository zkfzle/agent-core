import pytest

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.component.end_comp import End
from openjiuwen.core.component.start_comp import Start
from openjiuwen.core.component.workflow_comp import SubWorkflowComponent
from openjiuwen.core.runtime.workflow import WorkflowRuntime
from openjiuwen.core.workflow.base import Workflow
from openjiuwen.core.workflow.workflow_config import WorkflowConfig

pytestmark = pytest.mark.asyncio

class TestSubWorkflowComp:
    async def test_add_component(self):
        main_workflow = Workflow(WorkflowConfig(workflow_max_nesting_depth=2))
        main_workflow.set_start_comp("start", Start())
        main_workflow.add_workflow_comp("fick_comp", SubWorkflowComponent(main_workflow))
        main_workflow.set_end_comp("end", End())
        main_workflow.add_connection("start", 'fick_comp')
        main_workflow.add_connection('fick_comp', "end")
        with pytest.raises(JiuWenBaseException):
            await main_workflow.invoke(inputs={}, runtime=WorkflowRuntime())

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

    async def test_sub_invoke(self):
        with pytest.raises(JiuWenBaseException) as err:
            workflow_config = WorkflowConfig(workflow_max_nesting_depth=1)
            main_workflow = self.create_nesting_workflow(3, workflow_config)
            await main_workflow.invoke(inputs={}, runtime=WorkflowRuntime())
        assert err.value.message == StatusCode.COMPONENT_EXECUTE_ERROR.errmsg.format(node_id="sub2",
             ability="invoke",
             error=StatusCode.SUB_WORKFLOW_COMPONENT_RUNNING_ERROR.errmsg.format(
                 detail='workflow nesting hierarchy is too big, must <= 1'))

        workflow_config = WorkflowConfig(workflow_max_nesting_depth=3)
        main_workflow = self.create_nesting_workflow(3, workflow_config)

        await main_workflow.invoke(inputs={}, runtime=WorkflowRuntime())

        workflow_config = WorkflowConfig(workflow_max_nesting_depth=0)
        main_workflow = self.create_nesting_workflow(0, workflow_config)

        await main_workflow.invoke(inputs={}, runtime=WorkflowRuntime())


