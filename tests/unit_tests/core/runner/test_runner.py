import pytest

from jiuwen.agent.common.enum import ControllerType
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.runtime.agent import AgentRuntime
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.wrapper import TaskRuntime
from jiuwen.core.utils.tool.param import Param
from jiuwen.core.utils.tool.tool import tool
from jiuwen.core.workflow.base import Workflow, WorkflowOutput, WorkflowExecutionState
from jiuwen.core.workflow.workflow_config import WorkflowConfig, WorkflowMetadata
from jiuwen.core.runner.runner import Runner
from tests.unit_tests.core.workflow.mock_nodes import MockEndNode, Node1, MockStartNode


@pytest.fixture(scope="class")
def runtime():
    workflow_id = "test_workflow"
    name = "test_workflow"
    version = "1"
    description = "test_workflow"
    test_workflow_schema = WorkflowSchema(
        id=workflow_id,
        version=version,
        name=name,
        description=description,
        inputs={"query": {
            "type": "string",
        }},
    )
    workflow_config = WorkflowAgentConfig(
        workflows=[test_workflow_schema],
        tools=["add"],
        controller_type=ControllerType.WorkflowController
    )
    config = Config()
    config.set_agent_config(agent_config=workflow_config)
    session_id = "session_id"
    return TaskRuntime(None, AgentRuntime(session_id, config=config))

@pytest.mark.asyncio
class TestRunner:
    @staticmethod
    def _build_workflow(name, workflow_id, version):
        workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                id=workflow_id,
                version=version,
                name=name,
            )
        )
        flow = Workflow(workflow_config=workflow_config)
        flow.set_start_comp("start", MockStartNode("start"),
                            inputs_schema={
                                "query": "${query}"})
        flow.add_workflow_comp("node_a", Node1("node_a"),
                               inputs_schema={
                                   "output": "${start.query}"})
        flow.set_end_comp("end", MockEndNode("end"),
                          inputs_schema={
                              "result": "${node_a.output}"})
        flow.add_connection("start", "node_a")
        flow.add_connection("node_a", "end")
        return flow

    @staticmethod
    @tool(
        name="add",
        description="加法",
        params=[
            Param(name="a", description="加数", type="number", required=True),
            Param(name="b", description="被加数", type="number", required=True),
        ],
    )
    def add_function(a, b):
        """加法函数，使用tool注解装饰"""
        return a + b

    @staticmethod
    @tool(
        name="multiply",
        description="乘法",
        params=[
            Param(name="a", description="乘数", type="number", required=True),
            Param(name="b", description="被乘数", type="number", required=True),
        ],
    )
    def multiply_function(a, b):
        """乘法函数，使用tool注解装饰"""
        return a * b

    async def test_run_workflow(self, runtime):
        workflow_id = "test_workflow"
        name = "test_workflow"
        version = "1"
        workflow = self._build_workflow(name, workflow_id, version)
        result = await Runner.run_workflow(workflow, inputs = {"query": "query workflow"}, runtime=runtime)
        assert result == WorkflowOutput(result={"result": "query workflow"}, state=WorkflowExecutionState.COMPLETED)

    async def test_run_tool(self, runtime):
        result = await Runner.run_tool(tool=self.add_function, inputs={"a": 1, "b": 2}, runtime=runtime)
        assert result == 3

    async def test_run_workflow_not_bound(self, runtime):
        workflow_id = "test_workflow_not_bound"
        name = "test_workflow"
        version = "1"
        workflow = self._build_workflow(name, workflow_id, version)
        with pytest.raises(JiuWenBaseException) as exc_info:
            await Runner.run_workflow(workflow, inputs={"query": "query workflow"}, runtime=runtime)
        assert exc_info.value.error_code == StatusCode.WORKFLOW_NOT_BOUND_TO_AGENT.code
        assert exc_info.value.message == StatusCode.WORKFLOW_NOT_BOUND_TO_AGENT.errmsg

    async def test_run_tool_not_bound(self, runtime):
        with pytest.raises(JiuWenBaseException) as exc_info:
            await Runner.run_tool(tool=self.multiply_function, inputs={"a": 1, "b": 2}, runtime=runtime)
        assert exc_info.value.error_code == StatusCode.TOOL_NOT_BOUND_TO_AGENT.code
        assert exc_info.value.message == StatusCode.TOOL_NOT_BOUND_TO_AGENT.errmsg

