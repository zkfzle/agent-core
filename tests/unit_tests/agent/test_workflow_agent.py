import pytest

from jiuwen.agent.common.enum import ControllerType
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.agent.workflow_agent import WorkflowAgent
from jiuwen.core.agent.controller.workflow_controller import WorkflowController
from jiuwen.core.runtime.agent_context import AgentContext
from jiuwen.core.runtime.config import WorkflowConfig
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import WorkflowMetadata
from jiuwen.graph.pregel.graph import PregelGraph
from tests.unit_tests.workflow.test_mock_node import MockStartNode, Node1, MockEndNode


class TestWorkflowAgent:
    @staticmethod
    def _build_workflow(name, id, version):
        workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                id=id,
                version=version,
                name=name,
            )
        )
        flow = Workflow(workflow_config=workflow_config, graph=PregelGraph())
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

    # 真正实例化
    @pytest.fixture(scope="class")
    def agent(self):
        agent_context = AgentContext()
        id = "test_workflow"
        name = "test_workflow"
        version = "1"
        description = "test_workflow"
        workflow1 = self._build_workflow(name, id, version)
        test_workflow_schema = WorkflowSchema(
            id=id,
            version=version,
            name=name,
            description=description,
            inputs={"query": {
                "type": "string",
            }},
        )
        workflow_config = WorkflowAgentConfig(
            workflows=[test_workflow_schema],
            controller_type =ControllerType.WorkflowController
        )
        agent = WorkflowAgent(workflow_config, agent_context)
        agent.bind_workflows([workflow1])
        return agent

    # ---------- 测试用例 ----------
    @pytest.mark.asyncio
    async def test_invoke_single(self, agent):
        inputs = {"query": "hi"}
        result = await agent.invoke(inputs)  # ✅ 使用 await
        assert result == {'output': {'result': 'hi'}, 'result_type': 'answer'}
