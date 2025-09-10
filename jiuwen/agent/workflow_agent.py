from typing import Dict, List

from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.core.agent.task.task import Task
from jiuwen.core.runtime.agent_context import AgentContext
from jiuwen.core.context.controller_context.controller_context_manager import ControllerContextMgr
from jiuwen.core.agent.controller.workflow_controller import WorkflowController, WorkflowControllerOutput
from jiuwen.core.agent.agent import Agent
from jiuwen.core.agent.handler.base import AgentHandlerImpl, AgentHandlerInputs
from jiuwen.core.workflow.base import Workflow


def create_workflow_agent_config(agent_id: str,
                                 agent_version: str,
                                 description: str,
                                 workflows: List[WorkflowSchema]):
    config = WorkflowAgentConfig(id=agent_id,
                                 version=agent_version,
                                 description=description,
                                 workflows=workflows)
    return config


def create_workflow_agent(agent_config: WorkflowAgentConfig,
                          workflows: List[Workflow] = None):
    agent = WorkflowAgent(agent_config)
    agent.bind_workflows(workflows)
    return agent


class WorkflowAgent(Agent):
    def __init__(self, agent_config: WorkflowAgentConfig, agent_context: AgentContext = None):
        super().__init__(agent_config, agent_context)
        self._config = agent_config

    def _init_controller(self):
        return WorkflowController(self._config, self._controller_context_manager)

    def _init_agent_handler(self):
        return AgentHandlerImpl(self._config)

    def _init_controller_context_manager(self) -> ControllerContextMgr:
        return ControllerContextMgr(self._config)

    async def invoke(self, inputs: Dict) -> Dict:
        task: Task = self._task_manager.create_task(inputs.get("conversation_id"))
        context = task.context
        context.set_controller_context_manager(self._controller_context_manager)
        current_inputs = inputs

        while True:
            controller_output: WorkflowControllerOutput = self._controller.invoke(current_inputs, context)
            results = {}
            if controller_output.sub_tasks:
                for sub_task in controller_output.sub_tasks:
                    inputs = AgentHandlerInputs(context=context, name=sub_task.func_name, arguments=sub_task.func_args)
                    result = await self._agent_handler.invoke(sub_task.sub_task_type, inputs)
                    results[sub_task.func_name] = result
            if not self._controller.should_continue(controller_output):
                output = self.handle_workflow_results(results)
                return output
            current_inputs = results

    async def stream(self, inputs: Dict):
        pass

    def handle_workflow_results(self, results):
        if self._config.is_single_workflow:
            return results[self._config.workflows[0].name]
        raise Exception("Multi-workflow not implemented yet")
