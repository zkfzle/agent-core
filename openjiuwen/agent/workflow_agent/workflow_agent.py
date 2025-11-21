from typing import Dict, List, Any, AsyncIterator

from openjiuwen.agent.common.enum import ControllerType
from openjiuwen.agent.common.schema import WorkflowSchema
from openjiuwen.agent.config.workflow_config import WorkflowAgentConfig
from openjiuwen.core.agent.agent import ControllerAgent
from openjiuwen.agent.workflow_agent.workflow_controller import WorkflowController
from openjiuwen.core.runtime.runtime import Runtime, Workflow


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
    if workflows:
        agent.bind_workflows(workflows)
    return agent


class WorkflowAgent(ControllerAgent):
    """Workflow-based Agent - Executes predefined workflows with multi-workflow controller
    
    Implemented using ControllerAgent
    """

    def __init__(self, agent_config: WorkflowAgentConfig):
        # Validate controller_type
        if agent_config.controller_type != ControllerType.WorkflowController:
            raise NotImplementedError(
                f"WorkflowAgent requires WorkflowController, "
                f"got {agent_config.controller_type}"
            )

        # Create controller without parameters - will be auto-configured by ControllerAgent
        controller = WorkflowController()
        
        # Pass to parent - parent will auto-configure it
        super().__init__(agent_config, controller=controller)

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """Synchronous invocation - Delegate to controller
        
        Args:
            inputs: Input data, including query and conversation_id
            runtime: Runtime context (optional)
            
        Returns:
            Execution result
        """
        # Fully delegate to ControllerAgent implementation
        return await super().invoke(inputs, runtime)

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """Streaming invocation - Delegate to controller
        
        Args:
            inputs: Input data, including query and conversation_id
            runtime: Runtime context (optional)
            
        Yields:
            Streaming output
        """
        # Fully delegate to ControllerAgent implementation
        async for result in super().stream(inputs, runtime):
            yield result
