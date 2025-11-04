from typing import Dict, List, Any, AsyncIterator

from jiuwen.agent.common.enum import ControllerType
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.agent.workflow_agent.workflow_message_handler import WorkflowMessageHandler
from jiuwen.core.agent.agent import Agent
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.runtime import Runtime, Workflow


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
    """工作流模式的Agent - 执行预定义的工作流"""
    
    def __init__(self, agent_config: WorkflowAgentConfig):
        # 验证 controller_type
        if agent_config.controller_type != ControllerType.WorkflowController:
            raise NotImplementedError(f"WorkflowAgent requires WorkflowController, got {agent_config.controller_type}")

        # 创建配置并初始化基类
        config = Config()
        config.set_agent_config(agent_config=agent_config)
        super().__init__(config)
        
        # 设置消息处理器
        self.set_message_handler(WorkflowMessageHandler)

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """同步调用 - 使用基类的通用实现"""
        return await self.controller_invoke(inputs, runtime)

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """流式调用 - 使用基类的通用实现"""
        async for result in self.controller_stream(inputs, runtime):
            yield result
