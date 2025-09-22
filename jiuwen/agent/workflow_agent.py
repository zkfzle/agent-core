from typing import Dict, List, Iterator, Any

from jiuwen.agent.common.enum import ControllerType
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.core.agent.task.task import Task
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.runtime.agent_context import AgentContext
from jiuwen.core.context.controller_context.controller_context_manager import ControllerContextMgr
from jiuwen.core.agent.controller.workflow_controller import WorkflowController, WorkflowControllerOutput
from jiuwen.core.agent.agent import Agent
from jiuwen.core.agent.handler.base import AgentHandlerImpl, AgentHandlerInputs
from jiuwen.core.runtime.runtime import WorkflowRuntime
from jiuwen.core.runtime.state import InMemoryState
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

    def _init_controller(self):
        """初始化Controller - 延迟到invoke/stream时进行"""
        if self._config.controller_type != ControllerType.WorkflowController:
            raise NotImplementedError("")
        return None

    def _init_agent_handler(self):
        return AgentHandlerImpl(self._config)

    def _init_controller_context_manager(self) -> ControllerContextMgr:
        return ControllerContextMgr(self._config)

    def _create_context_engine(self, session_id: str) -> ContextEngine:
        """创建ContextEngine实例"""
        from jiuwen.core.context_engine.config import ContextEngineConfig
        context_config = ContextEngineConfig(
            conversation_history_length=self._config.constrain.reserved_max_chat_rounds * 2
        )
        return ContextEngine(
            agent_id=self._config.id,
            config=context_config,
            model=None  # 可以根据需要传入模型
        )

    def _create_runtime(self, session_id: str) -> WorkflowRuntime:
        """创建Runtime实例"""
        return WorkflowRuntime(
            state=InMemoryState(),
            session_id=session_id
        )

    def _create_controller(self, context_engine: ContextEngine, runtime: WorkflowRuntime) -> WorkflowController:
        """创建WorkflowController实例"""
        controller = WorkflowController(
            self._config,
            self._controller_context_manager,
            context_engine,
            runtime
        )
        controller.set_agent_handler(self._agent_handler)
        return controller

    async def invoke(self, inputs: Dict) -> Dict:
        """同步调用接口"""
        # 1. 初始化ContextEngine和Runtime
        session_id = inputs.get("conversation_id", "default_session")
        context_engine = self._create_context_engine(session_id)
        runtime = self._create_runtime(session_id)

        # 2. 创建Controller
        controller = self._create_controller(context_engine, runtime)

        # 3. 执行ReAct流程
        return await controller.execute(inputs)

    async def stream(self, inputs: Dict) -> Iterator[Any]:
        """流式调用接口"""
        # 1. 初始化ContextEngine和Runtime
        session_id = inputs.get("conversation_id", "default_session")
        context_engine = self._create_context_engine(session_id)
        runtime = self._create_runtime(session_id)

        # 2. 创建Controller
        controller = self._create_controller(context_engine, runtime)

        # 3. 执行流式Workflow agent流程
        async for result in controller.stream_execute(inputs):
            yield result

