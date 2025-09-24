import asyncio
from typing import Dict, List, Any, AsyncIterator

from jiuwen.agent.common.enum import ControllerType
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.core.context_engine.config import ContextEngineConfig
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.agent.controller.workflow_controller import WorkflowController
from jiuwen.core.agent.agent import Agent
from jiuwen.core.agent.handler.base import AgentHandlerImpl
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.stream.writer import TraceSchema
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
    def __init__(self, agent_config: WorkflowAgentConfig):
        super().__init__(agent_config)
        self.context_engine = self._create_context_engine()

    def _init_controller(self):
        """初始化Controller - 延迟到invoke/stream时进行"""
        if self._config.controller_type != ControllerType.WorkflowController:
            raise NotImplementedError("")
        return None

    def _init_agent_handler(self):
        return AgentHandlerImpl(self._config)

    def _create_context_engine(self) -> ContextEngine:
        """创建ContextEngine实例"""
        context_config = ContextEngineConfig(
            conversation_history_length=self._config.constrain.reserved_max_chat_rounds * 2
        )
        return ContextEngine(
            agent_id=self._config.id,
            config=context_config,
            model=None  # 可以根据需要传入模型
        )

    def _create_controller(self, context_engine: ContextEngine, runtime: Runtime) -> WorkflowController:
        """创建WorkflowController实例"""
        controller = WorkflowController(
            self._config,
            context_engine,
            runtime
        )
        controller.set_agent_handler(self._agent_handler)
        return controller

    async def invoke(self, inputs: Dict) -> Dict:
        """同步调用接口"""
        # 1. 初始化ContextEngine和Runtime
        session_id = inputs.get("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)

        # 2. 创建Controller
        controller = self._create_controller(self.context_engine, runtime)

        # 3. 执行WorkflowAgent流程
        result = await controller.execute(inputs)
        await runtime.post_run()
        return result

    async def stream(self, inputs: Dict) -> AsyncIterator[Any]:
        """流式调用接口"""
        # 1. 初始化ContextEngine和Runtime
        session_id = inputs.get("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)

        # 2. 创建Controller
        controller = self._create_controller(self.context_engine, runtime)

        async def stream_process():
            try:
                await controller.execute(inputs)
            finally:
                await runtime.post_run()

        task = asyncio.create_task(stream_process())
        # 3. 执行流式WorkflowAgent流程
        async for result in runtime.stream_iterator():
            if not isinstance(result, TraceSchema):
                yield result

        try:
            await task
        except Exception:
            raise
