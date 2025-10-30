#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from abc import ABC, abstractmethod
from typing import Any, Iterator, Dict, List, Union

from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.core.runtime.agent import StaticAgentRuntime
from jiuwen.core.runtime.resource_manager import ResourceMgr
from jiuwen.core.runtime.runtime import Runtime

from jiuwen.core.runtime.workflow_manager import generate_workflow_key
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.wrapper import WrappedRuntime, StaticWrappedRuntime, TaskRuntime
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.tool.function.function import LocalFunction
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from jiuwen.core.workflow.base import Workflow


class AgentRuntime(WrappedRuntime, StaticWrappedRuntime):
    def __init__(self, config: Config = None, resource_mgr: ResourceMgr = None):
        inner = StaticAgentRuntime(config, resource_mgr=resource_mgr)
        super().__init__(inner)
        self._runtime = inner

    async def write_stream(self, data: Union[dict, OutputSchema]):
        return await self.write_custom_stream(data)

    async def pre_run(self, **kwargs) -> Runtime:
        session_id = kwargs.get("session_id")
        if session_id is None:
            session_id = kwargs.get("trace_id")
        inputs = kwargs.get("inputs")
        inner = await self._runtime.create_agent_runtime(session_id, inputs)
        return TaskRuntime(inner=inner)

    async def release(self, session_id: str):
        await self._runtime.checkpointer().release(session_id)


class Agent(ABC):
    """
    The top-level abstract class and the common base class for all Agents.
    Subclasses must implement:
        - invoke : synchronous one-time call
        - stream : streaming call
    """

    def __init__(self, config: Config) -> None:
        self._runtime = AgentRuntime(config=config)
        self._config = config
        self._controller: "Controller | None" = self._init_controller()
        self._agent_handler: "AgentHandler | None" = self._init_agent_handler()
        self._message_handler = None  # 支持自定义MessageHandler

    def _init_controller(self) -> "Controller | None":
        return None

    def _init_agent_handler(self) -> "AgentHandler | None":
        return None

    def config(self):
        return self._config

    def set_message_handler(self, message_handler):
        """
        设置自定义MessageHandler

        Args:
            message_handler: 自定义的消息处理器实例，必须继承自MessageHandler基类

        Example:
            # 创建自定义MessageHandler
            custom_handler = CustomMessageHandler(config, context_engine, runtime)
            agent.set_message_handler(custom_handler)
        """
        self._message_handler = message_handler

    def get_message_handler(self):
        """获取当前使用的MessageHandler"""
        return self._message_handler

    @abstractmethod
    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        pass

    @abstractmethod
    async def stream(self, inputs: Dict, runtime: Runtime = None) -> Iterator[Any]:
        pass

    def bind_workflows(self, workflows: List[Workflow]):
        self._runtime.add_workflows(
            [(generate_workflow_key(workflow.config().metadata.id, workflow.config().metadata.version), workflow) for
             workflow in
             workflows])
        for workflow in workflows:
            self._config.get_agent_config().workflows.append(WorkflowSchema(id=workflow.config().metadata.id,
                                                         name=workflow.config().metadata.name,
                                                         version=workflow.config().metadata.version,
                                                         description=workflow.config().metadata.description))

    def bind_tools(self, tools: List[Tool]):
        self._runtime.add_tools(
            [(tool.name, tool) for tool in tools if (isinstance(tool, RestfulApi) or isinstance(tool, LocalFunction))])
        for tool in tools:
            self._config.get_agent_config().tools.append(tool.name)

    def get_llm_calls(self) -> Dict:
        raise NotImplementedError("")

    def copy(self) -> "Agent":
        raise NotImplementedError("")
