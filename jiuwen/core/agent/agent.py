#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Iterator, Dict, List, Union, AsyncIterator, TYPE_CHECKING

from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.context_engine.config import ContextEngineConfig
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.runtime.agent import StaticAgentRuntime
from jiuwen.core.runtime.resource_manager import ResourceMgr
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.runtime.workflow_manager import generate_workflow_key
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.wrapper import WrappedRuntime, StaticWrappedRuntime, TaskRuntime
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.tool.function.function import LocalFunction
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from jiuwen.core.workflow.base import Workflow

if TYPE_CHECKING:
    from jiuwen.core.agent.controller.controller import Controller


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
        # 所有核心属性统一在基类初始化
        self._config = config
        self._runtime = AgentRuntime(config=config)
        self._context_engine = self._create_context_engine()
        self._message_handler_class = None

    def config(self) -> Config:
        """获取 Agent 配置"""
        return self._config

    @property
    def context_engine(self) -> ContextEngine:
        """获取 Context Engine - 统一公共接口"""
        return self._context_engine

    def set_message_handler(self, message_handler_class):
        """设置 MessageHandler 类 - 子类在构造函数中调用此方法注入具体实现
        
        Args:
            message_handler_class: MessageHandler 的类（不是实例）
        """
        self._message_handler_class = message_handler_class

    def get_message_handler(self):
        """获取当前的 MessageHandler 类"""
        return self._message_handler_class

    @abstractmethod
    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        pass

    @abstractmethod
    async def stream(self, inputs: Dict, runtime: Runtime = None) -> Iterator[Any]:
        pass

    def _create_context_engine(self) -> ContextEngine:
        """创建 ContextEngine - 内部方法，在基类初始化时调用"""
        context_config = ContextEngineConfig(
            conversation_history_length=self._config.get_agent_config().constrain.reserved_max_chat_rounds * 2
        )
        return ContextEngine(
            agent_id=self._config.get_agent_config().id,
            config=context_config,
            model=None
        )

    def _create_controller(self, runtime: Runtime) -> "Controller":
        """创建 Controller 实例"""
        from jiuwen.core.agent.controller.controller import Controller
        
        if self._message_handler_class is None:
            return None

        # 实例化 MessageHandler
        message_handler = self._message_handler_class(
            self._config.get_agent_config(),
            self._context_engine,
            runtime
        )

        return Controller(
            self._config.get_agent_config(),
            self._context_engine,
            runtime,
            message_handler
        )

    @staticmethod
    def unwrap_result(result):
        """解包 OutputSchema 结果 - 子类可直接调用的公共方法"""
        if isinstance(result, list):
            if not result:
                return result
            if isinstance(result[0], OutputSchema):
                if len(result) == 1 and result[0].type == "workflow_final":
                    return result[0].payload
                return result
            return result

        if isinstance(result, OutputSchema):
            payload = result.payload
            if isinstance(payload, dict):
                return payload
            return payload

        return result

    async def controller_invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """基于 Controller 的同步调用 - 子类可直接调用的公共方法"""
        session_id = inputs.pop("conversation_id", "default_session")

        if runtime is None:
            agent_runtime = await self._runtime.pre_run(session_id=session_id)
        else:
            agent_runtime = runtime

        controller = None
        try:
            controller = self._create_controller(agent_runtime)
            await controller.start()
            result = await controller.process_inputs(inputs)

            logger.info("Controller completed, closing stream")

            if runtime is None:
                await agent_runtime.post_run()

            if result is not None:
                return self.unwrap_result(result)

            return {"output": "Message processed by scheduler", "result_type": "answer"}
        except Exception as e:
            if UserConfig.is_sensitive():
                logger.info(f"{self.__class__.__name__} invoke error.")
            else:
                logger.error(f"{self.__class__.__name__} invoke error: {e}")
            raise
        finally:
            if controller:
                await controller.stop()

    async def controller_stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """基于 Controller 的流式调用 - 子类可直接调用的公共方法"""
        session_id = inputs.pop("conversation_id", "default_session")

        if runtime is None:
            agent_runtime = await self._runtime.pre_run(session_id=session_id)
        else:
            agent_runtime = runtime

        async def stream_process():
            controller = None
            try:
                controller = self._create_controller(agent_runtime)
                await controller.start()
                await controller.process_inputs(inputs)
                logger.info("Controller completed, workflow done")
            except Exception as e:
                if UserConfig.is_sensitive():
                    logger.info(f"{self.__class__.__name__} stream error.")
                else:
                    logger.error(f"{self.__class__.__name__} stream error: {e}")
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          f"{self.__class__.__name__} stream error.")
            finally:
                if controller:
                    await controller.stop()
                if runtime is None:
                    await agent_runtime.post_run()
                else:
                    await agent_runtime.end_stream()

        task = asyncio.create_task(stream_process())
        async for result in agent_runtime.stream_iterator():
            yield result

        try:
            await task
        except Exception as e:
            logger.error(f"{self.__class__.__name__} stream error.")
            if UserConfig.is_sensitive():
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          f"{self.__class__.__name__} stream error.")
            else:
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          f"{self.__class__.__name__} stream error.") from e

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
