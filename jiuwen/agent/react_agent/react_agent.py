#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import asyncio
from typing import Dict, Any, AsyncIterator, List

from jiuwen.agent.common.enum import ControllerType
from jiuwen.agent.common.schema import WorkflowSchema, PluginSchema
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.agent.config.react_config import ReActAgentConfig
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.context_engine.config import ContextEngineConfig
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.agent.controller.controller import Controller
from jiuwen.agent.react_agent.react_message_handler import ReActMessageHandler
from jiuwen.core.agent.agent import Agent
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.runtime import Runtime, Workflow
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.utils.tool.base import Tool

def create_react_agent_config(agent_id: str,
                              agent_version: str,
                              description: str,
                              workflows: List[WorkflowSchema],
                              plugins: List[PluginSchema],
                              model: ModelConfig,
                              prompt_template: List[Dict]):
    config = ReActAgentConfig(id=agent_id,
                              version=agent_version,
                              description=description,
                              workflows=workflows,
                              plugins=plugins,
                              model=model,
                              prompt_template=prompt_template)
    return config


def create_react_agent(agent_config: ReActAgentConfig,
                       workflows: List[Workflow] = None,
                       tools: List[Tool] = None):
    agent = ReActAgent(agent_config)
    agent.bind_workflows(workflows)
    agent.bind_tools(tools or [])
    return agent

class ReActAgent(Agent):
    def __init__(self, agent_config: ReActAgentConfig):
        self._config = Config()
        self._config.set_agent_config(agent_config=agent_config)
        super().__init__(self._config)
        self.context_engine = self._create_context_engine()

    def _init_controller(self):
        if self._config.get_agent_config().controller_type != ControllerType.ReActController:
            raise NotImplementedError("")
        return None

    def _create_controller(self, runtime: Runtime) -> Controller:
        """每次调用都创建新的 controller"""
        message_handler = ReActMessageHandler(
            self._config.get_agent_config(),
            self.context_engine,
            runtime
        )

        controller = Controller(
            self._config.get_agent_config(),
            self.context_engine,
            runtime,
            message_handler
        )
        return controller

    def _create_context_engine(self) -> ContextEngine:
        context_config = ContextEngineConfig(
            conversation_history_length=self._config.get_agent_config().constrain.reserved_max_chat_rounds * 2
        )
        return ContextEngine(
            agent_id=self._config.get_agent_config().id,
            config=context_config,
            model=None
        )

    @staticmethod
    def _unwrap_result(result):
        """解包 OutputSchema 结果"""
        # 如果是列表
        if isinstance(result, list):
            # 空列表直接返回
            if not result:
                return result

            # 如果列表中是 OutputSchema，需要解包
            if isinstance(result[0], OutputSchema):
                # 对于完成状态，列表只有一个元素，解包它
                if len(result) == 1 and result[0].type == "workflow_final":
                    return result[0].payload
                # 对于中断状态，返回整个列表（交互请求）
                return result

            # 否则直接返回
            return result

        # 如果是 OutputSchema，解包其 payload
        if isinstance(result, OutputSchema):
            payload = result.payload
            # 如果 payload 是字典，直接返回
            if isinstance(payload, dict):
                return payload
            return payload

        # 否则直接返回
        return result

    async def invoke(self, inputs: Dict) -> Dict:
        """直接调用"""
        session_id = inputs.pop("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)
        controller = None
        try:
            # 每次调用都创建新的 controller
            controller = self._create_controller(runtime)
            await controller.start()

            # 处理输入并等待完成（统一接口，不再访问私有属性）
            result = await controller.process_inputs(inputs)
            
            logger.info("Controller completed, closing stream")
            
            # 调度器完成后，关闭流（发送 END_FRAME）
            await runtime.post_run()

            # 解包结果
            # result 来自 MessageHandler 的 final_result，是 OutputSchema 或 OutputSchema 列表
            if result is not None:
                return ReActAgent._unwrap_result(result)
            
            # 如果没有结果，返回默认值
            return {"output": "Message processed by scheduler", "result_type": "answer"}
        except Exception as e:
            if UserConfig.is_sensitive():
                logger.info(f"ReActAgent invoke error.")
            else:
                logger.error(f"ReActAgent invoke error: {e}")
            raise
        finally:
            if controller:
                await controller.stop()

    async def stream(self, inputs: Dict) -> AsyncIterator[Any]:
        """流式调用"""
        session_id = inputs.pop("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)

        async def stream_process():
            controller = None
            try:
                # 每次调用都创建新的 controller
                controller = self._create_controller(runtime)
                await controller.start()

                # 处理输入并等待完成（统一接口，不再访问私有属性）
                await controller.process_inputs(inputs)

                logger.info("Controller completed, workflow done")

            except Exception as e:
                if UserConfig.is_sensitive():
                    logger.info(f"ReActAgent stream error.")
                else:
                    logger.error(f"ReActAgent stream error: {e}")
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          "ReActAgent stream error.")
            finally:
                if controller:
                    await controller.stop()
                # 调度器完成后，关闭流
                await runtime.post_run()

        # 启动后台任务处理消息和任务
        task = asyncio.create_task(stream_process())

        # 前台负责 yield 所有流数据
        # stream_iterator() 会一直迭代直到收到 END_FRAME（由 runtime.post_run() 发送）
        async for result in runtime.stream_iterator():
            yield result

        # 等待后台任务完成
        try:
            await task
        except Exception as e:
            logger.error(f"ReActAgent stream error.")
            if UserConfig.is_sensitive():
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          "ReActAgent stream error.")
            else:
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          "ReActAgent stream error.") from e
