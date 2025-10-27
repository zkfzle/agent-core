import asyncio
from typing import Dict, Any, AsyncIterator, List

from jiuwen.agent.common.enum import ControllerType
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.core.agent.message.message import Message
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.context_engine.config import ContextEngineConfig
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.agent.controller.controller import Controller
from jiuwen.agent.workflow_agent.workflow_message_handler import WorkflowMessageHandler
from jiuwen.core.agent.agent import Agent
from jiuwen.core.agent.handler.base import AgentHandlerImpl
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.runtime import Runtime, Workflow
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.utils.config.user_config import UserConfig


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
        self._config = Config()
        self._config.set_agent_config(agent_config=agent_config)
        super().__init__(self._config)
        self.context_engine = self._create_context_engine()

    def _init_controller(self):
        if self._config.get_agent_config().controller_type != ControllerType.WorkflowController:
            raise NotImplementedError("")
        return None

    def _create_controller(self, runtime: Runtime) -> Controller:
        """每次调用都创建新的 controller"""
        # 优先使用自定义MessageHandler，否则使用默认的WorkflowMessageHandler
        if self._custom_message_handler is not None:
            message_handler = self._custom_message_handler
        else:
            message_handler = WorkflowMessageHandler(
                self._config.get_agent_config(),
                self.context_engine,
                runtime
            )

        # 创建 controller
        controller = Controller(
            self._config.get_agent_config(),
            self.context_engine,
            runtime,
            message_handler
        )
        controller.set_agent_handler(self._agent_handler)
        return controller

    def _init_agent_handler(self):
        return AgentHandlerImpl(self._config.get_agent_config())

    def _create_context_engine(self) -> ContextEngine:
        context_config = ContextEngineConfig(
            conversation_history_length=self._config.get_agent_config().constrain.reserved_max_chat_rounds * 2
        )
        return ContextEngine(
            agent_id=self._config.get_agent_config().id,
            config=context_config,
            model=None
        )

    async def _send_message_to_controller(self, inputs: Dict, controller: Controller) -> None:
        """发送消息到控制器的公共方法"""
        session_id = inputs.get("conversation_id", "default_session")
        message = Message.create_user_message(
            content=inputs.get("query", ""),
            conversation_id=session_id
        )
        await controller.receive_message(message)

    @staticmethod
    async def _wait_for_sync_result(runtime) -> Dict:
        """等待同步结果"""
        result = None
        timeout_count = 0
        max_timeout = 30  # 最多等待30秒
        try:
            async for stream_result in runtime.stream_iterator():
                timeout_count += 1

                # 超时检查
                if timeout_count > max_timeout:
                    logger.warning("Stream timeout, returning default result")
                    break

                # 检查是否是最终结果
                if isinstance(stream_result, OutputSchema):
                    # 检查是否是错误结果
                    if stream_result.type == "workflow_final":
                        payload = stream_result.payload
                        # 检查是否包含错误
                        if isinstance(payload, dict) and payload.get("error"):
                            logger.error(f"Receive error result from workflow: {payload.get('message')}")
                            result = payload
                        else:
                            logger.info(f"Receive batch result from workflow")
                            result = payload
                        break
                    else:
                        # 其他类型的 OutputSchema，继续等待最终结果
                        continue
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            result = {"output": f"Stream processing error: {e}", "result_type": "error", "error": True}

        # 如果没有获取到结果，返回默认值
        if result is None:
            result = {"output": "Message processed by scheduler", "result_type": "answer"}

        return result

    async def invoke(self, inputs: Dict) -> Dict:
        """同步调用工作流"""
        session_id = inputs.pop("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)
        logger.info(f"runtime id is {runtime}")
        try:
            # 每次调用都创建新的 controller
            controller = self._create_controller(runtime)
            await controller.start()

            try:
                # 发送消息
                await self._send_message_to_controller(inputs, controller)

                # 等待同步结果
                result = await WorkflowAgent._wait_for_sync_result(runtime)
                return result
            finally:
                await controller.stop()
        except Exception as e:
            if UserConfig.is_sensitive():
                logger.info(f"WorkflowAgent invoke error.")
            else:
                logger.error(f"WorkflowAgent invoke error: {e}")
        finally:
            await runtime.post_run()

    async def stream(self, inputs: Dict) -> AsyncIterator[Any]:
        """流式调用工作流"""
        session_id = inputs.pop("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)

        async def stream_process():
            try:
                # 每次调用都创建新的 controller
                controller = self._create_controller(runtime)
                await controller.start()

                try:
                    # 发送消息
                    await self._send_message_to_controller(inputs, controller)
                finally:
                    await controller.stop()
            except Exception as e:
                if UserConfig.is_sensitive():
                    logger.info(f"WorkflowAgent stream error.")
                else:
                    logger.error(f"WorkflowAgent stream error: {e}")
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          "WorkflowAgent stream error.")
            finally:
                await runtime.post_run()

        task = asyncio.create_task(stream_process())

        async for result in runtime.stream_iterator():
            yield result

        try:
            await task
        except Exception as e:
            logger.error(f"WorkflowAgent stream error.")
            if UserConfig.is_sensitive():
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          "WorkflowAgent stream error.")
            else:
                raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                          "WorkflowAgent stream error.") from e
