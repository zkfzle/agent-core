#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.message.message import Message, MessageType
from jiuwen.core.agent.controller.reasoner.agent_reasoner import AgentReasoner
from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.hash_util import generate_key
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.stream.base import OutputSchema
from jiuwen.agent.common.schema import WorkflowSchema


@dataclass
class MessageHandlerResult:
    """消息处理结果 - 统一的返回结构
    
    Attributes:
        tasks: 生成的任务列表
        should_continue: 是否应该继续调度循环
        final_result: 最终结果（当 should_continue=False 时使用）
    """
    tasks: List[Task]
    should_continue: bool = True
    final_result: Optional[Any] = None


class MessageHandler(ABC):
    """MessageHandler - 消息处理器基类，包含抽象方法，支持自定义实现"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        self.config = config
        self.context_engine = context_engine
        self.runtime = runtime
        self.reasoner: Optional[AgentReasoner] = None

    def set_reasoner(self, reasoner: AgentReasoner):
        """设置决策器引用"""
        self.reasoner = reasoner

    async def process_message(self, message: Message) -> MessageHandlerResult:
        """处理消息，返回处理结果 - 统一的消息处理入口"""
        try:
            # 1. 消息预处理
            processed_message = await self.preprocess_message(message)

            # 2. 自定义消息处理逻辑（子类实现）
            result = await self.handle_message(processed_message)

            # 3. 任务后处理
            processed_tasks = []
            for task in result.tasks:
                processed_task = await self.postprocess_task(task)
                processed_tasks.append(processed_task)

            return MessageHandlerResult(
                tasks=processed_tasks,
                should_continue=result.should_continue,
                final_result=result.final_result
            )

        except Exception as e:
            logger.error(f"Error processing message {message.msg_id}: {e}")
            # 出错时停止调度
            return MessageHandlerResult(tasks=[], should_continue=False, final_result=None)

    @abstractmethod
    async def handle_message(self, message: Message) -> MessageHandlerResult:
        """消息处理逻辑 - 子类必须实现此方法，返回 MessageHandlerResult"""
        pass

    async def preprocess_message(self, message: Message) -> Message:
        """消息预处理 - 子类可重写"""
        return message

    async def postprocess_task(self, task: Task) -> Task:
        """任务后处理 - 子类可重写"""
        return task

    async def _send_final_stream(self, msg):
        """发送最终结果流并返回 OutputSchema"""
        try:
            payload = {"output": msg, "result_type": "answer"}
            final_stream = OutputSchema(
                type="answer",
                index=0,
                payload=payload
            )
            await self.runtime.write_stream(final_stream)
            return final_stream
        except Exception as e:
            logger.error(f"Failed to send final stream: {e}")
            return OutputSchema(
                type="workflow_final",
                index=0,
                payload={
                    "error": True,
                    "message": str(e),
                    "status": "failed"
                }
            )

    async def _send_error_stream(self, error_message: str):
        """发送错误结果流并返回 OutputSchema"""
        try:
            error_stream = OutputSchema(
                type="workflow_final",
                index=0,
                payload={
                    "error": True,
                    "message": error_message,
                    "status": "failed"
                }
            )
            await self.runtime.write_stream(error_stream)
            return error_stream
        except Exception as e:
            logger.error(f"Failed to send error stream: {e}")
            return OutputSchema(
                type="workflow_final",
                index=0,
                payload={
                    "error": True,
                    "message": str(e),
                    "status": "failed"
                }
            )

    async def _write_message_stream_data(self, message: Message):
        """写入消息携带的流数据 List[OutputSchema]"""
        stream_data = message.content.stream_data
        try:
            for output_schema in stream_data:
                await self.runtime.write_stream(output_schema)
                logger.debug(f"Wrote stream data from message {message.msg_id}")
        except Exception as e:
            logger.warning(f"Failed to write message stream data: {e}")

    async def _handle_error(self, message: Message) -> "MessageHandlerResult":
        """处理错误消息：发送错误流并返回停止信号"""
        error_msg = message.content.query if message.content.query else "Unknown error"
        logger.error(f"Received error message: {error_msg}")
        error_result = await self._send_error_stream(str(error_msg))
        return MessageHandlerResult(
            tasks=[],
            should_continue=False,
            final_result=error_result
        )

    def _get_model(self):
        """根据模型配置获取或创建（并缓存到 runtime）模型实例"""
        model_id = generate_key(
            self.config.model.model_info.api_key,
            self.config.model.model_info.api_base,
            self.config.model.model_provider
        )

        model = self.runtime.get_model(model_id=model_id)
        if model is None:
            model = ModelFactory().get_model(
                model_provider=self.config.model.model_provider,
                api_base=self.config.model.model_info.api_base,
                api_key=self.config.model.model_info.api_key
            )
            self.runtime.add_model(model_id=model_id, model=model)

        return model

    @staticmethod
    def _extract_component_id_from_stream_data(stream_data: List) -> Optional[str]:
        """
        从 stream_data 中提取交互组件ID。

        Args:
            stream_data: 流数据列表 (List[OutputSchema])

        Returns:
            Optional[str]: 交互组件的ID，如果找不到则返回None
        """
        if not stream_data:
            return None

        try:
            # 遍历 stream_data，找到类型为 '__interaction__' 的输出
            for output_schema in stream_data:
                if hasattr(output_schema, 'type') and output_schema.type == '__interaction__':
                    # 从 payload 中提取 InteractionOutput.id
                    if hasattr(output_schema, 'payload') and hasattr(output_schema.payload, 'id'):
                        component_id = output_schema.payload.id
                        logger.debug(f"Extracted component_id from stream_data: {component_id}")
                        return component_id
        except Exception as e:
            logger.warning(f"Failed to extract component_id from stream_data: {e}")

        return None

    @staticmethod
    def _match_workflow_generic(detected_task: Task, workflows: List[WorkflowSchema]) -> Optional[WorkflowSchema]:
        """通用的 workflow 匹配：根据 task.input 的 id/name 在给定 workflows 中找匹配项"""
        if not detected_task or not detected_task.input:
            return None
        target_id = detected_task.input.target_id
        target_name = detected_task.input.target_name
        for workflow in workflows:
            workflow_full_id = f"{workflow.id}_{workflow.version}" if workflow.version else workflow.id
            if target_id and target_id in {workflow.id, workflow_full_id}:
                return workflow
            if target_name and workflow.name == target_name:
                return workflow
        return None

    @classmethod
    def _resolve_workflow_from_tasks_generic(cls, tasks: List[Task], workflows: List[WorkflowSchema]) -> Optional[WorkflowSchema]:
        """遍历 tasks，使用通用匹配逻辑返回第一个匹配的 workflow"""
        for task in tasks or []:
            workflow = cls._match_workflow_generic(task, workflows)
            if workflow:
                return workflow
        return None
