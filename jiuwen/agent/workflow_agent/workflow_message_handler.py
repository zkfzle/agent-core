#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List, Dict, Any, Union
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.message.message import Message
from jiuwen.core.agent.task.task import Task, TaskInput
from jiuwen.core.agent.controller.scheduler.message_handler import MessageHandler, MessageHandlerResult
from jiuwen.agent.common.enum import TaskType
from jiuwen.core.common.logging import logger
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.utils.llm.messages import HumanMessage
from jiuwen.core.agent.message.message import MessageType

class WorkflowState:
    """Workflow状态管理"""

    def __init__(self, runtime):
        self._runtime = runtime

    def is_interrupted(self) -> bool:
        """检查是否处于中断状态"""
        track_state = self._runtime.get_state("workflow_state")
        return (track_state and
                track_state.get("status") == "interrupted" and
                track_state.get("tasks") is not None)

    def get_interrupted_tasks(self):
        """获取中断的任务列表"""
        state_data = self._runtime.get_state("workflow_state") or {}
        tasks_data = state_data.get("tasks", [])
        
        # 如果是字典（序列化后的），重新构建为 Task 对象
        tasks = []
        for task_data in tasks_data:
            if isinstance(task_data, dict):
                tasks.append(Task(**task_data))
            else:
                tasks.append(task_data)
        return tasks

    def save_interrupt_state(self, tasks):
        """保存中断状态"""
        self._runtime.update_state({
            "workflow_state": {
                "status": "interrupted",
                "tasks": tasks
            }
        })

    def get_current_status(self) -> str:
        """获取当前状态"""
        track_state = self._runtime.get_state("workflow_state")
        if not track_state:
            return "normal"
        return track_state.get("status", "normal")

    def set_status(self, status: str, tasks=None):
        """设置状态"""
        self._runtime.update_state({
            "workflow_state": {
                "status": status,
                "tasks": tasks or []
            }
        })


class WorkflowMessageHandler(MessageHandler):
    """WorkflowMessageHandler - 工作流模式的消息处理器，包含Workflow状态管理"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        super().__init__(config, context_engine, runtime)
        self._state = WorkflowState(runtime)  # 包含Workflow状态管理

    async def handle_message(self, message: Message) -> MessageHandlerResult:
        """处理工作流模式的消息，返回处理结果"""
        try:
            # 根据消息类型分发处理
            if message.msg_type == MessageType.TASK_COMPLETED:
                # 任务完成：写入流数据，决定是否停止
                return await self._handle_task_completed(message)

            elif message.msg_type == MessageType.TASK_INTERRUPTED:
                # 任务中断：写入流数据，保存状态，决定是否停止
                return await self._handle_task_interrupted(message)

            elif message.msg_type == MessageType.USER_INPUT:
                # 用户输入：根据当前状态决定是新任务还是恢复任务
                return await self._handle_user_input(message)

            elif message.msg_type == MessageType.ERROR:
                # 错误消息：发送错误流，停止工作流
                return await self._handle_error(message)

            else:
                logger.warning(f"Unsupported message type: {message.msg_type}")
                # 不支持的消息类型，继续处理下一个消息
                return MessageHandlerResult(tasks=[], should_continue=True)

        except Exception as e:
            logger.error(f"Error in WorkflowMessageHandler: {e}")
            # 发送错误流式消息
            error_result = await self._send_error_stream(str(e))
            # 返回停止信号
            return MessageHandlerResult(
                tasks=[],
                should_continue=False,
                final_result=error_result
            )

    async def _handle_task_completed(self, message: Message) -> MessageHandlerResult:
        """处理任务完成消息：写入流数据，清除中断状态，返回停止信号"""
        # 写入任务完成的流数据
        await self._write_message_stream_data(message)

        # 清除中断状态（如果有）
        if self._state.get_current_status() == "interrupted":
            self._state.set_status("normal")
            logger.info("Cleared interrupt state after task completed")

        # 返回停止信号，携带最终结果
        result = message.content.stream_data
        logger.info(f"Task completed, returning stop signal with final result")
        
        return MessageHandlerResult(
            tasks=[],
            should_continue=False,
            final_result=result
        )

    async def _handle_task_interrupted(self, message: Message) -> MessageHandlerResult:
        """处理任务中断消息：写入流数据，保存状态，返回停止信号"""
        # 写入任务中断的流数据
        await self._write_message_stream_data(message)

        # 从 extensions 中获取中断的 Task 对象（用于恢复）
        interrupted_task = message.content.extensions.get("interrupted_task")
        if interrupted_task:
            # 保存中断状态（保存整个 Task 对象，因为恢复时需要 task.input 等信息）
            self._state.save_interrupt_state([interrupted_task])
            logger.info(f"Task {message.context.task_id} interrupted, state saved")

        # 返回停止信号，携带中断结果（交互请求列表）
        result = message.content.stream_data
        logger.info(f"Task interrupted, returning stop signal with interaction requests")
        
        return MessageHandlerResult(
            tasks=[],
            should_continue=False,
            final_result=result
        )

    async def _handle_error(self, message: Message) -> MessageHandlerResult:
        """处理错误消息：发送错误流，返回停止信号"""
        error_msg = message.content.query if message.content.query else "Unknown error"
        logger.error(f"Received error message: {error_msg}")

        # 发送错误流式消息
        error_result = await self._send_error_stream(str(error_msg))
        
        # 返回停止信号，携带错误结果
        logger.info(f"Error occurred, returning stop signal with error result")
        
        return MessageHandlerResult(
            tasks=[],
            should_continue=False,
            final_result=error_result
        )

    async def _handle_user_input(self, message: Message) -> MessageHandlerResult:
        """处理用户输入消息：根据当前状态决定是新任务还是恢复任务"""
        # 根据状态判断是否需要继续执行任务
        if self._state.is_interrupted():
            return await self._handle_interrupted_message_from_message(message)
        else:
            return await self._handle_normal_message_from_message(message)

    async def _select_workflow_task(self, message: Message) -> Task:
        """选择工作流任务 - 返回Task而不是字符串"""
        # 从配置中获取工作流信息
        if len(self.config.workflows) != 1:
            raise NotImplementedError("Multi-workflow not implemented yet")

        workflow = self.config.workflows[0]

        # 过滤输入参数 - 使用 get_query() 统一获取查询文本
        filtered_inputs = self._filter_inputs(
            schema=workflow.inputs or {},
            user_data={"query": message.content.get_query()}
        )

        # 创建工作流任务
        task = Task(
            task_id=f"workflow_{message.msg_id}",
            task_type=TaskType.WORKFLOW,
            input=TaskInput(
                target_name=workflow.name,
                target_id=f"{workflow.id}_{workflow.version}",
                arguments=filtered_inputs,
            )
        )

        return task

    async def _handle_normal_message_from_message(self, message: Message) -> MessageHandlerResult:
        """处理正常消息，生成工作流任务"""
        task = await self._select_workflow_task(message)

        # 添加消息到聊天历史 - 使用统一的 get_query() 方法
        query_text = message.content.get_query()
        user_message = HumanMessage(content=query_text)
        self._add_msg_to_chat_history(user_message)

        if UserConfig.is_sensitive():
            logger.info(f"Added user message to chat history")
        else:
            logger.info(f"Added user message to chat history: {query_text}")

        # 返回任务，继续调度
        return MessageHandlerResult(
            tasks=[task],
            should_continue=True
        )

    async def _handle_interrupted_message_from_message(self, message: Message) -> MessageHandlerResult:
        """处理中断恢复的消息"""
        tasks = self._state.get_interrupted_tasks()
        if not tasks:
            self._state.set_status("normal")
            return await self._handle_normal_message_from_message(message)

        # 更新任务的输入参数 - 使用 interactive_input 字段
        # 对于中断恢复，应该传递 InteractiveInput 对象
        if message.content.interactive_input:
            tasks[0].input.arguments = message.content.interactive_input
        else:
            # 如果不是 InteractiveInput，尝试使用查询文本
            tasks[0].input.arguments = message.content.get_query()

        # 返回恢复的任务，继续调度
        return MessageHandlerResult(
            tasks=tasks,
            should_continue=True
        )

    def _validate_inputs(self, inputs: Dict):
        """验证输入参数"""
        if isinstance(inputs.get("query"), InteractiveInput):
            raise JiuWenBaseException(5000, "Non-interrupt status data format error.")

    @staticmethod
    def _filter_inputs(schema: dict, user_data: dict) -> dict:
        """过滤和验证用户输入"""
        if not schema:
            return {}

        required_fields = {
            k for k, v in schema.items()
            if isinstance(v, dict) and v.get("required") is True
        }

        filtered = {}
        for k in schema:
            if k not in user_data:
                if k in required_fields:
                    raise KeyError(f"missing required parameter: {k}")
                continue
            filtered[k] = user_data[k]

        return filtered

    async def _write_message_stream_data(self, message: Message):
        """写入消息携带的流数据 - 统一处理列表"""
        # stream_data 是 List[OutputSchema]
        stream_data = message.content.stream_data
        
        try:
            # 直接遍历列表，空列表会自动跳过
            for output_schema in stream_data:
                await self.runtime.write_stream(output_schema)
                logger.debug(f"Wrote stream data from message {message.msg_id}")

        except Exception as e:
            logger.warning(f"Failed to write message stream data: {e}")

    async def _send_error_stream(self, error_message: str):
        """发送错误流式消息并返回错误结果"""
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
            # 返回一个默认错误结果
            return OutputSchema(
                type="workflow_final",
                index=0,
                payload={
                    "error": True,
                    "message": str(e),
                    "status": "failed"
                }
            )

    def _add_msg_to_chat_history(self, message):
        """添加消息到聊天历史"""
        workflow_context = self.context_engine.get_workflow_context(
            workflow_id=self.config.workflows[0].id,
            session_id=self.runtime.session_id()
        )
        workflow_context.add_message(message)
