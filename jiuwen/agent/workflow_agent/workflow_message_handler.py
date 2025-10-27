#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List, Dict, Any, Union
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.message.message import Message
from jiuwen.core.agent.task.task import Task, TaskInput
from jiuwen.core.agent.controller.scheduler.message_handler import MessageHandler
from jiuwen.agent.common.enum import TaskType
from jiuwen.core.common.logging import logger
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.utils.llm.messages import HumanMessage


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
        return state_data.get("tasks", [])

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
        self._agent_handler = None  # Agent处理器引用

    def set_agent_handler(self, agent_handler):
        """设置Agent处理器引用"""
        self._agent_handler = agent_handler

    async def handle_message(self, message: Message) -> List[Task]:
        """处理工作流模式的消息，生成工作流任务"""
        from jiuwen.core.agent.message.message import MessageType
        
        try:
            # 根据消息类型分发处理
            if message.msg_type == MessageType.TASK_COMPLETED:
                # 任务完成：写入流数据，结束工作流
                return await self._handle_task_completed(message)
                
            elif message.msg_type == MessageType.TASK_INTERRUPTED:
                # 任务中断：写入流数据，保存状态，等待用户输入
                return await self._handle_task_interrupted(message)
                
            elif message.msg_type == MessageType.USER_INPUT:
                # 用户输入：根据当前状态决定是新任务还是恢复任务
                return await self._handle_user_input(message)
                
            elif message.msg_type == MessageType.ERROR:
                # 错误消息：发送错误流，停止工作流
                return await self._handle_error(message)
                
            else:
                logger.warning(f"Unsupported message type: {message.msg_type}")
                return []

        except Exception as e:
            logger.error(f"Error in WorkflowMessageHandler: {e}")
            # 发送错误流式消息来终止迭代
            await self._send_error_stream(str(e))
            return []

    async def _handle_task_completed(self, message: Message) -> List[Task]:
        """处理任务完成消息：写入流数据，清除中断状态，结束工作流"""
        # 写入任务完成的流数据
        await self._write_message_stream_data(message)
        
        # 清除中断状态（如果有）
        if self._state.get_current_status() == "interrupted":
            self._state.set_status("normal")
            logger.info("Cleared interrupt state after task completed")
        
        # 任务完成，无需生成新任务
        logger.info(f"Task completed for message {message.msg_id}")
        return []
    
    async def _handle_task_interrupted(self, message: Message) -> List[Task]:
        """处理任务中断消息：写入流数据，保存状态"""
        from jiuwen.core.agent.task.task import Task
        
        # 写入任务中断的流数据
        await self._write_message_stream_data(message)
        
        # 从消息中获取 task 对象并保存中断状态
        task = message.content.task_data
        if isinstance(task, Task) and task.result:
            self._state.save_interrupt_state([task])
            logger.info(f"Task {task.task_id} interrupted, state saved")
        
        # 任务中断，等待用户输入，无需生成新任务
        return []
    
    async def _handle_error(self, message: Message) -> List[Task]:
        """处理错误消息：发送错误流，发送最终结果，终止工作流"""
        error_msg = message.content.query if message.content.query else "Unknown error"
        logger.error(f"Received error message: {error_msg}")
        
        # 1. 发送错误流式消息（中间状态）
        await self._send_error_stream(str(error_msg))
        
        # 错误消息不生成新任务，工作流停止
        return []
    
    async def _handle_user_input(self, message: Message) -> List[Task]:
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

    async def _handle_normal_message_from_message(self, message: Message) -> List[Task]:
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

        return [task]

    async def _handle_interrupted_message_from_message(self, message: Message) -> List[Task]:
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
        
        return tasks


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
        """写入消息携带的流数据"""
        # 从 data 字典中获取 stream_data
        stream_data = message.content.stream_data
        if not stream_data:
            return
            
        try:
            # 如果是列表，逐个写入（中断场景）
            if isinstance(stream_data, list):
                for output_scheme in stream_data:
                    await self.runtime.write_stream(output_scheme)
                    logger.debug(f"Wrote stream data from message {message.msg_id}")
            else:
                # 单个流数据（完成场景）
                await self.runtime.write_stream(stream_data)
                logger.debug(f"Wrote stream data from message {message.msg_id}")
                
        except Exception as e:
            logger.warning(f"Failed to write message stream data: {e}")

    async def _send_error_stream(self, error_message: str):
        """发送错误流式消息来终止迭代"""
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
        except Exception as e:
            logger.error(f"Failed to send error stream: {e}")

    def _add_msg_to_chat_history(self, message):
        """添加消息到聊天历史"""
        workflow_context = self.context_engine.get_workflow_context(
            workflow_id=self.config.workflows[0].id,
            session_id=self.runtime.session_id()
        )
        workflow_context.add_message(message)
