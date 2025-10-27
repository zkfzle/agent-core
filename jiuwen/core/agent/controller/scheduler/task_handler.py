#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import json
from jiuwen.agent.config.base import AgentConfig
from jiuwen.agent.common.enum import TaskType
from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.handler.base import AgentHandler, AgentHandlerInputs
from jiuwen.core.runtime.workflow_manager import generate_workflow_key
from jiuwen.core.workflow.base import Workflow, WorkflowOutput
from jiuwen.core.common.logging import logger
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.agent.message.message import Message
from jiuwen.core.stream.base import OutputSchema


class TaskHandler:
    """TaskHandler - 任务处理器，负责执行任务，统一的任务执行入口"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        self.config = config
        self.context_engine = context_engine
        self.runtime = runtime
        self._agent_handler = None

    def set_agent_handler(self, agent_handler: AgentHandler):
        """设置Agent处理器引用"""
        self._agent_handler = agent_handler

    async def execute(self, task: Task) -> Message:
        """执行任务 - 统一的任务执行入口，返回消息"""
        try:
            # 根据任务类型执行不同的处理逻辑
            if task.task_type == TaskType.WORKFLOW:
                return await self._execute_workflow_task(task)
            elif task.task_type == TaskType.PLUGIN:
                return await self._execute_plugin_task(task)
            elif task.task_type == TaskType.MCP:
                return await self._execute_mcp_task(task)
            else:
                logger.warning(f"Unknown task type: {task.task_type}")
                # 返回错误消息
                return Message.create_error_message(
                    conversation_id=self.runtime.session_id(),
                    error_msg=f"Unknown task type: {task.task_type}"
                )

        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            # 返回错误消息
            return Message.create_error_message(
                conversation_id=self.runtime.session_id(),
                error_msg=str(e)
            )

    async def _execute_workflow_task(self, task: Task) -> Message:
        """执行工作流任务 - 返回消息"""
        try:
            inputs = AgentHandlerInputs(
                context=self.runtime,
                name=task.input.target_name,
                arguments=task.input.arguments
            )

            # 查找工作流
            workflow = self._find_workflow(inputs)

            # 创建工作流运行时
            workflow_runtime = inputs.context.create_workflow_runtime()
            result: WorkflowOutput = await workflow.invoke(inputs.arguments, workflow_runtime)

            # 准备流数据
            try:
                # 根据状态发送不同格式的结果
                if result and hasattr(result, 'state') and result.state.value == "INPUT_REQUIRED":
                    # 中断状态：直接发送交互请求列表
                    payload = result.result
                else:
                    # 完成状态：按照旧版本格式包装
                    payload = {"output": result, "result_type": "answer"}

                stream_data = OutputSchema(
                    type="workflow_final",
                    index=0,
                    payload=payload
                )
            except Exception as e:
                logger.warning(f"Failed to prepare stream data: {e}")
                stream_data = None

            # 保存结果到任务对象（用于后续状态检查）
            task.result = result

            # 根据执行结果创建不同类型的消息
            if result and hasattr(result, 'state') and result.state.value == "INPUT_REQUIRED":
                # 中断状态 - 创建 TASK_INTERRUPTED 消息
                logger.info(f"Task {task.task_id} interrupted, waiting for user input, state: {result.state.value}")
                message = Message.create_task_interrupted(
                    conversation_id=self.runtime.session_id(),
                    task_id=task.task_id,
                    reason="Workflow requires user input",
                    workflow_id=task.input.target_id,
                    stream_data=stream_data,
                    task_data=task  # 直接在工厂方法中传入 task 数据
                )
                logger.info(f"Created TASK_INTERRUPTED message: {message.msg_id}")
                return message
            else:
                # 完成状态 - 创建 TASK_COMPLETED 消息
                logger.info(f"Task {task.task_id} completed successfully")
                message = Message.create_task_completed(
                    conversation_id=self.runtime.session_id(),
                    task_id=task.task_id,
                    result=result,
                    workflow_id=task.input.target_id,
                    stream_data=stream_data,
                    task_data=task  # 直接在工厂方法中传入 task 数据
                )
                logger.info(f"Created TASK_COMPLETED message: {message.msg_id}")
                return message

        except Exception as e:
            if UserConfig.is_sensitive():
                error_msg = f"Workflow execution failed"
                logger.info(f"Task {task.input.target_name} failed.")
            else:
                error_msg = f"Workflow execution failed: {str(e)}"
                logger.error(f"Task {task.input.target_name} failed: {error_msg}")

            error_result = {
                "error": True,
                "id": "0",
                "value": str(e),
                "message": error_msg,
                "tool_name": task.input.target_name
            }
            task.result = json.dumps(error_result, ensure_ascii=False)

            # 返回错误消息，附带错误流数据
            message = Message.create_error_message(
                conversation_id=self.runtime.session_id(),
                error_msg=str(e)
            )
            return message

    def _find_workflow(self, inputs: AgentHandlerInputs) -> Workflow:
        """查找工作流"""
        context = inputs.context
        workflow_name = inputs.name
        workflow_metadata = self._agent_handler.search_workflow_metadata_by_workflow_name(workflow_name)
        workflow_id = generate_workflow_key(workflow_metadata.id, workflow_metadata.version)
        workflow = context.get_workflow(workflow_id)
        return workflow

    async def _execute_plugin_task(self, task: Task) -> Message:
        """执行插件任务"""
        # TODO: 实现插件调用逻辑
        return Message.create_error_message(
            conversation_id=self.runtime.session_id(),
            error_msg="Plugin task execution not implemented yet"
        )

    async def _execute_mcp_task(self, task: Task) -> Message:
        """执行MCP任务"""
        # TODO: 实现MCP调用逻辑
        return Message.create_error_message(
            conversation_id=self.runtime.session_id(),
            error_msg="MCP task execution not implemented yet"
        )
