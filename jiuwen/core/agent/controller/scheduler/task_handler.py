#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from jiuwen.agent.config.base import AgentConfig
from jiuwen.agent.common.enum import TaskType, TaskStatus
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.core.agent.task.task import Task, TaskResult
from jiuwen.core.runtime.workflow_manager import generate_workflow_key
from jiuwen.core.workflow.base import Workflow, WorkflowOutput
from jiuwen.core.common.logging import logger
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.agent.message.message import Message
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.runner.runner import Runner
from typing import Any


class TaskHandler:
    """TaskHandler - 任务处理器，负责执行任务，统一的任务执行入口"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        self.config = config
        self.context_engine = context_engine
        self.runtime = runtime

    async def execute(self, task: Task) -> Message:
        """执行任务 - 统一的任务执行入口，返回消息"""
        try:
            # 根据任务类型执行不同的处理逻辑
            if task.task_type == TaskType.WORKFLOW:
                return await self._execute_workflow_task(task)
            elif task.task_type == TaskType.PLUGIN:
                return self._execute_plugin_task(task)
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
        """执行工作流任务 - 高层协调者"""
        try:
            result = await self._run_workflow(task)
            return self._create_message_from_result(task, result)
        except Exception as e:
            return self._handle_workflow_error(task, e)

    async def _run_workflow(self, task: Task) -> WorkflowOutput:
        """执行工作流并返回结果"""
        workflow = self._find_workflow(task.input.target_name)
        workflow_runtime = self.runtime.create_workflow_runtime()
        result = await Runner.run_workflow(workflow, inputs=task.input.arguments, runtime=workflow_runtime)
        return result

    def _prepare_stream_data(self, result: WorkflowOutput) -> list:
        """准备流数据 - 统一为列表格式"""
        try:
            if self._is_workflow_interrupted(result):
                # 中断状态：直接使用 workflow 返回的 chunk 列表
                return result.result
            else:
                # 完成状态：包装成列表
                payload = {"output": result, "result_type": "answer"}
                return [OutputSchema(type="workflow_final", index=0, payload=payload)]
        except Exception as e:
            logger.warning(f"Failed to prepare stream data: {e}")
            return []

    def _create_message_from_result(self, task: Task, result: WorkflowOutput) -> Message:
        """根据工作流执行结果创建对应的消息"""
        stream_data = self._prepare_stream_data(result)
        
        if self._is_workflow_interrupted(result):
            return self._create_interrupted_message(task, result, stream_data)
        else:
            return self._create_completed_message(task, result, stream_data)

    def _create_message_from_plugin_result(self, task: Task, result: Any) -> Message:
        """根据插件执行结果创建对应的消息"""
        payload = {"output": result, "result_type": "answer"}
        stream_data = [OutputSchema(type="plugin_final", index=0, payload=payload)]

        task.result = TaskResult(
            status=TaskStatus.SUCCESS,
            output=result,
            metadata={"tool_name": task.input.target_name}
        )

        return Message.create_task_completed(
            conversation_id=self.runtime.session_id(),
            task_id=task.task_id,
            task_result=task.result,
            stream_data=stream_data
        )

    def _create_interrupted_message(self, task: Task, result: WorkflowOutput, stream_data: list) -> Message:
        """创建中断消息"""
        task.result = TaskResult(
            status=TaskStatus.INTERRUPTED,
            output=result,
            metadata={"state": result.state.value}
        )
        logger.info(f"Task {task.task_id} interrupted, waiting for user input, state: {result.state.value}")
        
        message = Message.create_task_interrupted(
            conversation_id=self.runtime.session_id(),
            task_id=task.task_id,
            reason="Workflow requires user input",
            task_result=task.result,
            workflow_id=task.input.target_id,
            stream_data=stream_data
        )
        message.content.extensions["interrupted_task"] = task
        logger.info(f"Created TASK_INTERRUPTED message: {message.msg_id}, task: {task.task_id}")
        return message

    def _create_completed_message(self, task: Task, result: WorkflowOutput, stream_data: list) -> Message:
        """创建完成消息"""
        task.result = TaskResult(
            status=TaskStatus.SUCCESS,
            output=result,
            metadata={"state": result.state.value if hasattr(result, 'state') else "completed"}
        )
        logger.info(f"Task {task.task_id} completed successfully, workflow: {task.input.target_name}")
        
        message = Message.create_task_completed(
            conversation_id=self.runtime.session_id(),
            task_id=task.task_id,
            task_result=task.result,
            workflow_id=task.input.target_id,
            stream_data=stream_data
        )
        logger.info(f"Created TASK_COMPLETED message: {message.msg_id}, task: {task.task_id}")
        return message

    def _handle_workflow_error(self, task: Task, error: Exception) -> Message:
        """处理工作流执行错误"""
        if UserConfig.is_sensitive():
            error_msg = "Workflow execution failed"
            logger.info(f"Task {task.input.target_name} failed.")
        else:
            error_msg = f"Workflow execution failed: {str(error)}"
            logger.error(f"Task {task.input.target_name} failed: {error_msg}")

        task.result = TaskResult(
            status=TaskStatus.FAILED,
            error=error_msg,
            metadata={
                "error_type": type(error).__name__,
                "tool_name": task.input.target_name
            }
        )

        return Message.create_error_message(
            conversation_id=self.runtime.session_id(),
            error_msg=str(error)
        )

    def _is_workflow_interrupted(self, result: WorkflowOutput) -> bool:
        """判断工作流是否处于中断状态"""
        return result and hasattr(result, 'state') and result.state.value == "INPUT_REQUIRED"

    def _find_workflow(self, workflow_name: str) -> Workflow:
        """查找工作流"""
        workflow_metadata = self._search_workflow_metadata_by_name(workflow_name)
        workflow_id = generate_workflow_key(workflow_metadata.id, workflow_metadata.version)
        workflow = self.runtime.get_workflow(workflow_id)
        return workflow

    def _search_workflow_metadata_by_name(self, workflow_name: str) -> WorkflowSchema:
        """根据工作流名称查找工作流元数据"""
        for workflow in self.config.workflows:
            if workflow.name == workflow_name:
                return workflow
        raise JiuWenBaseException(
            message=f"Workflow '{workflow_name}' not found in configuration"
        )

    async def _execute_plugin_task(self, task: Task) -> Message:
        """execute plugin task"""
        # Compatible with Runtime 1.0 interfaces
        if result := self.__run_plugin_in_runtime(task):
            return result

        # Compatible with Runner interfaces
        try:
            tool_id = task.input.target_name
            result = await Runner.run_tool(tool_id, task.input.arguments, runtime=self.runtime)
            return self._create_message_from_plugin_result(task, result)
        except Exception as e:
            return self._handle_plugin_error(task, e)

    def _handle_plugin_error(self, task: Task, error: Exception) -> Message:
        """处理插件执行错误"""
        if UserConfig.is_sensitive():
            error_msg = "Plugin execution failed"
            logger.info(f"Task {task.input.target_name} failed.")
        else:
            error_msg = f"Plugin execution failed: {str(error)}"
            logger.error(f"Task {task.input.target_name} failed: {error_msg}")

        task.result = TaskResult(
            status=TaskStatus.FAILED,
            error=error_msg,
            metadata={
                "error_type": type(error).__name__,
                "tool_name": task.input.target_name
            }
        )

        return Message.create_error_message(
            conversation_id=self.runtime.session_id(),
            error_msg=str(error)
        )

    async def _execute_mcp_task(self, task: Task) -> Message:
        """执行MCP任务"""
        # TODO: 实现MCP调用逻辑
        return Message.create_error_message(
            conversation_id=self.runtime.session_id(),
            error_msg="MCP task execution not implemented yet"
        )

    def __run_plugin_in_runtime(self, task):
        """Temporary interface for backward compatibility with Runtime interface"""
        tool_id = task.input.target_name
        plugin = self.runtime.get_tool(tool_id)
        if plugin is not None:
            import asyncio
            result = asyncio.run(plugin.ainvoke(task.input.arguments))
            return self._create_message_from_plugin_result(task, result)
        return None
