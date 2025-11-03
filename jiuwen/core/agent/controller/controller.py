#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""Controller of Agent"""

from dataclasses import dataclass, field
from typing import List, Optional
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.controller.reasoner.agent_reasoner import AgentReasoner
from jiuwen.core.agent.controller.scheduler import MessageHandler, AgentScheduler, TaskHandler
from jiuwen.core.agent.message.message import Message
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.agent.task.task import Task

@dataclass
class ControllerState:
    """通用控制器状态：管理被中断的任务列表及其组件ID记录"""
    interrupted_tasks: List[Task] = field(default_factory=list)

    def is_interrupted(self) -> bool:
        return len(self.interrupted_tasks) > 0

    def clear_all(self):
        self.interrupted_tasks.clear()

    def clear_interrupted_task(self, workflow_id: str):
        self.interrupted_tasks = [
            task for task in self.interrupted_tasks
            if not (task.input and task.input.target_id == workflow_id)
        ]

    def add_interrupted_task(self, task: Task, component_id: Optional[str] = None,
                             component_id_key: str = "interrupted_component_id"):
        if not task:
            return
        if component_id:
            if not task.metadata:
                task.metadata = {}
            task.metadata[component_id_key] = component_id

        workflow_id = task.input.target_id if task.input else None
        if workflow_id:
            self.interrupted_tasks = [
                t for t in self.interrupted_tasks
                if not (t.input and t.input.target_id == workflow_id)
            ]
        self.interrupted_tasks.append(task)

    def get_interrupted_task(self, workflow_id: Optional[str] = None) -> Optional[Task]:
        if not self.interrupted_tasks:
            return None
        if workflow_id:
            for task in self.interrupted_tasks:
                if task.input and task.input.target_id:
                    if task.input.target_id == workflow_id:
                        return task
            # 兼容按名称匹配
            for task in self.interrupted_tasks:
                if task.input and task.input.target_name == workflow_id:
                    return task
            return None
        else:
            return self.interrupted_tasks[0] if self.interrupted_tasks else None

    def get_interrupted_component_id(self, workflow_id: Optional[str] = None,
                                     component_id_key: str = "interrupted_component_id") -> Optional[str]:
        task = self.get_interrupted_task(workflow_id)
        if task and task.metadata:
            return task.metadata.get(component_id_key)
        return None

class Controller:
    """统一控制器 - 支持多种Agent范式"""

    def __init__(self, config: AgentConfig, context_engine: ContextEngine,
                 runtime: Runtime, message_handler: MessageHandler):
        self._config = config
        self._context_engine = context_engine
        self._runtime = runtime
        self._agent_handler = None

        # 统一组件初始化
        self._scheduler = AgentScheduler(config)
        self._reasoner = AgentReasoner(config, context_engine, runtime)
        self._task_handler = TaskHandler(config, context_engine, runtime)

        # 直接使用传入的MessageHandler（包含自己的状态管理）
        self._message_handler = message_handler

        # 设置组件引用
        self._scheduler.set_handlers(self._message_handler, self._task_handler)
        self._message_handler.set_reasoner(self._reasoner)

    async def start(self):
        """启动控制器 - 启动调度器"""
        await self._scheduler.start()

    async def stop(self):
        """停止控制器 - 停止调度器"""
        await self._scheduler.stop()

    async def receive_message(self, message: Message):
        """接收消息 - 外部系统调用此接口，统一消息入口"""
        # 所有外部消息都统一放到AgentScheduler的消息队列中进行统一调度
        await self._scheduler.schedule_message(message)

    async def run_until_complete(self):
        """等待调度器运行完成，返回最终结果"""
        return await self._scheduler.run_until_complete()

    async def process_inputs(self, inputs: dict):
        """处理输入并等待完成 - 统一的调用入口
        
        Args:
            inputs: 输入字典，包含 query 和 conversation_id
            
        Returns:
            最终结果（从 MessageHandler 的 final_result 返回）
        """
        # 1. 创建消息
        session_id = inputs.get("conversation_id", "default_session")
        message = Message.create_user_message(
            content=inputs.get("query", ""),
            conversation_id=session_id
        )

        # 2. 发送消息到调度器
        await self._scheduler.schedule_message(message)

        # 3. 等待调度器完成并返回结果
        return await self._scheduler.run_until_complete()
