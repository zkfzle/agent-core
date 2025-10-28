#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""Controller of Agent"""

from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.controller.types import ControllerOutput, ControllerInput
from jiuwen.core.agent.controller.reasoner.agent_reasoner import AgentReasoner
from jiuwen.core.agent.controller.scheduler import MessageHandler, AgentScheduler, TaskHandler
from jiuwen.core.agent.handler.base import AgentHandler
from jiuwen.core.agent.message.message import Message
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.runtime.runtime import Runtime


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

    def set_agent_handler(self, agent_handler: AgentHandler):
        """设置Agent处理器"""
        self._agent_handler = agent_handler
        # 同时设置MessageHandler和TaskHandler的agent_handler引用
        if hasattr(self._message_handler, 'set_agent_handler'):
            self._message_handler.set_agent_handler(agent_handler)
        if hasattr(self._task_handler, 'set_agent_handler'):
            self._task_handler.set_agent_handler(agent_handler)

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
