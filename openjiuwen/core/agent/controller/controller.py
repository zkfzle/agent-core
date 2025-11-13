#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""Controller of Agent"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional

from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.core.agent.controller.reasoner.agent_reasoner import AgentReasoner
from openjiuwen.core.agent.controller.scheduler import MessageHandler, AgentScheduler, TaskHandler
from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.context_engine.engine import ContextEngine
from openjiuwen.core.runner.message_queue_base import InvokeQueueMessage
from openjiuwen.core.runner.message_queue_inmemory import MessageQueueInMemory
from openjiuwen.core.runtime.runtime import Runtime


class Controller:
    """控制器"""

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


class BaseController(ABC):
    """基于消息队列的 Controller

    设计思想（Linus 风格）：
    - 使用 MessageQueueInMemory 实现发布-订阅
    - invoke() 发布消息到队列
    - handle_message() 由队列自动调用（订阅者）
    - 解耦生产者和消费者
    """

    def __init__(self):
        self.agent = None  # 反向引用，由 Agent 设置

        # 创建消息队列
        self.msg_queue = MessageQueueInMemory()
        self.msg_queue.start()

        # 订阅主题
        self.topic = "controller_messages"
        subscription = self.msg_queue.subscribe(self.topic)
        subscription.set_message_handler(self._handle_message_wrapper)
        subscription.activate()

    async def invoke(self, inputs: Dict, runtime: Runtime) -> Dict:
        """同步调用入口

        流程：
        1. 创建消息
        2. 发布消息到队列（produce_message）
        3. 等待处理结果
        """
        # 1. 创建消息
        message = self.create_message(inputs)

        # 2. 创建队列消息并发布
        queue_message = InvokeQueueMessage()
        queue_message.request = {"message": message, "runtime": runtime}
        queue_message.response = asyncio.Future()

        # 3. 发布到消息队列
        await self.msg_queue.produce_message(self.topic, queue_message)

        # 4. 等待结果
        result = await queue_message.response

        return result if result is not None else {"output": "processed"}

    async def _handle_message_wrapper(self, request: Dict) -> Dict:
        """消息处理包装器 - 由消息队列自动调用

        Args:
            request: 包含 message 和 runtime 的字典
        Returns:
            处理结果
        """
        message = request["message"]
        runtime = request["runtime"]
        return await self.handle_message(message, runtime)

    # ===== 抽象方法（开发者必须实现）=====
    @abstractmethod
    async def handle_message(self, message: Message, runtime: Runtime) -> Optional[Dict]:
        """处理消息的核心方法（必须实现）

        Args:
            message: 消息对象
            runtime: 运行时上下文
        Returns:
            Optional[Dict]: 处理结果

        开发者在这里实现所有业务逻辑：
        - 根据 message.msg_type 分发处理
        - 执行任务（同步或异步）
        - 管理状态
        - 处理中断
        - 如果需要多轮处理，在这个方法内部循环
        """
        pass

    # ===== 扩展方法（开发者可选择性重写）=====
    def create_message(self, inputs: Dict) -> Message:
        """创建消息对象（可重写）

        默认：从 inputs 中提取 content 和 metadata，创建用户输入消息
        """
        content = inputs.get("content", "")
        conversation_id = inputs.get("conversation_id", "default_session")

        return Message.create_user_message(
            content=content,
            conversation_id=conversation_id
        )

    def stop(self):
        """停止 controller - 清理资源"""
        self.msg_queue.unsubscribe(self.topic)
        self.msg_queue.stop()
