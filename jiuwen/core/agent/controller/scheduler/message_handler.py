#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.message.message import Message
from jiuwen.core.agent.controller.reasoner.agent_reasoner import AgentReasoner
from jiuwen.core.common.logging import logger


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
