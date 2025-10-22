#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List, Optional
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.controller.enums import Message
from jiuwen.core.agent.controller.reasoner.agent_reasoner import AgentReasoner
from jiuwen.core.common.logging import logger


class MessageHandler:
    """MessageHandler - 消息处理器基类，负责处理消息并生成任务，支持自定义实现"""

    def __init__(self, agent_id: str, config: AgentConfig, context_engine=None, runtime=None):
        self.agent_id = agent_id
        self.config = config
        self.context_engine = context_engine
        self.runtime = runtime
        self.reasoner: Optional[AgentReasoner] = None

    def set_reasoner(self, reasoner: AgentReasoner):
        """设置决策器引用"""
        self.reasoner = reasoner

    async def process_message(self, message: Message) -> List[Task]:
        """处理消息，生成任务列表 - 统一的消息处理入口"""
        try:
            # 1. 消息预处理
            processed_message = await self.preprocess_message(message)

            # 2. 自定义消息处理逻辑（子类可重写）
            tasks = await self.handle_message(processed_message)

            # 3. 任务后处理
            processed_tasks = []
            for task in tasks:
                processed_task = await self.postprocess_task(task)
                processed_tasks.append(processed_task)

            return processed_tasks

        except Exception as e:
            logger.error(f"Error processing message {message.msg_id}: {e}")

    async def handle_message(self, message: Message) -> List[Task]:
        """消息处理逻辑 - 子类可重写此方法实现自定义逻辑"""
        pass

    async def preprocess_message(self, message: Message) -> Message:
        """消息预处理 """
        pass

    async def postprocess_task(self, task: Task) -> Task:
        """任务后处理"""
        pass
