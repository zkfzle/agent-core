#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List
from openjiuwen.core.common.logging import logger
from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.agent.task.task import Task
from openjiuwen.agent.common.enum import TaskType, TaskStatus


class Planner:
    """Planner - 规划器，负责复杂任务的规划和任务分解"""

    def __init__(self, config, context_engine, runtime):
        """
        初始化Planner
        
        Args:
            config: Planner配置
            context_engine: 上下文引擎
            runtime: 运行时环境
        """
        self.config = config
        self.context_engine = context_engine
        self.runtime = runtime

    @staticmethod
    def _create_default_task(message: Message) -> Task:
        # 临时实现：返回一个默认任务
        return Task(
            task_type=TaskType.UNDEFINED,
            description=f"Planner task for message: {message.content.get_query() if message.content else 'No content'}",
            status=TaskStatus.PENDING,
            metadata={
                "original_message_id": message.msg_id,
                "task_source": "planner"
            }
        )

    async def process_message(self, message: Message) -> List[Task]:
        """
        处理消息，进行任务规划并生成任务列表
        
        Args:
            message: 输入消息
            
        Returns:
            List[Task]: 生成的任务列表
        """
        # TODO: 实现具体的任务规划逻辑
        # 1. 分析消息内容，识别复杂任务
        # 2. 制定执行计划，分解为多个子任务
        # 3. 生成任务列表，设置任务依赖关系
        # 4. 返回任务列表

        logger.debug(f"Processing message {message.msg_id} with Planner")
        return [self._create_default_task(message)]
