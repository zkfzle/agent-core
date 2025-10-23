#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.task import Task, TaskResult


class TaskHandler:
    """TaskHandler - 任务处理器，负责执行任务，统一的任务执行入口"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        self.config = config
        self.context_engine = context_engine
        self.runtime = runtime

    async def execute(self, task: Task) -> TaskResult:
        """执行任务 - 统一的任务执行入口"""
        pass
