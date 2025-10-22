#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    """调度器配置类"""
    # 队列配置
    max_message_queue_size: int = 1000
    max_task_queue_size: int = 1000

    # 调度配置
    scheduler_loop_interval: float = 0.1  # 调度循环间隔（秒）
    max_concurrent_messages: int = 5  # 最大并发处理消息数
    max_concurrent_tasks: int = 10  # 最大并发执行任务数
