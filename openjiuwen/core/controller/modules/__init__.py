# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Controller Internal Modules

This module contains the core functional components of the controller:
- EventQueue: Manages the publishing and subscribing of events.
- TaskManager: Handles the CRUD operations and status management of tasks.
- TaskScheduler: Responsible for the execution scheduling of tasks.
- IntentRecognizer: Recognizes user intents.
- EventHandler: Base class for event handlers.
- EventHandlerWithIntentRecognition: Event handler based on intent recognition.
"""

from openjiuwen.core.controller.modules.event_handler import EventHandlerInput, EventHandler
from openjiuwen.core.controller.modules.event_queue import EventQueue
from openjiuwen.core.controller.modules.task_manager import TaskManagerState, TaskManager
from openjiuwen.core.controller.modules.task_scheduler import TaskExecutor, TaskExecutorRegistry, TaskScheduler
from openjiuwen.core.controller.modules.intent_reconizer import IntentRecognizer, EventHandlerWithIntentRecognition


__all__ = [
    # 事件队列和事件处理
    "EventHandlerInput",
    "EventHandler",
    "EventQueue",
    # 任务管理
    "TaskManager",
    "TaskManagerState",
    # 任务执行调度
    "TaskExecutor",
    "TaskExecutorRegistry",
    "TaskScheduler",
    # 基于意图识别的事件处理
    "IntentRecognizer",
    "EventHandlerWithIntentRecognition"
]