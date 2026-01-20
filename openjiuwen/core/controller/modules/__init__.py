"""控制器内部模块

该模块包含控制器的核心功能模块：
- EventQueue: 事件队列，负责事件的发布和订阅
- TaskManager: 任务管理器，负责任务的CRUD和状态管理
- TaskScheduler: 任务调度器，负责任务的执行调度
- IntentRecognizer: 意图识别器，识别用户意图
- EventHandler: 事件处理器基类
- EventHandlerWithIntentRecognition: 基于意图识别的事件处理器
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
    "TaskManagerState",
    "TaskManager",
    # 任务执行调度
    "TaskExecutor",
    "TaskExecutorRegistry",
    "TaskScheduler",
    # 基于意图识别的事件处理
    "IntentRecognizer",
    "EventHandlerWithIntentRecognition"
]
