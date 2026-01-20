"""控制器数据模型定义

该模块定义了控制器相关的所有数据模型，包括：
- DataFrame: 数据帧（文本、文件、JSON）
- Event: 事件（输入事件、任务执行中交互事件、任务完成事件、任务失败事件）和事件类型
- ControllerOutput: 控制器输出（批处理和流式）
- Intent: 意图和意图类型
- Task: 任务和任务执行状态
"""
from openjiuwen.core.controller.schema.controller_output import (
    ControllerOutputPayload, ControllerOutputChunk, ControllerOutput
)
from openjiuwen.core.controller.schema.dataframe import TextDataFrame, FileDataFrame, JsonDataFrame, DataFrame
from openjiuwen.core.controller.schema.event import (
    EventType, Event, InputEvent, TaskInteractionEvent, TaskCompletionEvent, TaskFailedEvent
)
from openjiuwen.core.controller.schema.intent import IntentType, Intent
from openjiuwen.core.controller.schema.task import TaskStatus, Task


__all__ = [
    # 数据单元
    "TextDataFrame",
    "FileDataFrame",
    "JsonDataFrame",
    "DataFrame",
    # 事件（控制器输入）
    "EventType",
    "Event",
    "InputEvent",
    "TaskInteractionEvent",
    "TaskCompletionEvent",
    "TaskFailedEvent",
    # 控制器输出
    "ControllerOutputPayload",
    "ControllerOutputChunk",
    "ControllerOutput",
    # 意图
    "IntentType",
    "Intent",
    # 任务
    "TaskStatus",
    "Task",

]