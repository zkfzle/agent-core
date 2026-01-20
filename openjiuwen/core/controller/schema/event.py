"""事件数据模型定义

该模块定义了控制器事件相关的数据模型，包括：
- EventType: 事件类型枚举
- Event: 事件基类
- InputEvent: 输入事件
- TaskInteractionEvent: 任务执行中交互事件
- TaskCompletionEvent: 任务完成事件
- TaskFailedEvent: 任务失败事件

事件是控制器的主要输入形式，用于在控制器内部传递信息。
"""
from enum import Enum
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from openjiuwen.core.controller.schema.dataframe import DataFrame
from openjiuwen.core.controller.schema.task import Task


class EventType(str, Enum):
    """事件类型枚举
    
    定义所有支持的事件类型：
    - INPUT: 用户输入事件
    - TASK_INTERACTION: 任务交互事件（任务执行过程中需要用户交互）
    - TASK_COMPLETION: 任务完成事件
    - TASK_FAILED: 任务失败事件
    """
    INPUT = "input"
    TASK_INTERACTION = "task_interaction",
    TASK_COMPLETION = "task_completion",
    TASK_FAILED = "task_failed"


class Event(BaseModel):
    """事件基类
    
    所有事件的基类，包含事件类型、事件ID和元数据。
    """
    event_type: EventType
    event_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """初始化后处理
        
        确保metadata不为None。
        """
        if self.metadata is None:
            self.metadata = {}


class InputEvent(Event):
    """输入事件
    
    用户输入的事件，包含输入数据。
    这是控制器的主要输入类型，用于接收用户的请求。
    
    Attributes:
        event_type: 事件类型，固定为EventType.INPUT
        input_data: 输入数据列表，支持文本、文件和JSON格式
    """
    event_type: EventType = EventType.INPUT
    input_data: List[DataFrame] = Field(default_factory=list)

    @classmethod
    def from_user_input(cls, user_input: str) -> "Event":
        """从用户输入创建输入事件
        
        便捷方法，将字符串输入转换为InputEvent。
        
        Args:
            user_input: 用户输入的字符串
            
        Returns:
            Event: 输入事件对象
        """
        ...


class TaskInteractionEvent(Event):
    """任务交互事件
    
    任务执行过程中需要用户交互时产生的事件。
    当任务需要用户提供额外信息或确认时，会生成此事件。
    
    Attributes:
        event_type: 事件类型，固定为EventType.TASK_INTERACTION
        interaction: 交互内容列表，包含需要用户交互的信息
        task: 关联的任务对象
    """
    event_type = EventType.TASK_INTERACTION
    interaction: List[DataFrame] = Field(default_factory=list)
    task: Task = None


class TaskCompletionEvent(Event):
    """任务完成事件
    
    任务执行完成时产生的事件，包含任务结果。
    当任务成功完成时，会生成此事件并包含任务的输出结果。
    
    Attributes:
        event_type: 事件类型，固定为EventType.TASK_COMPLETION
        task_result: 任务结果列表，包含任务的输出数据
        task: 关联的任务对象
    """
    event_type = EventType.TASK_COMPLETION
    task_result: List[DataFrame] = Field(default_factory=list)
    task: Task = None


class TaskFailedEvent(Event):
    """任务失败事件
    
    任务执行失败时产生的事件，包含错误信息。
    当任务执行过程中发生错误时，会生成此事件并包含错误详情。
    
    Attributes:
        event_type: 事件类型，固定为EventType.TASK_FAILED
        error_message: 错误消息，描述任务失败的原因
        task: 关联的任务对象
    """
    event_type = EventType.TASK_FAILED
    error_message: str = None
    task: Task = None

