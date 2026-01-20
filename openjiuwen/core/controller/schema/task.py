"""任务数据模型定义

该模块定义了任务相关的数据模型，包括：
- TaskStatus: 任务状态枚举
- Task: 任务数据模型

任务状态流转：
submitted -> working -> (completed | failed | paused | canceled)
                |
                -> input-required -> (继续执行或取消)
"""
from enum import Enum
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field

from openjiuwen.core.controller.schema import InputEvent
from openjiuwen.core.controller.schema.controller_output import ControllerOutputChunk


class TaskStatus(str, Enum):
    """任务状态枚举
    
    定义任务的所有可能状态：
    - SUBMITTED: 已提交，等待执行
    - WORKING: 正在执行中
    - PAUSED: 已暂停
    - INPUT_REQUIRED: 需要用户输入
    - COMPLETED: 已完成
    - CANCELED: 已取消
    - FAILED: 执行失败
    - WAITING: 等待中（可能等待依赖任务完成）
    - UNKNOWN: 未知状态
    """
    SUBMITTED = "submitted"
    WORKING = "working"
    PAUSED = "paused"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    WAITING = "waiting"
    UNKNOWN = "unknown"


class Task(BaseModel):
    """任务模型
    
    定义任务的结构，包含任务的基本信息、状态、输入输出和层级关系。
    
    Attributes:
        session_id: 会话ID，标识任务所属的会话
        task_id: 任务ID，唯一标识一个任务
        task_type: 任务类型，用于查找对应的TaskExecutor
        description: 任务描述
        priority: 任务优先级，数字越小优先级越高，默认为1
        inputs: 所有跟本次任务有关的输入事件列表
        outputs: 任务执行过程中的输出帧列表
        Status: 任务状态
        parent_task_id: 父任务ID，用于构建任务层级关系
        context_id: 上下文ID，用于关联任务的上下文信息
        input_required_fields: 需要用户输入的字段定义（当状态为INPUT_REQUIRED时使用）
        error_message: 错误信息（当状态为FAILED时使用）
        metadata: 任务的元数据，可以存储额外的任务信息
    """
    session_id: str
    task_id: str
    task_type: str
    description: Optional[str]
    priority: int = 1
    inputs: List[InputEvent] = None
    outputs: List[ControllerOutputChunk] = Field(default_factory=list)
    Status: TaskStatus = TaskStatus.UNKNOWN
    parent_task_id: str = None
    context_id: str = None
    input_required_fields: Optional[Union[dict[str, Any], BaseModel]] = Field(default=None)
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

