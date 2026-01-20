"""控制器输出数据模型定义

该模块定义了控制器输出的数据模型，包括：
- ControllerOutputPayload: 控制器输出负载
- ControllerOutputChunk: 控制器输出块（用于流式输出）
- ControllerOutput: 控制器输出（用于批处理输出）

输出类型：
- task_completion: 任务完成
- task_interaction: 任务交互（需要用户输入）
- task_failed: 任务失败
- processing: 处理中
"""
from typing import Optional, Dict, Any, List, Literal

from pydantic import BaseModel, Field

from openjiuwen.core.controller.schema.dataframe import DataFrame

from openjiuwen.core.controller.schema.event import EventType
from openjiuwen.core.session.stream.base import OutputSchema


class ControllerOutputPayload(BaseModel):
    """控制器输出负载
    
    包含输出的类型、数据和元数据信息。
    这是控制器输出的核心数据部分。
    
    Attributes:
        type: 输出类型，可以是任务完成、任务交互、任务失败或处理中
        data: 输出数据列表，包含实际的输出内容
        metadata: 元数据，可以包含额外的输出信息
    """
    type: Literal[EventType.TASK_COMPLETION, EventType.TASK_INTERACTION, EventType.TASK_FAILED, "processing"]
    data: List[DataFrame] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class ControllerOutputChunk(OutputSchema):
    """控制器输出块
    
    流式输出中的单个数据块，包含索引、类型、负载和是否为最后一块的标志。
    用于流式输出场景，支持逐步返回处理结果。
    
    Attributes:
        index: 输出块的索引，用于标识输出块的顺序
        type: 输出类型，固定为"controller_output"
        payload: 输出负载，包含实际的输出数据
        last_chunk: 是否为最后一块，用于标识流式输出是否结束
    """
    index: int
    type: str = "controller_output"
    payload: ControllerOutputPayload = None
    last_chunk = False


class ControllerOutput(BaseModel):
    """控制器输出
    
    批处理输出的结果，包含类型、数据列表和输入事件ID。
    用于非流式输出场景，一次性返回所有结果。
    
    Attributes:
        type: 输出类型，可以是任务完成、任务交互、任务失败或处理中
        data: 输出数据，可以是ControllerOutputChunk列表或字典
        input_event_id: 关联的输入事件ID，用于追踪输入输出关系
    """
    type: Literal[EventType.TASK_COMPLETION, EventType.TASK_INTERACTION, EventType.TASK_FAILED, "processing"]
    data: List[ControllerOutputChunk] | Dict
    input_event_id: Optional[str] = None

