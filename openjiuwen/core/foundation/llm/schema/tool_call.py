from typing import Optional

from pydantic import BaseModel


class ToolCall(BaseModel):
    """
    工具调用
    Attributes:
        id: 工具调用ID
        type: 工具调用类型
        name: 工具名称
        arguments: 工具参数
        index: 工具调用索引，用于区分多个工具调用
    """
    id: Optional[str]
    type: str
    name: str
    arguments: str
    index: Optional[int] = None
