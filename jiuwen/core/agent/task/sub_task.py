from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator

from jiuwen.agent.common.enum import SubTaskType
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput


class SubTask(BaseModel):
    id: str = Field(default="")
    sub_task_type: SubTaskType = Field(default=SubTaskType.UNDEFINED)
    func_id: str = Field(default="")
    func_name: str = Field(default="")
    func_args: Any = Field(default_factory=dict)
    result: Optional[Union[str, dict]] = Field(default=None)
    sub_task_context: Any = Field(default=None)

    @field_validator('func_args', mode='before')
    @classmethod
    def validate_func_args(cls, v):
        """验证func_args字段，确保字典不会被错误地转换为InteractiveInput"""
        # 如果已经是InteractiveInput实例，直接返回
        if isinstance(v, InteractiveInput):
            return v
        # 如果是字典，直接返回字典
        elif isinstance(v, dict):
            return v
        # 其他情况直接返回原值
        else:
            return v
