from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from jiuwen.agent.common.enum import SubTaskType


class SubTask(BaseModel):
    id: str = Field(default="")
    sub_task_type: SubTaskType = Field(default=SubTaskType.UNDEFINED)
    func_id: str = Field(default="")
    func_name: str = Field(default="")
    func_args: Any = Field(default_factory=dict)
    result: Optional[Union[str, dict]] = Field(default=None)
    sub_task_context: Any = Field(default=None)
