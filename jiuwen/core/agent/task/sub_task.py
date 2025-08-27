from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from jiuwen.agent.common.enum import SubTaskType
from jiuwen.core.graph.interrupt.interactive_input import InteractiveInput


class SubTask(BaseModel):
    id: str = Field(default="")
    sub_task_type: SubTaskType = Field(default=SubTaskType.UNDEFINED)
    func_id: str = Field(default="")
    func_name: str = Field(default="")
    func_args: Union[dict, InteractiveInput] = Field(default_factory=dict)
    result: Optional[str] = Field(default=None)
    sub_task_context: Any = Field(default=None)
