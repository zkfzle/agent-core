from typing import List

from pydantic import BaseModel, Field

from openjiuwen.core.common.schema import Param


class ToolInfo(BaseModel):
    type: str = Field(default="function")
    name: str = Field(default="")
    description: str = Field(default="")
    parameters: List[Param] = Field(default_factory=list)
