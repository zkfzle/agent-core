#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel

from openjiuwen.core.common import BaseCard
from openjiuwen.core.foundation.tool import ToolInfo
from openjiuwen.core.session.stream import OutputSchema, CustomSchema, TraceSchema

WORKFLOW_DRAWABLE = "WORKFLOW_DRAWABLE"


class WorkflowCard(BaseCard):
    version: str = ''
    inputs_schema: Optional[dict[str, Any] | BaseModel] = None

    def tool_info(self):
        return ToolInfo(
            name=self.name,
            description=self.description,
            parameters={
                "type": self.inputs_schema.type if self.inputs_schema else None,
                "properties": self.inputs_schema.properties if self.inputs_schema else None,
                "required": self.inputs_schema.required if self.inputs_schema else None,
            }
        )


class WorkflowExecutionState(Enum):
    COMPLETED = "COMPLETED"
    INPUT_REQUIRED = "INPUT_REQUIRED"


class WorkflowOutput(BaseModel):
    result: Any
    state: WorkflowExecutionState


WorkflowChunk = Union[OutputSchema, CustomSchema, TraceSchema]


def generate_workflow_key(workflow_id: str, workflow_version: str) -> str:
    return f"{workflow_id}_{workflow_version}"
