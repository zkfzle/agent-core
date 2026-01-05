#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from enum import Enum
from typing import Any, Union

from pydantic import BaseModel

from openjiuwen.core.common import BaseCard
from openjiuwen.core.session.stream import OutputSchema, CustomSchema, TraceSchema


WORKFLOW_DRAWABLE = "WORKFLOW_DRAWABLE"


class WorkflowCard(BaseCard):
    ...


class WorkflowExecutionState(Enum):
    COMPLETED = "COMPLETED"
    INPUT_REQUIRED = "INPUT_REQUIRED"


class WorkflowOutput(BaseModel):
    result: Any
    state: WorkflowExecutionState


WorkflowChunk = Union[OutputSchema, CustomSchema, TraceSchema]


def generate_workflow_key(workflow_id: str, workflow_version: str) -> str:
    return f"{workflow_id}_{workflow_version}"