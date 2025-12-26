#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import uuid
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from openjiuwen.core.common.schema.workflow_spec import WorkflowSpec


class WorkflowMetadata(BaseModel):
    name: str = Field(default="")
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    version: str = Field(default="")
    description: str = Field(default="")


class WorkflowInputsSchema(BaseModel):
    type: str = Field(default="")
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class WorkflowConfig(BaseModel):
    metadata: Optional[WorkflowMetadata] = Field(default_factory=WorkflowMetadata)
    spec: Optional[WorkflowSpec] = Field(default_factory=WorkflowSpec)
    workflow_inputs_schema: Optional[WorkflowInputsSchema] = Field(default_factory=WorkflowInputsSchema)
    workflow_max_nesting_depth: int = Field(default=5, ge=0, le=10)
