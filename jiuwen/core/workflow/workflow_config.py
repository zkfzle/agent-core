#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from enum import Enum
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from jiuwen.core.runtime.state import Transformer


class CompIOConfig(BaseModel):
    inputs_schema: Optional[Dict] = None
    outputs_schema: Optional[Dict] = None
    inputs_transformer: Optional[Transformer] = None
    outputs_transformer: Optional[Transformer] = None


class WorkflowMetadata(BaseModel):
    name: str = Field(default="")
    id: str = Field(default="")
    version: str = Field(default="")


class NodeSpec(BaseModel):
    io_config: CompIOConfig
    stream_io_configs: CompIOConfig
    abilites: List[Any] = Field(default_factory=list)


class WorkflowSpec(BaseModel):
    comp_configs: Dict[str, NodeSpec] = Field(default_factory=dict)
    stream_edges: Dict[str, list[str]] = Field(default_factory=dict)


class WorkflowConfig(BaseModel):
    metadata: Optional[WorkflowMetadata] = Field(default=None)
    comp_configs: Dict[str, Any] = Field(default_factory=dict)
    stream_timeout: float = Field(default=0.2)
    spec: Optional[WorkflowSpec] = Field(default=None)


class ComponentAbility(Enum):
    INVOKE = ("invoke", "batch in, batch out")
    STREAM = ("stream", "batch in, stream out")
    COLLECT = ("collect", "stream in, batch out")
    TRANSFORM = ("transform", "stream in, stream out")

    def __init__(self, name: str, desc: str):
        self._name = name
        self._desc = desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def desc(self) -> str:
        return self._desc