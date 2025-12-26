#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Optional, Dict, List

from pydantic import BaseModel, Field

from openjiuwen.core.common.constants.enums import ComponentAbility
from openjiuwen.core.session import Transformer


class CompIOConfig(BaseModel):
    inputs_schema: Optional[Dict] = None
    outputs_schema: Optional[Dict] = None
    inputs_transformer: Optional[Transformer] = None
    outputs_transformer: Optional[Transformer] = None


class NodeSpec(BaseModel):
    io_config: CompIOConfig
    stream_io_configs: CompIOConfig
    abilities: List[ComponentAbility] = Field(default_factory=list)


class WorkflowSpec(BaseModel):
    comp_configs: Dict[str, NodeSpec] = Field(default_factory=dict)
    stream_edges: Dict[str, list[str]] = Field(default_factory=dict)
    edges: Dict[str, list[str]] = Field(default_factory=dict)
