#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


# class InteractiveInput(BaseModel):
class InteractiveInput(BaseModel):
    # key is id of interaction, value is input for the id
    user_inputs: Dict[str, Any] = Field(default_factory=dict)

    # input not bind to any id, used for the first interaction
    raw_inputs: Any = Field(default=None)

    def __init__(self, raw_inputs: Any = None, /, **data: Any):
        super().__init__(**data)
        self.raw_inputs = raw_inputs

    def update(self, node_id: str, value: Any):
        self.user_inputs[node_id] = value
