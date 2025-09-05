#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


# class InteractiveInput(BaseModel):
class InteractiveInput(BaseModel):
    user_inputs: Dict[str, Any] = Field(default_factory=dict)

    def update(self, node_id: str, value: Any):
        self.user_inputs[node_id] = value
