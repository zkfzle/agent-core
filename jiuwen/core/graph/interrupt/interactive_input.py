#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# class InteractiveInput(BaseModel):
class InteractiveInput(BaseModel):
    def __init__(self):
        """ user_input is a map of node_id to input, used together with interaction """
        self._user_input = {}

    @property
    def user_input(self):
        return self._user_input
    
    def update(self, node_id: str, value: Any):
        self._user_input[node_id] = value