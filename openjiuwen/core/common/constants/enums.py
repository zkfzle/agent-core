#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Core enum constants"""

from enum import Enum


class ControllerType(Enum):
    """Controller type enumeration"""
    ReActController = "react"
    WorkflowController = "workflow"
    Undefined = "undefined"


class TaskType(Enum):
    """Task type enumeration"""
    PLUGIN = "plugin"
    WORKFLOW = "workflow"
    MCP = "mcp"
    UNDEFINED = "undefined"


class ComponentAbility(Enum):
    INVOKE = ("invoke", "batch in, batch out")
    STREAM = ("stream", "batch in, stream out")
    COLLECT = ("collect", "stream in, batch out")
    TRANSFORM = ("transform", "stream in, stream out")

    def __init__(self, _ability_name: str, desc: str):
        self._ability_name = _ability_name
        self._desc = desc

    @property
    def ability_name(self):
        return self._ability_name

    @property
    def desc(self) -> str:
        return self._desc
