#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""Controller基础类型定义 - 避免循环导入"""

from typing import Union
from pydantic import BaseModel, Field
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput


class ControllerOutput(BaseModel):
    """控制器输出基类"""
    is_task: bool = False


class ControllerInput(BaseModel):
    """控制器输入基类"""
    query: Union[str, InteractiveInput] = Field(default="")

