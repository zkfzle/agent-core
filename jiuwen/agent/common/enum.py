#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""enum constants"""

from enum import Enum, auto


class ControllerType(Enum):
    ReActController = "react"
    WorkflowController = "workflow"
    Undefined = "undefined"


class SubTaskType(Enum):
    PLUGIN = "plugin"
    WORKFLOW = "workflow"
    MCP = "mcp"
    UNDEFINED = "undefined"


class TaskStatus(Enum):
    PENDING = auto()  # 已创建，尚未开始
    RUNNING = auto()  # 正在执行
    SUCCESS = auto()  # 成功结束
    FAILED = auto()  # 异常结束
    CANCELLED = auto()  # 被取消


class ReActControllerStatus(Enum):
    """ReAct控制器状态枚举"""
    NORMAL = "NORMAL"  # 正常运行状态
    INTERRUPTED = "INTERRUPTED"  # 中断状态
    COMPLETED = "COMPLETED"  # 完成状态
    TIMEOUT = "TIMEOUT"  # 超时状态
