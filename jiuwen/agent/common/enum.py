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


class ReActStatus(Enum):
    INITIALIZED = "initialized"
    LLM_RESPONSE = "llm_response"
    TOOL_INVOKED = "tool_invoked"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"

class ReActEvent(Enum):
    NO_EVENT = "no_event"
    USER_INVOKE = "user_invoke"
    INVOKE_TOOL = "invoke_tool"
    INVOKE_TOOL_FINISHED = "invoke_tool_finished"
    INTERRUPT = "interrupt"
    FINISH = "finish"

class TaskStatus(Enum):
    PENDING = auto()  # 已创建，尚未开始
    RUNNING = auto()  # 正在执行
    SUCCESS = auto()  # 成功结束
    FAILED = auto()  # 异常结束
    CANCELLED = auto()  # 被取消

class WorkflowAgentStatus(Enum):
    INITIALIZED = "initialized"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"

class WorkflowAgentEvent(Enum):
    NO_EVENT = "no_event"
    USER_INVOKE = "user_invoke"