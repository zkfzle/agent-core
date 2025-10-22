#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import uuid


# 消息类型枚举
class MessageType(Enum):
    """消息类型枚举"""
    USER_INPUT = "user_input"  # 用户输入
    WORKFLOW_INTERRUPT = "workflow_interrupt"  # 工作流中断
    WORKFLOW_END = "workflow_end"  # 工作流执行完成
    ERROR = "error"  # 执行出错
    HANDOFF = "handoff"  # Agent间切换消息


class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageRole(Enum):
    """消息发送者角色"""
    USER = "USER"
    AGENT = "AGENT"


class TaskStatus(Enum):
    """任务状态枚举"""
    SUBMITTED = "submitted"  # 已提交
    PENDING = "pending"  # 等待执行
    WORKING = "working"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 执行失败
    CANCELLED = "cancelled"  # 已取消


class TaskType(Enum):
    """任务类型枚举"""
    TOOL_CALL = "tool_call"  # 工具调用
    WORKFLOW_CALL = "workflow_call"  # 工作流调用
    MCP_CALL = "mcp_call"  # MCP工具调用
    BUSINESS_LOGIC = "business_logic"  # 业务逻辑


# 消息数据模型
@dataclass
class MessageSource:
    """消息来源"""
    source_id: str
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageData:
    """消息数据"""
    content: str
    raw_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """消息基类"""
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    source: Optional[MessageSource] = None
    type: MessageType = MessageType.USER_INPUT
    data: Optional[MessageData] = None
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# 任务相关数据模型
@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
