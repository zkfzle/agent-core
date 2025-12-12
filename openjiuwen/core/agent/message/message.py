#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

from openjiuwen.core.runtime.interaction.interactive_input import InteractiveInput


class MessageType(Enum):
    """消息类型枚举"""
    # 用户交互类
    USER_INPUT = "user_input"  # 用户输入消息

    # Agent交互类
    AGENT_RESPONSE = "agent_response"  # Agent响应消息
    AGENT_HANDOFF = "agent_handoff"  # Agent间切换

    # 任务执行类
    TASK_COMPLETED = "task_completed"  # 任务完成
    TASK_INTERRUPTED = "task_interrupted"  # 任务中断

    # 事件通知类
    ERROR = "error"  # 错误消息
    INFO = "info"  # 信息消息


class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class SourceType(Enum):
    """消息来源类型枚举"""
    USER = "user"  # 用户
    AGENT = "agent"  # Agent
    TASK = "task"  # 任务
    WORKFLOW = "workflow"  # 工作流
    SYSTEM = "system"  # 系统


@dataclass
class MessageSource:
    """消息来源信息"""
    conversation_id: str  # 对话ID
    source_type: SourceType  # 来源类型
    user_id: Optional[str] = None


@dataclass
class MessageContent:
    """消息内容 - 显式字段设计，消除魔法字符串，类型明确"""
    # 文本内容
    query: Optional[str] = None
    
    # 交互输入（用于中断恢复）
    interactive_input: Optional['InteractiveInput'] = None
    
    # 流数据 - 统一为列表类型，消除类型不一致
    stream_data: List[Any] = field(default_factory=list)  # List[OutputSchema]
    
    # 任务结果 - 明确类型，不再是 Any
    task_result: Optional[Any] = None  # TaskResult，避免循环导入暂时用 Any
    
    # 扩展字段（真正不确定的数据才放这里）
    extensions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """确保列表和字典字段不为 None"""
        if self.stream_data is None:
            self.stream_data = []
        if self.extensions is None:
            self.extensions = {}
    
    def get_query(self) -> str:
        """获取查询文本 - 统一处理所有情况"""
        # 优先返回 query
        if self.query is not None:
            return self.query
        
        # 如果有 interactive_input，提取其文本
        if self.interactive_input is not None:
            return self._extract_interactive_text(self.interactive_input)
        
        # 默认返回空字符串
        return ""
    
    @staticmethod
    def _extract_interactive_text(interactive_input: 'InteractiveInput') -> str:
        """从 InteractiveInput 中提取文本"""
        if interactive_input.raw_inputs is not None:
            return str(interactive_input.raw_inputs)
        
        if interactive_input.user_inputs:
            # 取第一个值
            return str(list(interactive_input.user_inputs.values())[0])
        
        return ""


@dataclass
class MessageContext:
    """消息上下文信息"""
    correlation_id: Optional[str] = None  # 关联ID（用于追踪消息链）
    conversation_id: Optional[str] = None  # 对话ID
    task_id: Optional[str] = None  # 相关任务ID
    workflow_id: Optional[str] = None  # 相关工作流ID

    def __post_init__(self):
        pass


@dataclass
class Message:
    """统一消息类"""
    # 基础信息
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    msg_type: MessageType = MessageType.USER_INPUT
    priority: MessagePriority = MessagePriority.NORMAL

    # 来源
    source: MessageSource = field(default_factory=lambda: MessageSource("unknown", SourceType.SYSTEM))

    # 内容
    content: MessageContent = field(default_factory=MessageContent)

    # 上下文
    context: MessageContext = field(default_factory=MessageContext)

    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)

    # 扩展元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    # AgentGroup 路由支持
    receiver_id: Optional[str] = None  # 目标 Agent ID（用于点对点发送）
    message_type: Optional[str] = None  # 自定义消息类型（用于订阅路由）

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    # ========== 工厂方法 ==========

    @classmethod
    def create_user_message(cls, content: Union[str, InteractiveInput], conversation_id: str = "default",
                            user_id: Optional[str] = None) -> 'Message':
        """创建用户消息 - 统一处理字符串和 InteractiveInput"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=SourceType.USER,
            user_id=user_id
        )
        
        # 根据类型分配到不同字段
        if isinstance(content, InteractiveInput):
            msg_content = MessageContent(interactive_input=content)
        else:
            msg_content = MessageContent(query=str(content))
        
        context = MessageContext(
            conversation_id=conversation_id,
            correlation_id=str(uuid.uuid4())
        )

        return cls(
            msg_type=MessageType.USER_INPUT,
            source=source,
            content=msg_content,
            context=context
        )

    @classmethod
    def create_agent_response(cls, content: str, conversation_id: str,
                              reply_to_msg_id: Optional[str] = None) -> 'Message':
        """创建Agent响应消息"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=SourceType.AGENT
        )
        msg_content = MessageContent(query=content)
        context = MessageContext(
            conversation_id=conversation_id,
            correlation_id=reply_to_msg_id
        )

        return cls(
            msg_type=MessageType.AGENT_RESPONSE,
            source=source,
            content=msg_content,
            context=context
        )

    @classmethod
    def create_agent_handoff(cls, conversation_id: str, to_agent_id: str,
                             handoff_reason: str) -> 'Message':
        """创建Agent切换消息"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=SourceType.AGENT
        )
        msg_content = MessageContent(
            query=handoff_reason,
            extensions={"to_agent_id": to_agent_id}
        )
        context = MessageContext(
            conversation_id=conversation_id
        )

        return cls(
            msg_type=MessageType.AGENT_HANDOFF,
            source=source,
            content=msg_content,
            context=context
        )

    @classmethod
    def create_task_completed(cls, conversation_id: str, task_id: str,
                              task_result: Any,  # TaskResult，避免循环导入暂时用 Any
                              workflow_id: Optional[str] = None,
                              stream_data: Optional[List[Any]] = None) -> 'Message':
        """创建任务完成消息"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=SourceType.TASK
        )

        # 处理 stream_data 默认值
        if stream_data is None:
            stream_data = []
        
        msg_content = MessageContent(
            stream_data=stream_data,
            task_result=task_result
        )

        context = MessageContext(
            conversation_id=conversation_id,
            task_id=task_id,
            workflow_id=workflow_id
        )

        return cls(
            msg_type=MessageType.TASK_COMPLETED,
            source=source,
            content=msg_content,
            context=context
        )

    @classmethod
    def create_task_interrupted(cls, conversation_id: str, task_id: str, reason: str,
                                task_result: Any,  # TaskResult，避免循环导入暂时用 Any
                                workflow_id: Optional[str] = None,
                                stream_data: Optional[List[Any]] = None) -> 'Message':
        """创建任务中断消息"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=SourceType.TASK
        )
        
        # 处理 stream_data 默认值
        if stream_data is None:
            stream_data = []
        
        msg_content = MessageContent(
            query=reason,
            stream_data=stream_data,
            task_result=task_result
        )
        context = MessageContext(
            conversation_id=conversation_id,
            task_id=task_id,
            workflow_id=workflow_id
        )

        return cls(
            msg_type=MessageType.TASK_INTERRUPTED,
            source=source,
            content=msg_content,
            context=context,
            priority=MessagePriority.HIGH
        )

    @classmethod
    def create_error_message(cls, conversation_id: str, error_msg: str,
                             source_type: SourceType = SourceType.SYSTEM) -> 'Message':
        """创建错误消息"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=source_type
        )
        msg_content = MessageContent(query=error_msg)

        return cls(
            msg_type=MessageType.ERROR,
            source=source,
            content=msg_content,
            priority=MessagePriority.HIGH
        )

    @classmethod
    def create_info_message(cls, conversation_id: str, info_msg: str,
                            source_type: SourceType = SourceType.SYSTEM) -> 'Message':
        """创建信息消息"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=source_type
        )
        msg_content = MessageContent(query=info_msg)

        return cls(
            msg_type=MessageType.INFO,
            source=source,
            content=msg_content
        )

    # ========== 便利方法 ==========

    def set_correlation(self, correlation_id: str) -> None:
        """设置关联ID"""
        self.context.correlation_id = correlation_id

    def set_conversation(self, conversation_id: str) -> None:
        """设置对话ID"""
        self.context.conversation_id = conversation_id

    def is_from_user(self) -> bool:
        """是否来自用户"""
        return self.source.source_type == SourceType.USER

    def is_from_agent(self) -> bool:
        """是否来自Agent"""
        return self.source.source_type == SourceType.AGENT

    def is_task_related(self) -> bool:
        """是否与任务相关"""
        return self.context.task_id is not None

    def is_workflow_related(self) -> bool:
        """是否与工作流相关"""
        return self.context.workflow_id is not None

    def get_display_content(self) -> str:
        """获取用于显示的内容"""
        return self.content.get_query()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""

        def convert_enum(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):  # 处理嵌套的dataclass
                result[key] = {k: convert_enum(v) for k, v in value.__dict__.items()}
            else:
                result[key] = convert_enum(value)

        # 特殊处理datetime
        if isinstance(self.created_at, datetime):
            result['created_at'] = self.created_at.isoformat()

        return result
