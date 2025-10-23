# 消息数据模型
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Union
from datetime import datetime


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


@dataclass
class MessageContent:
    """消息内容"""
    text: Optional[str] = None  # 文本内容
    data: Optional[Dict[str, Any]] = None  # 结构化数据

    def __post_init__(self):
        if self.data is None:
            self.data = {}

    def get_display_text(self) -> str:
        """获取用于显示的文本"""
        if self.text:
            return self.text
        elif self.data:
            return str(self.data)
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

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    # ========== 工厂方法 ==========

    @classmethod
    def create_user_message(cls, content: str, conversation_id: str = "default",
                            user_id: str = "user") -> 'Message':
        """创建用户消息"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=SourceType.USER
        )
        msg_content = MessageContent(text=content)
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
        msg_content = MessageContent(text=content)
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
            text=handoff_reason,
            data={"to_agent_id": to_agent_id}
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
    def create_task_completed(cls, conversation_id: str, task_id: str, result: Union[str, Dict[str, Any]],
                              workflow_id: Optional[str] = None) -> 'Message':
        """创建任务完成消息"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=SourceType.TASK
        )

        if isinstance(result, str):
            msg_content = MessageContent(text=result)
        else:
            msg_content = MessageContent(data=result)

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
                                workflow_id: Optional[str] = None) -> 'Message':
        """创建任务中断消息"""
        source = MessageSource(
            conversation_id=conversation_id,
            source_type=SourceType.TASK
        )
        msg_content = MessageContent(text=reason)
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
        msg_content = MessageContent(text=error_msg)

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
        msg_content = MessageContent(text=info_msg)

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
        return self.content.get_display_text()

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
