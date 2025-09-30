"""多Agent系统核心枚举类"""
from enum import Enum

class MessageState(Enum):
    """消息状态枚举"""
    QUEUED = "queued"
    PROCESSING = "processing"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


class ProcessStatus(Enum):
    """Runner处理状态"""
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    ERROR = "error"
    NO_MESSAGES = "no_messages"
    INTERRUPTED = "interrupted"


class QueueState(Enum):
    """队列状态枚举"""
    PROCESSING = "processing"
    SUSPENDED = "suspended"
    IDLE = "idle"
    DRAINING = "draining"


class AgentStatus(Enum):
    """代理状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"