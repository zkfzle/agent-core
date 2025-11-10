from enum import Enum
from typing import Optional

from jiuwen.core.runner.message_queue_base import QueueMessage

from dataclasses import dataclass
import time


class DMessageType(str, Enum):
    """Distributed message type"""
    INPUT = "INPUT"
    STOP = "STOP"
    OUTPUT = "OUTPUT"


class ResultType(str, Enum):
    """ResultType"""
    MESSAGE = "MESSAGE"
    ERROR = "ERROR"


@dataclass
class DmqRequestMessage(QueueMessage):
    """分布式请求消息"""
    type: str = DMessageType.INPUT
    reply_topic: str = ""
    request_id: str = ""
    sender_id: str = ""
    receiver_id: str = ""
    enable_stream: bool = False
    expire_at: Optional[float] = None


@dataclass
class DmqResponseMessage(QueueMessage):
    """分布式响应消息"""
    type: str = DMessageType.OUTPUT
    result_type: ResultType = ResultType.MESSAGE
    request_id: str = ""
    sender_id: str = ""
    receiver_id: str = ""
    seq: int = 0
    last_chunk: bool = False
    expire_at: Optional[float] = None
