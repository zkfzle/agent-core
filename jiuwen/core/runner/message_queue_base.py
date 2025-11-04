import threading
from abc import ABC, abstractmethod, ABCMeta
from typing import Any, Callable, Awaitable

from pydantic import BaseModel
from enum import Enum


class QueueMessage(BaseModel):
    message_id: str
    request: Any


class MessageResult(BaseModel):
    message_id: str
    request: Any
    response: Any


AsyncMessageHandle = Callable[[Any], Awaitable[Any]]


class MessageQueueType(Enum):
    MessageQueueInMemory = "message_queue_inmemory"
    MessageQueueDistributed = "message_queue_distributed"


singleton_lock = threading.Lock()


class SingletonMeta(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        with singleton_lock:
            if cls not in cls._instances:
                cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
            return cls._instances[cls]


class MessageQueueBase(ABC, metaclass=SingletonMeta):
    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: AsyncMessageHandle):
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str):
        pass
