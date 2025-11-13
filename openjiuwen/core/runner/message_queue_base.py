import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, AsyncIterator, TypeVar
from openjiuwen.core.common.exception.status_code import StatusCode

Output = TypeVar("Output", covariant=True)


@dataclass
class QueueMessage:
    message_id: str = ""
    payload: Any = None
    error_code: int = StatusCode.SUCCESS.code
    error_msg: str = ""


@dataclass
class InvokeQueueMessage(QueueMessage):
    response: asyncio.Future[Output] = None

    def __post_init__(self):
        if self.response is None:
            self.response = asyncio.Future()


@dataclass
class StreamQueueMessage(QueueMessage):
    response: asyncio.Future[AsyncIterator[Output]] = None

    def __post_init__(self):
        if self.response is None:
            self.response = asyncio.Future()


AsyncMessageHandler = Callable[[Any], Awaitable[Any]]


class SubscriptionBase(ABC):

    def set_message_handler(self, handler: AsyncMessageHandler):
        pass

    def activate(self):
        pass

    async def deactivate(self):
        pass

    def is_active(self):
        pass


class MessageQueueBase(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    def subscribe(self, topic: str) -> SubscriptionBase:
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str):
        pass

    @abstractmethod
    async def produce_message(self, topic: str, message: QueueMessage):
        pass
