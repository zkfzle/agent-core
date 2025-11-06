import asyncio
import uuid
from typing import Awaitable
from jiuwen.core.common.logging import logger

from jiuwen.core.runtime.thread_safe_dict import ThreadSafeDict
from jiuwen.core.runner.message_queue_base import (
    MessageQueueBase,
    SubscriptionBase,
    QueueMessage,
    InvokeQueueMessage,
    StreamQueueMessage,
    AsyncMessageHandle,
)


class SubscriptionInMemory(SubscriptionBase):

    def __init__(self, max_size=10000):
        self._queue_max_size = max_size
        self._queue = asyncio.Queue(maxsize=self._queue_max_size)
        self._consume_task = None
        self._handler = None
        self._is_active = False
        self._timeout = 20.0

    def set_message_handler(self, handler: AsyncMessageHandle):
        self._handler = handler

    def activate(self):
        if not self._is_active:
            self._is_active = True
            self._consume_task = asyncio.create_task(self._consume_message())

    def deactivate(self):
        if self._is_active:
            self._is_active = False
            if self._consume_task:
                self._consume_task.cancel()
                self._consume_task = None
            self._queue = asyncio.Queue(maxsize=self._queue_max_size)

    def is_active(self):
        return self._is_active

    async def push_message(self, message: QueueMessage):
        if not message.message_id:
            message.message_id = str(uuid.uuid4())
        await self._queue.put(message)

    async def _consume_message(self):
        while self._is_active and self._handler:
            message = await self._queue.get()
            try:
                response = self._handler(message.request)
                if isinstance(response, Awaitable):
                    response = await asyncio.wait_for(response, timeout=self._timeout)
                if isinstance(message, InvokeQueueMessage) or isinstance(message, StreamQueueMessage):
                    if response:
                        message.response.set_result(response)
            except Exception as e:
                logger.error(f"Handle message error: {e}")
            finally:
                self._queue.task_done()


class MessageQueueInMemory(MessageQueueBase):
    def __init__(self, queue_max_size=10000):
        self._is_running = False
        self._subscribers: ThreadSafeDict[str, SubscriptionInMemory] = ThreadSafeDict()
        self._queue_max_size = queue_max_size
        self._queue = asyncio.Queue(maxsize=self._queue_max_size)
        self._consume_task = None

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._consume_task = asyncio.create_task(self._consume_message())

    def stop(self):
        if self._is_running:
            self._is_running = False
            if self._consume_task:
                self._consume_task.cancel()
                self._consume_task = None
            self._queue = asyncio.Queue(maxsize=self._queue_max_size)

    def subscribe(self, topic: str) -> SubscriptionInMemory:
        if topic in self._subscribers:
            raise ValueError(f"Topic '{topic}' is already subscribed.")
        subscription = SubscriptionInMemory(max_size=self._queue_max_size)
        self._subscribers[topic] = subscription
        return subscription

    def unsubscribe(self, topic):
        if topic in self._subscribers:
            self._subscribers[topic].deactivate()
            del self._subscribers[topic]

    async def produce_message(self, topic: str, message: QueueMessage):
        await self._queue.put((topic, message))

    async def _consume_message(self):
        while self._is_running:
            topic, message = await self._queue.get()
            if topic in self._subscribers and self._subscribers[topic].is_active():
                await self._subscribers[topic].push_message(message)
            self._queue.task_done()
