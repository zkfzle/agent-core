import asyncio
import uuid
from typing import Dict, AsyncIterator, Any, Union, Optional

from jiuwen.core.runtime.thread_safe_dict import ThreadSafeDict
from jiuwen.core.runner.message_queue_base import MessageQueueBase, QueueMessage, AsyncMessageHandle


class Subscription:
    def __init__(self, handler: AsyncMessageHandle, max_size=10000):
        self._is_running = False
        self._queue_max_size = max_size
        self._queue = asyncio.Queue(maxsize=self._queue_max_size)
        self._timeout = 1.0
        self._results: Dict[str, asyncio.Future] = {}
        self._monitor_task = None
        self._handler = handler
        self._active_tasks = set()

    async def start(self):
        if not self._is_running:
            self._is_running = True
            self._monitor_task = asyncio.create_task(self._monitor_message())

    async def stop(self):
        if self._is_running:
            self._is_running = False
            if self._monitor_task:
                self._monitor_task.cancel()
                self._monitor_task = None
            # 确保任务被执行完/取消
            for task in self._active_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._active_tasks.clear()
            self._results.clear()
            # 清空self._queue
            self._queue = asyncio.Queue(maxsize=self._queue_max_size)

    async def _monitor_message(self):
        while self._is_running:
            try:
                queue_message = await self._queue.get()
                task = asyncio.create_task(self._consume_message(queue_message))
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Monitor error: {e}")

    async def produce_message(self, request: Any, reply=True) -> str:
        message_id = str(uuid.uuid4())
        queue_message = QueueMessage(message_id=message_id, request=request)
        if reply:
            future = asyncio.Future()
            self._results[message_id] = future
        await self._queue.put(queue_message)
        return message_id

    async def _consume_message(self, queue_message):
        handler = self._handler
        try:
            result = handler(queue_message.request)
            if isinstance(result, AsyncIterator):
                async def stream_response():
                    async for item in result:
                        yield item

                response = stream_response()
            else:
                response = await result
        except Exception as e:
            response = f"Error: {e}"
        future = self._results.get(queue_message.message_id)
        if future:
            future.set_result(response)

    async def get_response(self, message_id: str, timeout: Optional[float] = None) -> Union[Any, AsyncIterator[Any]]:
        future = self._results.get(message_id)
        message_result = None
        if future:
            message_result = await asyncio.wait_for(future, timeout=timeout)
            await self._results.pop(message_id)
        return message_result


class MessageQueueInMemory(MessageQueueBase):
    def __init__(self, queue_max_size=10000):
        self._is_running = False
        self._subscribers: ThreadSafeDict[str, Subscription] = ThreadSafeDict()
        self._queue_max_size = queue_max_size
        self._timeout = 1.0
        self._queue_lock = asyncio.Lock()

    async def start(self):
        self._is_running = True
        for sub in self._subscribers.values():
            await sub.start()

    async def stop(self):
        self._is_running = False
        for sub in self._subscribers.values():
            await sub.stop()
        self._subscribers.clear()

    async def subscribe(self, topic: str, handler: AsyncMessageHandle) -> Subscription:
        subscription = Subscription(handler=handler, max_size=self._queue_max_size)
        if topic in self._subscribers:
            raise ValueError(f"Topic '{topic}' is already subscribed.")
        self._subscribers[topic] = subscription
        if self._is_running:
            await subscription.start()
        return subscription

    async def unsubscribe(self, topic):
        if topic in self._subscribers:
            await self._subscribers[topic].stop()
            del self._subscribers[topic]
