import asyncio
import uuid
from abc import ABC
from typing import Dict, List, Any, Optional
from pydantic.dataclasses import dataclass


@dataclass
class QueueMessage:
    # 消息队列消息
    message_id: str
    message: Any


@dataclass
class MessageResult:
    # 消息处理结果
    message_id: str
    message: Any
    result: Any

class MessageHandler:
    async def handle_message(self, message: Any):
        pass


class Subscription:
    def __init__(self, topic, message_queue):
        self._topic = topic
        self._response_queue = asyncio.Queue(maxsize=1000)
        self._message_queue = message_queue
        self._message_handler = None
        self._pending_requests: Dict[str, asyncio.Future] = {}  # message_id处理请求
        self._is_active = False

    def set_message_handler(self, message_handler):
        self._message_handler = message_handler

    def activate(self):
        self._is_active = True

    def consume_message(self, queue_message):
        # message_id = queue_message.message_id
        # message = queue_message.message
        # 1.消费消息
        #result = await self._message_handler.handle_message(actual_message)
        # 2. 记录pending的request
        #self._pending_requests[message_id].set_result(result)
        # 3. 将结果放到响应队列
        # await self._response_queue.put(MessageResult(message_id=message_id, message=message, result=result))
        pass

    async def produce_message(self, message):
        # 生产一个消息到消息队列中
        queue_message = QueueMessage(message_id=str(uuid.uuid4()), message=message)
        await self._message_queue.produce_message(message)
        return queue_message.message_id

    async def get_response(self, message_id, timeout: Optional[float] = None):
        # result = await asyncio.wait_for(self._pending_requests[message_id], timeout=timeout)
        pass

    def is_active(self):
        # 是否还在订阅中
        return self._is_active

    def deactivate(self):
        # 停用订阅
        pass


class MessageQueue(ABC):
    def __init__(self):
        pass

    def start(self):
        pass

    def end(self):
        pass

    async def subscribe(self, topic) -> Subscription:
        pass

    async def unsubscribe(self, topic, subscription):
        pass

class LocalMessageQueue(MessageQueue):
    def __init__(self, max_size=10000):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=max_size)
        self.subscribers: Dict[str, List[Any]] = {}
        self._timeout = 1.0
        self._is_running = False

    async def start(self):
        # 启动消息队列
        pass

    async def stop(self):
        # 停止消息队列
        pass

    async def produce_message(self, topic: str, message: QueueMessage):
        # 添加一个消息
        await self._queue.put((topic, message))
        pass

    async def subscribe(self, topic: str) -> Subscription:
        subscription = Subscription(topic, self)
        self.subscribers[topic].append(subscription)
        return subscription

    async def unsubscribe(self, topic, subscription):
        pass

    async def _message_consume(self) -> None:
        while self._is_running:
            # 获取消息
            topic, message = await asyncio.wait_for(self._queue.get(), timeout=self._timeout)

            # 处理消息
            await self._dispatch_message(topic, message)

            # 标记任务完成
            self._queue.task_done()

        pass

    async def _dispatch_message(self, topic, message):
        # tasks = []
        # for subscription in self.subscribers:
        #     if subscription.is_active():
        #         # 创建一个任务
        #         task = asyncio.create_task(subscription.consume_message(message))
        #         tasks.append(task)
        #         # 增加一个任务
        #         subscription._processing_tasks.add(task)
        #         # 清理已完成的任务
        #         subscription._processing_tasks = {
        #             t for t in subscription._processing_tasks if not t.done()
        #         }
        #
        # # 等待所有处理完成
        # if tasks:
        #     await asyncio.gather(*tasks, return_exceptions=True)

        pass
