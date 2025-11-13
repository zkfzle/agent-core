import asyncio
import json
import uuid
from typing import Awaitable, AsyncIterator, Any
from pulsar import Client
from pulsar.asyncio import Client as AsyncClient, Producer

from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runtime.thread_safe_dict import ThreadSafeDict
from openjiuwen.core.runner.message_queue_base import (
    MessageQueueBase,
    SubscriptionBase,
    QueueMessage,
    InvokeQueueMessage,
    StreamQueueMessage,
    AsyncMessageHandler,
)

SENTINEL = object()

def serialize_message(message: Any) -> bytes:
    if message is None:
        return b""
    return json.dumps(message).encode("utf-8")

def deserialize_message(data: bytes) -> Any:
    if not data:
        return None
    return json.loads(data.decode("utf-8"))

async def queue_async_generator(q: asyncio.Queue, sentinel=None):
    while True:
        item = await q.get()
        if item is sentinel:
            break
        yield item
        q.task_done()


async def send_async(producer: Producer, content_json):
    logger.debug(f"send msg: {content_json}")
    content = serialize_message(content_json)
    await producer.send(content)


class ProducerPulsar():

    def __init__(self, topic: str, client: Client, async_client: AsyncClient, max_size=10000):
        self._queue_max_size = max_size
        self._queue = asyncio.Queue(maxsize=self._queue_max_size)
        self._recv_task = None
        self._send_task = None
        self._is_active = False
        self._topic = topic
        self._reply_topic = f"{topic}-reply-{uuid.uuid4()}"
        self._client = client
        self._async_client = async_client
        self._consumer = None
        self._producer = None
        self._pending_messages: ThreadSafeDict[str, QueueMessage] = ThreadSafeDict()

    async def activate(self):
        if not self._is_active:
            self._is_active = True
            self._consumer = self._client.subscribe(self._reply_topic, subscription_name=self._reply_topic + "-sub")
            self._producer = await self._async_client.create_producer(self._topic)
            self._send_task = asyncio.create_task(self._send_message())
            self._recv_task = asyncio.create_task(self._recv_response())

    async def deactivate(self):
        if self._is_active:
            self._is_active = False
            if self._recv_task:
                self._recv_task.cancel()
                try:
                    await self._recv_task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._recv_task = None

            if self._send_task:
                self._send_task.cancel()
                try:
                    await self._send_task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._send_task = None

            if self._consumer:
                self._consumer.close()
                self._consumer = None

            if self._producer:
                await self._producer.close()
                self._producer = None

            self._queue = asyncio.Queue(maxsize=self._queue_max_size)

    async def push_message(self, message: QueueMessage):
        await self._queue.put(message)

    async def _send_message(self):
        while self._is_active:
            message = await self._queue.get()
            content_json = {}
            if not message.message_id:
                message.message_id = str(uuid.uuid4())
            if isinstance(message, InvokeQueueMessage):
                self._pending_messages[message.message_id] = message
                content_json = {"reply_to": self._reply_topic, "message_id": message.message_id, "mode": "batch"}
            if isinstance(message, StreamQueueMessage):
                self._pending_messages[message.message_id] = message
                content_json = {
                    "reply_to": self._reply_topic,
                    "message_id": message.message_id,
                    "mode": "stream",
                }

            try:
                content_json["payload"] = message.payload
                await send_async(self._producer, content_json)
            except Exception as e:
                logger.warning(f"exception info: {e}")
                message.error_code = StatusCode.ERROR.code
                message.error_msg = str(e)
            finally:
                self._queue.task_done()

    async def _recv_response(self):
        while self._is_active:
            msg = await asyncio.to_thread(self._consumer.receive)
            if not msg:
                continue
            try:
                content_json = deserialize_message(msg.data())
                logger.debug(f"receive msg: {content_json}")
                message_id = content_json.get("message_id")
                error_code = content_json.get("error_code")
                error_code = int(error_code)
                error_msg = content_json.get("error_msg")
                if message_id and message_id in self._pending_messages:
                    message = self._pending_messages[message_id]
                    message.error_code = error_code
                    message.error_msg = error_msg

                    if error_code == StatusCode.SUCCESS.code:
                        data = content_json["response"]
                        if isinstance(message, InvokeQueueMessage):
                            message.response.set_result(data)
                            self._pending_messages.pop(message_id)
                        if isinstance(message, StreamQueueMessage):
                            if not hasattr(message, "reply_queue"):
                                message.reply_queue = asyncio.Queue()
                                gen = queue_async_generator(message.reply_queue, SENTINEL)
                                message.response.set_result(gen)
                            if data == f"{message_id}_end":
                                await message.reply_queue.put(SENTINEL)
                                self._pending_messages.pop(message_id)
                            else:
                                await message.reply_queue.put(data)
                    await asyncio.to_thread(self._consumer.acknowledge, msg)
                else:
                    await asyncio.to_thread(self._consumer.negative_acknowledge, msg)
            except Exception as e:
                logger.warning(f"exception info: {e}")
                try:
                    await asyncio.to_thread(self._consumer.negative_acknowledge, msg)
                except Exception as e:
                    logger.warning(f"exception info: {e}")
                    pass


class SubscriptionPulsar(SubscriptionBase):

    def __init__(self, topic: str, client: Client, async_client: AsyncClient):
        self._consume_task = None
        self._handler = None
        self._is_active = False
        self._timeout = 100.0
        self._client = client
        self._async_client = async_client
        self._topic = topic
        self._consumer = None
        self._producer_cache = {}

    def set_message_handler(self, handler: AsyncMessageHandler):
        self._handler = handler

    def activate(self):
        if not self._is_active:
            self._is_active = True
            self._consumer = self._client.subscribe(self._topic, subscription_name=self._topic + "-sub")
            self._consume_task = asyncio.create_task(self._consume_message())

    async def deactivate(self):
        if self._is_active:
            self._is_active = False
            if self._consume_task:
                self._consume_task.cancel()
                try:
                    await self._consume_task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._consume_task = None

            if self._consumer:
                self._consumer.close()
                self._consumer = None

            for _, producer in self._producer_cache.items():
                if producer:
                    await producer.close()
            self._producer_cache.clear()

    def is_active(self):
        return self._is_active

    async def _consume_message(self):
        while self._is_active and self._handler:
            msg = await asyncio.to_thread(self._consumer.receive)
            try:
                content_json = deserialize_message(msg.data())
                logger.debug(f"receive msg: {content_json}")
                payload = content_json.get("payload")
                response = self._handler(payload)
                if isinstance(response, Awaitable):
                    response = await asyncio.wait_for(response, timeout=self._timeout)

                reply_topic = content_json.get("reply_to")
                message_id = content_json.get("message_id")
                mode = content_json.get("mode")
                if reply_topic and message_id:
                    await self._handle_response(reply_topic, message_id, mode, response)
                await asyncio.to_thread(self._consumer.acknowledge, msg)
            except JiuWenBaseException as e:
                try:
                    producer = await self._get_producer(reply_topic)
                    content_json = {"message_id": message_id, "error_code": str(e.error_code), "error_msg": e.message}
                    content_json["response"] = e.message
                    await send_async(producer, content_json)

                    await asyncio.to_thread(self._consumer.acknowledge, msg)
                except Exception as e:
                    logger.warning(f"exception info: {e}")
                    pass
            except Exception as e:
                logger.warning(f"exception info: {e}")
                try:
                    await asyncio.to_thread(self._consumer.negative_acknowledge, msg)
                except Exception as e:
                    logger.warning(f"exception info: {e}")
                    pass

    async def _handle_response(self, reply_topic, message_id, mode, response):
        producer = await self._get_producer(reply_topic)
        if mode == "stream":
            if not response:
                raise JiuWenBaseException(StatusCode.ERROR.code, "Response is empty")
            if not isinstance(response, AsyncIterator):
                raise JiuWenBaseException(StatusCode.ERROR.code, "Stream mode need AsyncIterator response")

            async for resp in response:
                content_json = {"message_id": message_id, "error_code": str(StatusCode.SUCCESS.code), "error_msg": ""}
                content_json["response"] = resp
                await send_async(producer, content_json)

            # send end stream flags
            content_json = {"message_id": message_id, "error_code": str(StatusCode.SUCCESS.code), "error_msg": ""}
            content_json["response"] = f"{message_id}_end"
            await send_async(producer, content_json)
        else:
            if not response:
                raise JiuWenBaseException(StatusCode.ERROR.code, "Response is empty")
            if isinstance(response, AsyncIterator):
                raise JiuWenBaseException(StatusCode.ERROR.code, "Batch mode need not AsyncIterator response")
            content_json = {"message_id": message_id, "error_code": str(StatusCode.SUCCESS.code), "error_msg": ""}
            content_json["response"] = response
            await send_async(producer, content_json)

    async def _get_producer(self, topic: str):
        if topic not in self._producer_cache:
            self._producer_cache[topic] = await self._async_client.create_producer(topic)
        return self._producer_cache[topic]


class MessageQueuePulsar(MessageQueueBase):

    def __init__(self, pulsar_url: str, queue_max_size=10000):
        self._is_running = False
        self._producers: ThreadSafeDict[str, ProducerPulsar] = ThreadSafeDict()
        self._subscribers: ThreadSafeDict[str, SubscriptionPulsar] = ThreadSafeDict()
        self._url = pulsar_url
        self._queue_max_size = queue_max_size
        self._queue = asyncio.Queue(maxsize=self._queue_max_size)
        self._consume_task = None
        self._client = None
        self._async_client = None

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._client = Client(self._url)
            self._async_client = AsyncClient(self._url)
            self._consume_task = asyncio.create_task(self._push_message())

    async def stop(self):
        if self._is_running:
            self._is_running = False
            if self._consume_task:
                self._consume_task.cancel()
                try:
                    await self._consume_task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._consume_task = None
            for _, sub in self._subscribers.items():
                await sub.deactivate()
            for _, pro in self._producers.items():
                await pro.deactivate()
            self._queue = asyncio.Queue(maxsize=self._queue_max_size)
            self._client.close()
            await self._async_client.close()

    def subscribe(self, topic: str) -> SubscriptionBase:
        if topic in self._subscribers:
            raise ValueError(f"Topic '{topic}' is already subscribed.")
        subscription = SubscriptionPulsar(topic, self._client, self._async_client)
        self._subscribers[topic] = subscription
        return subscription

    async def unsubscribe(self, topic):
        if topic in self._subscribers:
            await self._subscribers[topic].deactivate()
            del self._subscribers[topic]

    async def produce_message(self, topic: str, message: QueueMessage):
        await self._queue.put((topic, message))

    async def _push_message(self):
        while self._is_running:
            topic, message = await self._queue.get()
            if topic not in self._producers:
                self._producers[topic] = ProducerPulsar(
                    topic, self._client, self._async_client, max_size=self._queue_max_size
                )
                await self._producers[topic].activate()

            await self._producers[topic].push_message(message)
            self._queue.task_done()
