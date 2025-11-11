#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict
import pulsar
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.runner.drunner.dmessage_queue.message_serializer import serialize_message, deserialize_message
from openjiuwen.core.runner.message_queue_base import (
    MessageQueueBase,
    SubscriptionBase,
    QueueMessage,
    AsyncMessageHandler,
)
from openjiuwen.core.runner.runner_config import PulsarConfig


class PulsarSubscription(SubscriptionBase):
    def __init__(self, topic: str, consumer: pulsar.Consumer, executor: ThreadPoolExecutor):
        self._topic = topic
        self._consumer = consumer
        self._executor = executor
        self._handler: Optional[AsyncMessageHandler] = None
        self._task: Optional[asyncio.Task] = None
        self._active = False

    def set_message_handler(self, handler: AsyncMessageHandler):
        self._handler = handler

    def activate(self):
        if not self._active:
            self._active = True
            self._task = asyncio.create_task(self._consume_loop())
            logger.info(f"[PulsarSubscription] activated topic={self._topic}")

    async def deactivate(self):
        if not self._active:
            return
        self._active = False

        if self._task:
            self._task.cancel()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._consumer.close)

        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info(f"[PulsarSubscription] deactivated topic={self._topic}")

    def is_active(self) -> bool:
        return self._active

    async def _consume_loop(self):
        loop = asyncio.get_running_loop()
        while self._active:
            try:
                msg = await loop.run_in_executor(
                    self._executor, lambda: self._consumer.receive(timeout_millis=1000)
                )
                data = msg.data()
                payload = deserialize_message(data)
                if self._handler:
                    await self._handler(payload)
                await loop.run_in_executor(self._executor, lambda: self._consumer.acknowledge(msg))
            except pulsar.Timeout:
                continue
            except Exception as e:
                logger.warning(f"[PulsarSubscription] receive error: {e}")


class MessageQueuePulsar(MessageQueueBase):
    """Pulsar MQ 封装"""

    def __init__(self, pulsar_config: PulsarConfig):
        self._url = pulsar_config.url
        self._max_workers = pulsar_config.max_workers or 8
        self._client: Optional[pulsar.Client] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._producers: Dict[str, pulsar.Producer] = {}
        self._subs: Dict[str, PulsarSubscription] = {}
        self._is_running = False
        self._lock = asyncio.Lock()

    def start(self):
        if self._is_running:
            return
        self._client = pulsar.Client(self._url)
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._is_running = True
        logger.info(f"[MessageQueuePulsar] started with url={self._url}")

    async def stop(self):
        if not self._is_running:
            return
        self._is_running = False
        for sub in self._subs.values():
            await sub.deactivate()
        self._executor.shutdown(wait=True)
        self._client.close()
        logger.info(f"[MessageQueuePulsar] stopped")

    def subscribe(self, topic: str) -> PulsarSubscription:
        if not self._is_running:
            raise JiuWenBaseException(StatusCode.MESSAGE_QUEUE_NOT_RUNNING.code,
                                      StatusCode.MESSAGE_QUEUE_NOT_RUNNING.errmsg.format(f"subscribe {topic} failed"))
        if topic in self._subs:
            return self._subs[topic]
        consumer = self._client.subscribe(topic, subscription_name=f"default-{topic}")
        # 整个pulsar所有操作复用同一个线程池
        sub = PulsarSubscription(topic, consumer, self._executor)
        self._subs[topic] = sub
        logger.info(f"[MessageQueuePulsar] Create new subscription, topic={topic}")
        return sub

    async def unsubscribe(self, topic: str):
        sub = self._subs.pop(topic, None)
        if sub:
            await sub.deactivate()
            logger.info(f"[MessageQueuePulsar] unsubscribed {topic}")

    async def produce_message(self, topic: str, message: QueueMessage):
        if not self._is_running:
            raise JiuWenBaseException(StatusCode.MESSAGE_QUEUE_NOT_RUNNING.code,
                                      "MQ stopped, cannot send message")
        content = serialize_message(message)
        if topic not in self._producers:
            async with self._lock:
                if topic not in self._producers:
                    loop = asyncio.get_running_loop()
                    producer = await loop.run_in_executor(
                        self._executor,
                        lambda: self._client.create_producer(topic)
                    )
                    self._producers[topic] = producer
                producer = self._producers[topic]
        else:
            producer = self._producers[topic]

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor,
                                   lambda: producer.send(content=content, partition_key=message.message_id))
        logger.debug(f"[MessageQueuePulsar] sent message to topic={topic}, id={message.message_id}")
