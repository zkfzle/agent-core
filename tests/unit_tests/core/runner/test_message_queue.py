#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import unittest
from typing import Any, AsyncIterator
import asyncio

from jiuwen.core.runner.message_queue_base import StreamQueueMessage, InvokeQueueMessage
from jiuwen.core.runner.message_queue_inmemory import MessageQueueInMemory


class MockMessagehandler_A:
    async def handle_message(self, request: Any) -> AsyncIterator[str]:
        for i in range(1, 10):
            response = f"MockMessagehandler_A response for msg : {request}, i is {i}"
            yield response


class MockMessagehandler_B:
    async def handle_message(self, request: Any) -> str:
        return "MockMessagehandler_B response for msg : " + request


class TestMessageQueue(unittest.IsolatedAsyncioTestCase):
    async def test_messagequeue_inmemory(self):
        mq = MessageQueueInMemory()
        mq.start()

        # topic_a
        subscription1 =  mq.subscribe("topic_a")
        subscription1.set_message_handler(MockMessagehandler_A().handle_message)
        subscription1.activate()
        # topic_b
        subscription2 =  mq.subscribe("topic_b")
        subscription2.set_message_handler(MockMessagehandler_B().handle_message)
        subscription2.activate()

        # send to topic_a
        message = StreamQueueMessage()
        message.request = "上海温度多少"
        message.response = asyncio.Future()
        await mq.produce_message("topic_a", message)
        response = await message.response
        i = 1
        async for ret in response:
            self.assertEqual(ret, f"MockMessagehandler_A response for msg : 上海温度多少, i is {i}")
            i += 1

        # send to topic_b
        message1 = InvokeQueueMessage()
        message1.request = "北京温度多少"
        message1.response = asyncio.Future()
        await mq.produce_message("topic_b", message1)
        reponse1 = await message1.response

        message2 = InvokeQueueMessage()
        message2.request = "北京人口多少"
        await mq.produce_message("topic_b", message2)
        reponse2 = await message2.response

        self.assertEqual(reponse1, "MockMessagehandler_B response for msg : 北京温度多少")
        self.assertEqual(reponse2, "MockMessagehandler_B response for msg : 北京人口多少")

        # agnetGroup 2
        mq.unsubscribe("topic_b")
        mq.unsubscribe("topic_a")

        mq.stop()
