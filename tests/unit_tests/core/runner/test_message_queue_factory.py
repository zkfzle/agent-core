#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import unittest
from typing import Any, AsyncIterator

from jiuwen.core.runner.message_queue_factory import MessageQueueFactory


class MockMessagehandler_A:
    async def handle_message(self, request: Any) -> AsyncIterator[str]:
        for i in range(1, 10):
            response = f"MockMessagehandler_B reponse for msg : {request}, i is {i}"
            yield response


class MockMessagehandler_B:
    async def handle_message(self, request: Any) -> str:
        return "MockMessagehandler_A reponse for msg : " + request


class TestMessageQueue(unittest.IsolatedAsyncioTestCase):
    async def test_messagequeue_inmemory(self):
        mq = MessageQueueFactory.get_message_queue()
        await mq.start()

        # agnetGroup 1
        subscription1 = await mq.subscribe("topic_a", MockMessagehandler_A().handle_message)
        # agnetGroup 2
        subscription2 = await mq.subscribe("topic_b", MockMessagehandler_B().handle_message)

        # agnetGroup 1 -> agnetGroup 2
        message_id_1 = await subscription1.produce_message("上海温度多少")
        result_1 = await subscription1.get_response(message_id_1, timeout=1.0)
        i = 1
        async for ret in result_1:
            self.assertEqual(ret, f"MockMessagehandler_B reponse for msg : 上海温度多少, i is {i}")
            i += 1

        # agnetGroup 2 -> agnetGroup 1
        message_id_1 = await subscription2.produce_message("北京温度多少")
        result_1 = await subscription2.get_response(message_id_1, timeout=1.0)
        message_id_2 = await subscription2.produce_message("北京人口多少")
        result_2 = await subscription2.get_response(message_id_2, timeout=1.0)
        self.assertEqual(result_1, "MockMessagehandler_A reponse for msg : 北京温度多少")
        self.assertEqual(result_2, "MockMessagehandler_A reponse for msg : 北京人口多少")

        # agnetGroup 2
        await mq.unsubscribe("topic_b")

        await mq.stop()
