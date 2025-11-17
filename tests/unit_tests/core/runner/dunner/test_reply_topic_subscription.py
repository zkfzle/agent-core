#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import pytest
import asyncio

import pytest_asyncio

from openjiuwen.core.runner.drunner.dmessage_queue.dsubscription.reply_topic_subscription import ReplyTopicSubscription
from openjiuwen.core.runner.drunner.dmessage_queue.message import DmqResponseMessage, DMessageType
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.runner.runner_config import RunnerConfig, DistributedConfig, MessageQueueConfig, get_runner_config


@pytest.mark.asyncio
class TestReplyTopicSubscription:
    def setup_method(self):
        fake_mq = RunnerConfig(
            distributed_mode=True,
            distributed_config=DistributedConfig(
                request_timeout=5.0,
                message_queue_config=MessageQueueConfig(
                    type="fake",
                )
            )
        )
        Runner.set_config(fake_mq)

    async def test_normal_message_reception(self):
        """测试正常注册并接收消息流程"""
        try:
            await Runner.start()
            reply_sub = Runner.system_reply_sub
            message_id = "test_msg_123"
            remote_id = "agent_456"
            collector = await reply_sub.register_collector(message_id, remote_id)

            # 验证collector注册
            assert reply_sub._make_key(remote_id, message_id) in reply_sub.collectors

            # 发送测试消息
            msg = DmqResponseMessage(
                type=DMessageType.OUTPUT,
                sender_id=remote_id,
                message_id=message_id,
                payload="test_payload",
                last_chunk=True,
            )
            await reply_sub.on_message(msg)

            # 验证消息接收
            result = await asyncio.wait_for(collector.result(), timeout=1.0)
            assert result == "test_payload"
        finally:
            await Runner.stop()

    async def test_unregistered_message_handling(self):
        """测试未注册collector时的消息处理"""
        try:
            await Runner.start()
            unreg_msg = DmqResponseMessage(
                type=DMessageType.OUTPUT,
                sender_id="unknown_agent",
                message_id="unknown_msg",
                payload="unregistered"
            )
            # 未注册的消息处理不应异常
            reply_sub = Runner.system_reply_sub
            await reply_sub.on_message(unreg_msg)
        finally:
            await Runner.stop()
