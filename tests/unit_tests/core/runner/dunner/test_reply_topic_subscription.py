#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import pytest
import asyncio

import pytest_asyncio

from openjiuwen.core.runner.drunner.dmessage_queue.dsubscription.response_collector import ResponseCollector
from openjiuwen.core.runner.drunner.dmessage_queue.dsubscription.reply_topic_subscription import ReplyTopicSubscription
from openjiuwen.core.runner.drunner.dmessage_queue.message import DmqResponseMessage, DMessageType
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.runner.runner_config import RunnerConfig, DistributedConfig, MessageQueueConfig


@pytest_asyncio.fixture
async def reply_sub():
    """创建ReplyTopicSubscription实例并激活"""
    sub = ReplyTopicSubscription()
    sub.activate()
    yield sub
    # 清理操作
    await sub.unregister_collector()


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
    async def test_normal_message_reception(self, reply_sub):
        """测试正常注册并接收消息流程"""
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

    async def test_unregistered_message_handling(self, reply_sub):
        """测试未注册collector时的消息处理"""
        unreg_msg = DmqResponseMessage(
            type=DMessageType.OUTPUT,
            sender_id="unknown_agent",
            message_id="unknown_msg",
            payload="unregistered"
        )
        # 未注册的消息处理不应引发异常
        await reply_sub.on_message(unreg_msg)

    async def test_collector_cleanup_after_unregister(self, reply_sub):
        """测试取消注册后collector的清理"""
        message_id = "cleanup_test"
        remote_id = "cleanup_agent"
        # 显式指定参数以保持一致性
        collector = await reply_sub.register_collector(message_id, remote_id, request_id=None)
        # 验证collector已注册
        assert reply_sub._make_key(remote_id, message_id, request_id=None) in reply_sub.collectors

        await reply_sub.unregister_collector(message_id=message_id)
        assert len(reply_sub.collectors) == 0
