#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# 构造测试消息
import time
import uuid

from jiuwen.core.runner.drunner.dmessage_queue.message import DmqRequestMessage, DMessageType
from jiuwen.core.runner.drunner.dmessage_queue.message_serializer import serialize_message, deserialize_message

message_id = str(uuid.uuid4())
test_input = {"key": "value"}
timeout = 30.0

original_msg = DmqRequestMessage(
    type=DMessageType.INPUT,
    reply_topic="test-reply-topic",
    message_id=message_id,
    sender_id="test-sender",
    receiver_id="test-receiver",
    enable_stream=False,
    payload=test_input,
    expire_at=time.time() + timeout,
)

# 序列化
serialized_data = serialize_message(original_msg)
print(f"Serialized data: {serialized_data}")

# 反序列化
deserialized_msg = deserialize_message(serialized_data)
print(f"Deserialized message: {deserialized_msg}")


# 验证字段一致性
assert deserialized_msg.type == original_msg.type
assert deserialized_msg.reply_topic == original_msg.reply_topic
assert deserialized_msg.message_id == original_msg.message_id
assert deserialized_msg.sender_id == original_msg.sender_id
assert deserialized_msg.receiver_id == original_msg.receiver_id
assert deserialized_msg.enable_stream == original_msg.enable_stream
assert deserialized_msg.payload == original_msg.payload
assert abs(deserialized_msg.expire_at - original_msg.expire_at) < 0.1  # 允许小的时间差