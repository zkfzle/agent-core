#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import dataclasses
import json
from enum import Enum

from openjiuwen.core.runner.drunner.dmessage_queue.message import DmqRequestMessage, DmqResponseMessage, DmqMessage, \
    DMessageType
from openjiuwen.core.runner.message_queue_base import QueueMessage


def serialize_message(message: QueueMessage) -> bytes:
    """支持 dataclass、Enum、对象自动序列化为 JSON"""
    if message is None:
        return b""

    def default(o):
        if isinstance(o, Enum):
            return o.value
        return str(o)

    return json.dumps(dataclasses.asdict(message), ensure_ascii=False, default=default).encode("utf-8")


def deserialize_message(data: bytes) -> QueueMessage:
    if not data:
        return None
    # 先解析为字典
    msg_dict = json.loads(data.decode("utf-8"))

    # 根据消息类型创建对应的具体对象
    msg_type = msg_dict.get('type')
    if msg_type == DMessageType.INPUT:
        # 使用关键字参数创建 DmqRequestMessage 对象
        return DmqRequestMessage(**msg_dict)
    elif msg_type == DMessageType.OUTPUT:
        # 如果有响应消息类型，类似处理
        return DmqResponseMessage(**msg_dict)
    else:
        # 默认返回基础消息类型
        return DmqMessage(**msg_dict)