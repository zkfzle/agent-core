#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import json
from openjiuwen.core.runner.drunner.dmessage_queue.message import DmqRequestMessage, DmqResponseMessage, DmqMessage, \
    DMessageType
from openjiuwen.core.runner.message_queue_base import QueueMessage


def serialize_message(message: QueueMessage) -> bytes:
    return message.model_dump_json().encode("utf-8")


def deserialize_message(data: bytes) -> QueueMessage:
    if not data:
        return None
    msg_dict = json.loads(data.decode("utf-8"))

    msg_type = msg_dict.get('type')
    if msg_type == DMessageType.INPUT:
        return DmqRequestMessage(**msg_dict)
    elif msg_type == DMessageType.OUTPUT:
        return DmqResponseMessage(**msg_dict)
    else:
        return DmqMessage(**msg_dict)
