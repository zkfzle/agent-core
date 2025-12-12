#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Any


class Message:
    def __init__(self, sender: str, target: str, payload: Any = None):
        self.sender = sender
        self.target = target
        self.payload = payload


class TriggerMessage(Message):
    """Activate a target node next superstep"""
    pass


class BarrierMessage(Message):
    """N→1 fan-in"""
    pass
