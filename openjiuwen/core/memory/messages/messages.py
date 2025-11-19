#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""Messages of Memory"""
from datetime import datetime

from ...utils.llm.messages import BaseMessage


class SeqMessage(BaseMessage):
    """
    Sequence Message
    """
    timestamp: datetime
