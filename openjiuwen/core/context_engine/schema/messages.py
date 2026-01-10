#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from abc import abstractmethod
import uuid
from typing import List
from pydantic import Field

from openjiuwen.core.foundation.llm import BaseMessage


class OffloadMessage(BaseMessage):
    offload_id: str = Field(default_factory=lambda x: uuid.uuid4().hex)

    @abstractmethod
    async def reload(self) -> List[BaseMessage]:
        pass

    @abstractmethod
    async def offload(self, messages: List[BaseMessage]):
        pass


class MemoryOffloadMessage(OffloadMessage):
    original_messages: List[BaseMessage] = Field(default=None)

    async def reload(self) -> List[BaseMessage]:
        return self.original_messages

    async def offload(self, messages: List[BaseMessage]):
        self.original_messages = messages
