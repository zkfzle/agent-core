#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import abstractmethod, ABC
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from jiuwen.core.common.enum.enum import MessageRole
from jiuwen.core.utils.llm.messages import BaseMessage, HumanMessage, AIMessage

class Serializable(ABC):
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]):
        pass


class ConversationMessage(BaseModel):
    order_id: int
    message: BaseMessage
    session: str = Field(default="")
    owner: List[str] = Field(default=[])
    tags: Dict[str, str] = Field(default={})

    @staticmethod
    def create_message_by_role(role: str, content: str) -> BaseMessage:
        if not role:
            return BaseMessage(content=content, role="unknown")
        if role == MessageRole.USER.value:
            return HumanMessage(content=content)
        elif role == MessageRole.ASSISTANT.value:
            return AIMessage(content=content)
        return BaseMessage(role="unknown",content=content)


class ContextVariable(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default="")
    value: Optional[str] = Field(default=None)
    extractable: bool = Field(default=True)