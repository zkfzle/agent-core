#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from pydantic import BaseModel, Field
from typing import List, Dict, Union, Any

from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.context.model_context.base import ConversationMessage
from jiuwen.core.context.model_context.base import Serializable

DEFAULT_HISTORY_LENGTH = 20


class ConversationHistory(Serializable):
    def __init__(self):
        self.__history = []
        self.__expired_history = []
        self.__compressed_history = []
        self.__conversation_order_id = 0
        self.__history_capacity: int = DEFAULT_HISTORY_LENGTH

    def __len__(self):
        return len(self.__history)

    def add_message(self, content: Union[str, BaseMessage],
                    role: str = "",
                    owner: List[str] = None,
                    tags: Dict[str, str] = None):
        message = content if isinstance(content, BaseMessage) \
            else ConversationMessage.create_message_by_role(role, content)
        self.__history.append(ConversationMessage(
            order_id=self.__conversation_order_id,
            message=message,
            owner=owner or [],
            tags=tags or {}
        ))
        self.__conversation_order_id += 1

    def get_all_history(self) -> List[BaseMessage]:
        return [message.message for message in self.__history]

    def get_history(self,
                    num: int,
                    owner: str = None,
                    tags: Dict[str, str] = None) -> List[BaseMessage]:

        filtered_history = []
        for message in self.__history:
            if owner and message.owner != owner:
                continue
            matched = True
            if tags:
                for key, value in tags.items():
                    msg_tag = message.tags.get(key)
                    if msg_tag != value:
                        matched = False
                        break
            if not matched:
                continue
            filtered_history.append(message.message)
        return filtered_history[-1 * num:]

    def serialize(self) -> Dict:
        return dict(history=[message.model_dump() for message in self.__history])

    def deserialize(self, data: Dict[str, Any]):
        if not data:
            return
        history = data.get("history", [])
        for message in history:
            self.__history.append(ConversationMessage(**message))

    """ async update processing"""
