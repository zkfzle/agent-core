#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from pydantic import BaseModel, Field
from typing import List, Dict, Union

from jiuwen.core.utils.llm.messages import BaseMessage


class HistoricalMessage(BaseModel):
    message: BaseMessage
    owner: List[str] = Field(default=[])
    tags: Dict[str, str] = Field(default={})


class ConversationHistory:
    def __init__(self):
        self.__history = []

    def add_message(self, message: Union[BaseMessage, Dict],
                    owner: List[str] = None,
                    tags: Dict[str, str] = None):
        pass

    def get_history(self, owner: List[str] = None,
                    tags: Dict[str, str] = None) -> List[BaseMessage]:
        pass