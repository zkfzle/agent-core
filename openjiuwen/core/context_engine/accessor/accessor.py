#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Optional
from openjiuwen.core.context_engine.accessor.history import ConversationHistory
from openjiuwen.core.context_engine.base import ContextWindow, ContextOwner
from openjiuwen.core.context_engine.config import ContextEngineConfig


class ContextAccessor:
    def __init__(self, config: ContextEngineConfig):
        self._chat_history: ConversationHistory = ConversationHistory()
        self._init_from_config(config)

    def history(self) -> ConversationHistory:
        return self._chat_history

    def create_context_window(self, owner: Optional[ContextOwner] = None) -> ContextWindow:
        return ContextWindow(
            chat_history=self._chat_history.get_messages(-1, owner=owner)
        )

    def _init_from_config(self, config: ContextEngineConfig):
        pass
