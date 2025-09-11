#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Optional
from functools import partial
from jiuwen.core.context_engine.accessor.history import ConversationHistory
from jiuwen.core.context_engine.accessor.variables import VariableManager
from jiuwen.core.context_engine.accessor.memory import MemoryAccessor
from jiuwen.core.context_engine.base import ContextWindow, ContextOwner
from jiuwen.core.context_engine.config import ContextEngineConfig


class ContextAccessor:
    def __init__(self, config: ContextEngineConfig):
        self._long_tern_memory = MemoryAccessor(None)
        self._chat_history: ConversationHistory = ConversationHistory()
        self._variables: VariableManager = VariableManager()
        self._update_callbacks = self._init_callbacks()
        self._init_from_config(config)

    def history(self) -> ConversationHistory:
        return self._chat_history

    def variables(self) -> VariableManager:
        return self._variables

    def create_context_window(self, owner: Optional[ContextOwner] = None) -> ContextWindow:
        return ContextWindow(
            variables=self._variables.get_all(),
            chat_history=self._chat_history.get_messages(-1, owner=owner)
        )

    def update_context_by_type(self, update_type: str, update_output: ContextWindow):
        callback = self._update_callbacks.get(update_type, None)
        if callback:
            callback(update_output)

    def _init_from_config(self, config: ContextEngineConfig):
        if not config:
            return
        for variable in config.variables:
            self._variables.set(variable.name, variable)

    def _init_callbacks(self):
        return dict(
            update_variables=partial(ContextUpdateCallbacks.update_variables, self._variables)
        )


class ContextUpdateCallbacks:
    @staticmethod
    def update_variables(variables: VariableManager, update_output: ContextWindow):
        for var_name, var_value in update_output.variables.items():
            variables.update(var_name, var_value)