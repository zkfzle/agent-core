#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import List, Optional, Union

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.context_engine.token.base import TokenCounter
from openjiuwen.core.foundation.llm import BaseMessage
from openjiuwen.core.foundation.tool import ToolInfo
from openjiuwen.core.context_engine.base import ModelContext, ContextWindow, ContextStats
from openjiuwen.core.context_engine.context.message_buffer import ContextMessageBuffer


class SessionModelContext(ModelContext):
    def __init__(self,
                 context_id: str,
                 session_id: str,
                 history_messages: List[BaseMessage],
                 window_size_limit: int,
                 token_counter: TokenCounter = None,
                 ):
        self._message_id = 0
        self._validate_and_init_messages(history_messages)
        self._context_id = context_id
        self._session_id = session_id
        self._message_buffer = ContextMessageBuffer(history_messages or [])
        self._window_size_limit = window_size_limit
        self._token_counter = token_counter

    def __len__(self):
        return self._message_buffer.size()

    def session_id(self) -> str:
        return self._session_id

    def context_id(self) -> str:
        return self._context_id

    async def add_messages(self, messages: BaseMessage | List[BaseMessage]) -> List[BaseMessage]:
        self._validate_and_init_messages(messages)
        self._message_buffer.add_back(messages)
        return messages if isinstance(messages, list) else [messages]

    def pop_messages(self, size: int = 1, with_history: bool = True) -> List[BaseMessage]:
        if size is not None and size < 0:
            raise JiuWenBaseException()
        popped_messages = self._message_buffer.pop_back(size, with_history)
        return popped_messages

    def get_messages(self, size: Optional[int] = None, with_history: bool = True) -> List[BaseMessage]:
        if size is not None and size < 0:
            raise JiuWenBaseException()
        messages = self._message_buffer.get_back(size, with_history=with_history)
        return messages

    def set_messages(self, messages: List[BaseMessage], with_history: bool = True):
        self._validate_and_init_messages(messages)
        self._message_buffer.set_messages(messages, with_history)

    def clear_messages(self, with_history: bool = True):
        self.pop_messages(len(self), with_history=with_history)
        return

    async def get_context_window(self,
                                 system_messages: List[BaseMessage] = None,
                                 tools: List[ToolInfo] = None,
                                 context_window_messages_limit: Optional[int] = None,
                                 **kwargs
                                 ) -> ContextWindow:
        if context_window_messages_limit is not None and context_window_messages_limit <= 0:
            raise JiuWenBaseException()
        if context_window_messages_limit is None:
            context_window_messages_limit = self._window_size_limit

        system_messages = system_messages or []
        system_messages_size = min(len(system_messages), context_window_messages_limit)
        system_messages = system_messages[:system_messages_size]

        context_messages_size = context_window_messages_limit - system_messages_size
        context_messages = self._message_buffer.get_back(context_messages_size)

        window = ContextWindow(
            system_messages=system_messages,
            context_messages=context_messages,
            tools=tools or []
        )
        window.statistic = self._stat_context_window(window)
        return window

    def statistic(self) -> ContextStats:
        messages = self.get_messages()
        stat = ContextStats()
        self._stat_messages(stat, messages)
        return stat

    def _stat_context_window(self, context_window: ContextWindow) -> ContextStats:
        messages = context_window.get_messages()
        tools = context_window.get_tools()
        stat = ContextStats()
        self._stat_messages(stat, messages)
        self._stat_tools(stat, tools)
        return stat

    def _stat_tools(self, stat: ContextStats, tools: List[ToolInfo]):
        def count_tools(tools: List[ToolInfo]) -> int:
            if self._token_counter:
                return self._token_counter.count_tools(tools)
            return 0

        stat.tools = len(tools)
        stat.tool_tokens = count_tools(tools)

    def _stat_messages(self, stat: ContextStats, messages: List[BaseMessage]):
        def count_message(message: BaseMessage) -> int:
            if self._token_counter:
                return self._token_counter.count_messages([message])
            return 0

        stat.total_messages = len(messages)
        for msg in messages:
            if msg.role == "assistant":
                stat.assistant_messages += 1
                stat.assistant_message_tokens += count_message(msg)
            elif msg.role == "user":
                stat.user_messages += 1
                stat.user_message_tokens += count_message(msg)
            elif msg.role == "system":
                stat.system_messages += 1
                stat.system_message_tokens += count_message(msg)
            elif msg.role == "tool":
                stat.tool_messages += 1
                stat.tool_message_tokens += count_message(msg)
        stat.tool_tokens += (
            stat.assistant_message_tokens +
            stat.user_message_tokens +
            stat.system_message_tokens +
            stat.tool_message_tokens
        )

    @staticmethod
    def _validate_and_init_messages(messages: BaseMessage | List[BaseMessage]):
        if isinstance(messages, BaseMessage):
            return
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, BaseMessage):
                    raise JiuWenBaseException()
            return
        raise JiuWenBaseException()