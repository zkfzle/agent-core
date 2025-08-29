#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Dict, Any, Optional, Union, List, Tuple

from jiuwen.core.common.logging import logger
from jiuwen.core.common.enum.enum import MessageRole
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.context.model_context.accessor.base import BaseContextAccessor

try:
    import jiuwen.core.utils.memory.memory_engine_factory as memory_engine_factory
    from jiuwen.core.utils.memory.memory_engine_schema import MemorySearchRequest, MemorySearchResponse
    MEMORY_ENGINE_ENABLE = True
except Exception:
    MEMORY_ENGINE_ENABLE = False

DEFAULT_USER_ID: str = "<default_user_id>"
DEFAULT_APP_ID: str = "<default_app_id>"
DEFAULT_SESSION_ID: str = "<default_session_id>"
DEFAULT_AGENT_ID: str = "<default_agent_id>"


class MemoryAccessor(BaseContextAccessor):
    def __init__(self, config: Dict[str, Any]):
        self.__memory_core = None
        self.__config: Dict[str, Any] = config
        try:
            self.__memory_core = memory_engine_factory.open(config)
        except Exception:
            logger.warning("please install jiuwen.core.utils.memory.memory_engine")

    @staticmethod
    def __align_memory_role(role: str) -> str:
        if role == MessageRole.USER:
            return "human"
        elif role == MessageRole.ASSISTANT:
            return "ai"
        return role

    def add_memory(self, message: Union[BaseMessage, List[BaseMessage]], filters: Dict[str, str]):
        if isinstance(message, BaseMessage):
            return self.__add_single_memory(message, filters)
        elif isinstance(message, list):
            for msg in message:
                self.__add_single_memory(msg, filters)
            return
        # TODO: 异常处理

    def search_memory(self, query: str, num: int, filters: Dict[str, str]) -> List[Tuple[int, str]]:
        search_response = self.__memory_core.search_memory(MemorySearchRequest(
            agent_memory={
                "user_id": filters.get("user_id") or DEFAULT_USER_ID,
                "app_id": filters.get("app_id") or DEFAULT_APP_ID,
                "agent_id": filters.get("agent_id") or DEFAULT_SESSION_ID,
                "content": query,
                "num": num
            }
        ))
        return search_response.agent_memory or []

    def search_variables(self, variables: str, filters: Dict[str, str]) -> Optional[str]:
        search_response = self.__memory_core.search_memory(MemorySearchRequest(
            session_variable={
                "user_id": filters.get("user_id") or DEFAULT_USER_ID,
                "app_id": filters.get("app_id") or DEFAULT_APP_ID,
                "session_id": filters.get("session_id") or DEFAULT_SESSION_ID,
                "name": variables
            }
        ))
        return search_response.session_variable or ""

    def search_summary(self, filters: Dict[str, str]) -> str:
        search_response = self.__memory_core.search_memory(MemorySearchRequest(
            session_summary={
                "user_id": filters.get("user_id") or DEFAULT_USER_ID,
                "app_id": filters.get("app_id") or DEFAULT_APP_ID,
                "session_id": filters.get("session_id") or DEFAULT_SESSION_ID,
            }
        ))
        print("search_response.session_summary", search_response.session_summary)
        return search_response.session_summary or ""

    def __add_single_memory(self, message: BaseMessage, filters: Dict[str, str]):
        self.__memory_core.add_conversation_mem(
            user_id=filters.get("user_id") or DEFAULT_USER_ID,
            app_id=filters.get("app_id") or DEFAULT_APP_ID,
            session_id=filters.get("session_id") or DEFAULT_SESSION_ID,
            role=self.__align_memory_role(message.role),
            mem=message.content,
            config=self.__config,
            mem_async=False
        )
        print("add_memory", message.content)
