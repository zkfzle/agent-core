#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Optional, Union, Dict, Any, List

from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.common.logging import logger
from jiuwen.core.common.enum.enum import MessageRole
from jiuwen.core.context.model_context.history import ConversationHistory
from jiuwen.core.context.model_context.config import ModelContextConfig
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.context.model_context.base import Serializable


class ModelContext(Serializable):
    def __init__(self, session_id: str, config: ModelContextConfig = None):
        self.__session_id = session_id
        self.__config: ModelContextConfig = config
        self.__history: ConversationHistory = ConversationHistory()
        self.__variables: Dict[str, Any] = dict()
        self.__engine: ContextEngine = ContextEngine()

    def set_variable(self, key: str, value: Any):
        self.__variables[key] = value

    def get_variable(self, key: str) -> Optional[Any]:
        return self.__variables.get(key)

    def add_user_message(self,
                         message: Union[str, BaseMessage],
                         owner: List[str] = None,
                         tags: Dict[str, str] = None):
        self.__history.add_message(message, MessageRole.USER, owner, tags)

    def add_assistant_message(self,
                              message: Union[str, BaseMessage],
                              owner: List[str] = None,
                              tags: Dict[str, str] = None):
        self.__history.add_message(message, MessageRole.ASSISTANT, owner, tags)

    def get_history(self, owner: Optional[str] = None) -> List[BaseMessage]:
        if not owner:
            return self.__history.get_all_history()
        return self.__history.get_history(owner)

    def get_history_by_tags(self, tags: Dict[str, str], owner: Optional[str] = None) -> List[BaseMessage]:
        if not owner and not tags:
            return self.__history.get_all_history()
        return self.__history.get_history(owner, tags)

    def set_config(self, config: ModelContextConfig):
        self.__config = config
        # TODO: 添加加載配置邏輯

    def get_config(self) -> ModelContextConfig:
        return self.__config

    def derive_from(self, parent_context: "ModelContext"):
        if parent_context:
            self.__history = parent_context.__history

    def serialize(self) -> Dict[str, Any]:
        serialized_context = dict(
            conversation_history=self.__history.serialize(),
            variables=self.__variables,
        )
        return serialized_context

    def deserialize(self, data: Dict[str, Any]):
        if not data:
            logger.warning("get empty context")
            return
        serialized_history = data.get("conversation_history", {})
        self.__history = self.__history.deserialize(serialized_history)


class AgentModelContext(ModelContext):
    def __init__(self, uid: str, config: Optional[ModelContextConfig] = None):
        super().__init__(uid, config)

    def create_workflow_model_context(self) -> "WorkflowModelContext":
        context = WorkflowModelContext(self.__session_id)
        context.derive_from(self)
        return context


class WorkflowModelContext(ModelContext):
    def __init__(self, session_id: str, config: Optional[ModelContextConfig] = None):
        super().__init__(session_id, config)


class NodeModelContext(ModelContext):
    def __init__(self, node_id: str, session_id: str,
                 config: Optional[ModelContextConfig] = None,
                 parent_context: Optional[ModelContext] = None):
        config = config or (parent_context.get_config() if parent_context else None)
        super().__init__(session_id, config)
        self.__node_id = node_id
        if parent_context:
            self.derive_from(parent_context)

    def add_user_message(self,
                         message: Union[str, Dict[str, Any]],
                         owner: List[str] = None,
                         tags: Dict[str, str] = None):
        if not owner:
            owner = [self.__session_id]
        self.__history.add_message(message, MessageRole.USER, owner, tags)