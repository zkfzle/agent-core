#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Optional, Union, Dict, Any, List

from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.common.logging import logger
from jiuwen.core.common.enum.enum import MessageRole
from jiuwen.core.context.model_context.history.history import ConversationHistory
from jiuwen.core.context.model_context.variable.variable import VariableManager
from jiuwen.core.context.model_context.config import ModelContextConfig
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.context_engine.base import EngineInput, EngineOutput
from jiuwen.core.context.model_context.base import Serializable
from jiuwen.core.context.model_context.accessor.memory_accessor import MemoryAccessor


class ModelContext(Serializable):
    def __init__(self, session_id: str, config: ModelContextConfig = None):
        self.__session_id = session_id
        self.__config: ModelContextConfig = config or ModelContextConfig()
        self.__history: ConversationHistory = ConversationHistory()
        self.__variables: VariableManager = VariableManager()
        self.__init_variables(config)
        self.__engine: ContextEngine = ContextEngine(config.engine_config if config else None,
                                                     self.__create_context_engine_input,
                                                     self.__get_context_update_callbacks())
        self.__memory: MemoryAccessor = self.__init_memory(self.__config)

    def set_variable(self, key: str, value: Any):
        self.__variables.set(key, value)

    def get_variable(self, key: str) -> Optional[Any]:
        return self.__variables.get(key)

    def add_user_message(self,
                         message: Union[str, BaseMessage],
                         owner: List[str] = None,
                         tags: Dict[str, str] = None):
        self.__history.add_message(message, MessageRole.USER, owner, tags)
        if self.__config and self.__config.enable_memory:
            self.__memory.add_memory(message, filters={
                "session_id": self.__session_id,
            })

    def add_assistant_message(self,
                              message: Union[str, BaseMessage],
                              owner: List[str] = None,
                              tags: Dict[str, str] = None):
        self.__history.add_message(message, MessageRole.ASSISTANT, owner, tags)
        if self.__config and self.__config.enable_memory:
            self.__memory.add_memory(message, filters={
                "session_id": self.__session_id,
            })

    def get_history(self, last_n: int = -1, owner: Optional[str] = None) -> List[BaseMessage]:
        history = []
        last_n = self.__config.conversation_history_length if last_n <= 0 else last_n
        if self.__config and self.__config.enable_memory:
            try:
                history = [BaseMessage(role="user",
                                       content=self.__memory.search_summary(filters={"session_id": self.__session_id}))]
            except Exception:
                history = []
        history.extend(self.__history.get_history(last_n - len(history), owner)
                       if owner else self.__history.get_history(last_n - len(history)))
        return history


    def get_history_by_tags(self, tags: Dict[str, str],
                            last_n: int = -1, owner: Optional[str] = None) -> List[BaseMessage]:
        if not owner and not tags:
            return self.__history.get_all_history()
        return self.__history.get_history(last_n, owner, tags)

    def process(self,
                query: str,
                system_prompt: Union[str, Template] = None,
                variables: Dict[str, str] = None,
                tools: Union[str, Dict] = None) -> EngineOutput:
        engine_input = EngineInput(
            user_input=query,
            system_prompt=system_prompt or "",
            variables=self.__variables.serialize(),
            user_variables=variables or {},
            chat_history=self.__history.get_all_history(),
            tools=tools or "",
        )
        return self.__engine.process(engine_input)

    def set_config(self, config: ModelContextConfig):
        self.__config = config
        self.__engine.build_from_config(config.engine_config,
                                        self.__create_context_engine_input,
                                        self.__get_context_update_callbacks())
        self.__memory = self.__init_memory(config)

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

    def __init_memory(self, config):
        if config and config.memory_config:
            return MemoryAccessor(config.memory_config)
        return None

    def __init_variables(self, config: ModelContextConfig):
        if config and config.variables:
            for var in config.variables:
                self.__variables.set(var.name, var)

    def __create_context_engine_input(self) -> EngineInput:
        return EngineInput(
            variables=self.__variables.serialize(),
            chat_history=self.__history.get_all_history(),
        )

    def __get_context_update_callbacks(self):
        return dict(
            update_variable=self.__variables.update_variable
        )


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

    def add_assistant_message(self,
                              message: Union[str, BaseMessage],
                              owner: List[str] = None,
                              tags: Dict[str, str] = None):
        if not owner:
            owner = [self.__session_id]
        self.__history.add_message(message, MessageRole.ASSISTANT, owner, tags)

    def set_config(self, config: ModelContextConfig):
        self.__config = config
        engine_config = config.get_node_engine_config(self.__node_id)
        self.__engine.build_from_config(engine_config, self.__create_context_engine_input,
                                        self.__get_context_update_callbacks())