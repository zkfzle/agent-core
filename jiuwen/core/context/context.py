#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import uuid
from abc import ABC, abstractmethod
from typing import Any

from jiuwen.core.context.config import Config
from jiuwen.core.context.mq_manager import MessageQueueManager
from jiuwen.core.context.state import State, InMemoryState
from jiuwen.core.context.store import Store
from jiuwen.core.context.model_context.model_context import ModelContext, WorkflowModelContext, NodeModelContext
from jiuwen.core.runtime.callback_manager import CallbackManager
from jiuwen.core.stream.manager import StreamWriterManager
from jiuwen.core.tracer.tracer import Tracer


class Context(ABC):
    @abstractmethod
    def config(self) -> Config:
        pass

    @abstractmethod
    def context(self) -> ModelContext:
        pass

    @abstractmethod
    def state(self) -> State:
        pass

    @abstractmethod
    def store(self) -> Store:
        pass

    @abstractmethod
    def tracer(self) -> Any:
        pass

    @abstractmethod
    def stream_writer_manager(self) -> StreamWriterManager:
        pass

    @abstractmethod
    def callback_manager(self) -> CallbackManager:
        pass

    @abstractmethod
    def controller_context_manager(self):
        pass

    @abstractmethod
    def queue_manager(self) -> MessageQueueManager:
        pass

    @abstractmethod
    def session_id(self) -> str:
        pass

    def set_controller_context_manager(self, controller_context_manager) -> None:
        return

    def set_tracer(self, tracer: Tracer) -> None:
        return

    def set_stream_writer_manager(self, stream_writer_manager: StreamWriterManager) -> None:
        return

    def set_queue_manager(self, queue_manager: MessageQueueManager):
        return


class WorkflowContext(Context):
    def __init__(self, state: State = None, config: Config = None, store: Store = None, tracer: Tracer = None,
                 session_id: str = None, parent_model_context: ModelContext = None,
                 controller_context_manager: Any = None):
        self.__config = config if config is not None else Config()
        self.__model_context = WorkflowModelContext(session_id)
        self.__model_context.derive_from(parent_model_context)
        self.__state = state if state is not None else InMemoryState()
        self.__store = store
        self.__tracer = tracer
        self.__callback_manager = CallbackManager()
        self.__stream_writer_manager = None  # type: StreamWriterManager
        self.__controller_context_manager = controller_context_manager
        self.__session_id = session_id if session_id else uuid.uuid4().hex
        self.__queue_manager = None  # type: MessageQueueManager

    def context(self) -> ModelContext:
        return self.__model_context

    def set_stream_writer_manager(self, stream_writer_manager: StreamWriterManager) -> None:
        if self.__stream_writer_manager is not None:
            return
        self.__stream_writer_manager = stream_writer_manager

    def set_tracer(self, tracer: Tracer) -> None:
        self.__tracer = tracer

    def set_controller_context_manager(self, controller_context_manager) -> None:
        self.__controller_context_manager = controller_context_manager

    def config(self) -> Config:
        return self.__config

    def state(self) -> State:
        return self.__state

    def store(self) -> Store:
        return self.__store

    def tracer(self) -> Any:
        return self.__tracer

    def stream_writer_manager(self) -> StreamWriterManager:
        return self.__stream_writer_manager

    def callback_manager(self) -> CallbackManager:
        return self.__callback_manager

    def controller_context_manager(self):
        return self.__controller_context_manager

    def set_queue_manager(self, queue_manager: MessageQueueManager):
        if self.__queue_manager is not None:
            return
        self.__queue_manager = queue_manager

    def queue_manager(self) -> MessageQueueManager:
        return self.__queue_manager

    def session_id(self) -> str:
        return self.__session_id


class NodeContext(Context):
    def __init__(self, context: Context, node_id: str):
        self.__node_id = node_id
        self.__parent_id = context.executable_id() if isinstance(context, NodeContext) else ''
        self.__executable_id = self.__parent_id + "." + node_id if len(self.__parent_id) != 0 else node_id
        self.__state = context.state().create_node_state(self.__executable_id, self.__parent_id)
        self.__context = context
        self.__model_context = NodeModelContext(self.__node_id, self.session_id(),
                                                config=context.context().get_config(),
                                                parent_context=context.context())

    def context(self) -> ModelContext:
        return self.__model_context

    def node_id(self):
        return self.__node_id

    def executable_id(self):
        return self.__executable_id

    def parent_id(self):
        return self.__parent_id

    def tracer(self) -> Any:
        return self.__context.tracer()

    def state(self) -> State:
        return self.__state

    def config(self) -> Config:
        return self.__context.config()

    def store(self) -> Store:
        return self.__context.store()

    def stream_writer_manager(self) -> StreamWriterManager:
        return self.__context.stream_writer_manager()

    def callback_manager(self) -> CallbackManager:
        return self.__context.callback_manager()

    def controller_context_manager(self):
        return self.__context.controller_context_manager()

    def queue_manager(self) -> MessageQueueManager:
        return self.__context.queue_manager()

    def session_id(self) -> str:
        return self.__context.session_id()

    def parent_context(self):
        return self.__context