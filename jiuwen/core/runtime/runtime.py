#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import uuid
from abc import ABC, abstractmethod
from typing import Any, Union, Optional

from jiuwen.core.runtime.callback_manager import CallbackManager
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.mq_manager import MessageQueueManager
from jiuwen.core.runtime.state import State, InMemoryState
from jiuwen.core.runtime.store import Store
from jiuwen.core.stream.manager import StreamWriterManager
from jiuwen.core.stream.writer import OutputSchema, StreamWriter
from jiuwen.core.tracer.tracer import Tracer


class BaseRuntime(ABC):
    @abstractmethod
    def config(self) -> Config:
        pass

    @abstractmethod
    def state(self) -> State:
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


class WorkflowRuntime(BaseRuntime):
    def __init__(self, state: State = None, config: Config = None, store: Store = None, tracer: Tracer = None,
                 session_id: str = None, controller_context_manager: Any = None):
        self._config = config if config is not None else Config()
        self._state = state if state is not None else InMemoryState()
        self._store = store
        self._tracer = tracer
        self._callback_manager = CallbackManager()
        self._stream_writer_manager = None  # type: StreamWriterManager
        self._controller_context_manager = controller_context_manager
        self._session_id = session_id if session_id else uuid.uuid4().hex
        self._queue_manager = None  # type: MessageQueueManager

    def set_stream_writer_manager(self, stream_writer_manager: StreamWriterManager) -> None:
        if self._stream_writer_manager is not None:
            return
        self._stream_writer_manager = stream_writer_manager

    def set_tracer(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def set_controller_context_manager(self, controller_context_manager) -> None:
        self._controller_context_manager = controller_context_manager

    def config(self) -> Config:
        return self._config

    def state(self) -> State:
        return self._state

    def tracer(self) -> Any:
        return self._tracer

    def stream_writer_manager(self) -> StreamWriterManager:
        return self._stream_writer_manager

    def callback_manager(self) -> CallbackManager:
        return self._callback_manager

    def controller_context_manager(self):
        return self._controller_context_manager

    def set_queue_manager(self, queue_manager: MessageQueueManager):
        if self._queue_manager is not None:
            return
        self._queue_manager = queue_manager

    def queue_manager(self) -> MessageQueueManager:
        return self._queue_manager

    def session_id(self) -> str:
        return self._session_id


class NodeRuntime(BaseRuntime):
    def __init__(self, runtime: BaseRuntime, node_id: str):
        self._node_id = node_id
        self._parent_id = runtime.executable_id() if isinstance(runtime, NodeRuntime) else ''
        self._executable_id = self._parent_id + "." + node_id if len(self._parent_id) != 0 else node_id
        self._state = runtime.state().create_node_state(self._executable_id, self._parent_id)
        self._runtime = runtime

    def node_id(self):
        return self._node_id

    def executable_id(self):
        return self._executable_id

    def parent_id(self):
        return self._parent_id

    def tracer(self) -> Tracer:
        return self._runtime.tracer()

    def state(self) -> State:
        return self._state

    def config(self) -> Config:
        return self._runtime.config()

    def stream_writer_manager(self) -> StreamWriterManager:
        return self._runtime.stream_writer_manager()

    def callback_manager(self) -> CallbackManager:
        return self._runtime.callback_manager()

    def controller_context_manager(self):
        return self._runtime.controller_context_manager()

    def queue_manager(self) -> MessageQueueManager:
        return self._runtime.queue_manager()

    def session_id(self) -> str:
        return self._runtime.session_id()

    def parent(self):
        return self._runtime


class Runtime(ABC):
    @abstractmethod
    def executable_id(self) -> str:
        pass

    @abstractmethod
    def trace_id(self) -> str:
        pass

    @abstractmethod
    def update_state(self, data: dict):
        pass

    @abstractmethod
    def get_state(self, key: Union[str, list, dict] = None) -> Any:
        pass

    @abstractmethod
    def update_global_state(self, data: dict):
        pass

    @abstractmethod
    def get_global_state(self, key: Union[str, list, dict] = None) -> Any:
        pass

    @abstractmethod
    def stream_writer(self) -> Optional[StreamWriter]:
        pass

    @abstractmethod
    def custom_writer(self) -> Optional[StreamWriter]:
        pass

    @abstractmethod
    async def write_stream(self, data: Union[dict, OutputSchema]):
        pass

    @abstractmethod
    async def write_custom_stream(self, data: dict):
        pass

    @abstractmethod
    async def trace(self, data: dict):
        pass

    @abstractmethod
    async def trace_error(self, error: Exception):
        pass

    @abstractmethod
    async def interaction(self, value):
        pass

    @abstractmethod
    def base(self) -> BaseRuntime:
        pass

class ProxyRuntime(BaseRuntime):
    def __init__(self, stub: BaseRuntime = None):
        self._stub = stub

    def set_runtime(self, stub: BaseRuntime):
        self._stub = stub

    def config(self) -> Config:
        return self._stub.config()

    def state(self) -> State:
        return self._stub.state()

    def tracer(self) -> Any:
        return self._stub.tracer()

    def stream_writer_manager(self) -> StreamWriterManager:
        return self._stub.stream_writer_manager()

    def callback_manager(self) -> CallbackManager:
        return self._stub.callback_manager()

    def controller_context_manager(self):
        return self._stub.controller_context_manager()

    def queue_manager(self) -> MessageQueueManager:
        return self._stub.queue_manager()

    def session_id(self) -> str:
        return self._stub.session_id()