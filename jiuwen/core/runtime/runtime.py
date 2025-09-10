#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import uuid
from abc import ABC, abstractmethod
from typing import Any, Union, Optional

from jiuwen.core.context.model_context.model_context import ModelContext, WorkflowModelContext, NodeModelContext
from jiuwen.core.runtime.callback_manager import CallbackManager
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.mq_manager import MessageQueueManager
from jiuwen.core.runtime.state import State, InMemoryState
from jiuwen.core.runtime.store import Store
from jiuwen.core.stream.manager import StreamWriterManager
from jiuwen.core.stream.writer import OutputSchema, StreamWriter
from jiuwen.core.tracer.handler import TracerHandlerName
from jiuwen.core.tracer.tracer import Tracer


class BaseRuntime(ABC):
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


class WorkflowRuntime(BaseRuntime):
    def __init__(self, state: State = None, config: Config = None, store: Store = None, tracer: Tracer = None,
                 session_id: str = None, parent_model_context: ModelContext = None,
                 controller_context_manager: Any = None):
        self._config = config if config is not None else Config()
        self._model_context = WorkflowModelContext(session_id)
        self._model_context.derive_from(parent_model_context)
        self._state = state if state is not None else InMemoryState()
        self._store = store
        self._tracer = tracer
        self._callback_manager = CallbackManager()
        self._stream_writer_manager = None  # type: StreamWriterManager
        self._controller_context_manager = controller_context_manager
        self._session_id = session_id if session_id else uuid.uuid4().hex
        self._queue_manager = None  # type: MessageQueueManager

    def context(self) -> ModelContext:
        return self._model_context

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

    def store(self) -> Store:
        return self._store

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
    def __init__(self, context: BaseRuntime, node_id: str):
        self._node_id = node_id
        self._parent_id = context.executable_id() if isinstance(context, NodeRuntime) else ''
        self._executable_id = self._parent_id + "." + node_id if len(self._parent_id) != 0 else node_id
        self._state = context.state().create_node_state(self._executable_id, self._parent_id)
        self._context = context
        self._model_context = NodeModelContext(self._node_id, self.session_id(),
                                               config=context.context().get_config(),
                                               parent_context=context.context())

    def context(self) -> ModelContext:
        return self._model_context

    def node_id(self):
        return self._node_id

    def executable_id(self):
        return self._executable_id

    def parent_id(self):
        return self._parent_id

    def tracer(self) -> Tracer:
        return self._context.tracer()

    def state(self) -> State:
        return self._state

    def config(self) -> Config:
        return self._context.config()

    def store(self) -> Store:
        return self._context.store()

    def stream_writer_manager(self) -> StreamWriterManager:
        return self._context.stream_writer_manager()

    def callback_manager(self) -> CallbackManager:
        return self._context.callback_manager()

    def controller_context_manager(self):
        return self._context.controller_context_manager()

    def queue_manager(self) -> MessageQueueManager:
        return self._context.queue_manager()

    def session_id(self) -> str:
        return self._context.session_id()

    def parent(self):
        return self._context


class Runtime:
    def __init__(self, runtime: NodeRuntime):
        self._inner = runtime

    def executable_id(self) -> str:
        return self._inner.executable_id()

    def trace_id(self) -> str:
        return self._inner.session_id()

    def update_state(self, data: dict):
        return self._inner.state().update(data)

    def get_state(self, key: Union[str, list, dict] = None) -> Any:
        return self._inner.state().get(key)

    def update_global_state(self, data: dict):
        return self._inner.state().update_global(data)

    def get_global_state(self, key: Union[str, list, dict] = None) -> Any:
        return self._inner.state().get_global(key)

    def stream_writer(self) -> Optional[StreamWriter]:
        manager = self._inner.stream_writer_manager()
        if manager:
            return manager.get_output_writer()
        return None

    def custom_writer(self) -> Optional[StreamWriter]:
        manager = self._inner.stream_writer_manager()
        if manager:
            return manager.get_custom_writer()
        return None

    async def write_stream(self, data: Union[dict, OutputSchema]):
        writer = self.stream_writer()
        if writer:
            await writer.write(data)

    async def write_custom_stream(self, data: dict):
        writer = self.custom_writer()
        if writer:
            await writer.write(data)

    async def trace(self, data: dict):
        tracer = self._inner.tracer()
        invoke_id = self._inner.executable_id()
        parent_id = self._inner.parent_id()
        await tracer.trigger(TracerHandlerName.TRACER_WORKFLOW.value, "on_invoke",
                             invoke_id=invoke_id,
                             parent_node_id=parent_id,
                             on_invoke_data=data)
        self._inner.state().update_trace(tracer.get_workflow_span(invoke_id, parent_id))

    async def trace_error(self, error: Exception):
        tracer = self._inner.tracer()
        invoke_id = self._inner.executable_id()
        parent_id = self._inner.parent_id()
        await self._inner.tracer().trigger(TracerHandlerName.TRACER_WORKFLOW.value, "on_invoke",
                                           invoke_id=invoke_id,
                                           parent_node_id=parent_id,
                                           error=error)
        self._inner.state().update_trace(tracer.get_workflow_span(invoke_id, parent_id))

    def base(self) -> NodeRuntime:
        return self._inner
