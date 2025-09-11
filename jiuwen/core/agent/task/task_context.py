#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import Any

from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.runtime import WorkflowRuntime
from jiuwen.core.runtime.state import InMemoryState, StateLike, InMemoryCommitState, \
    InMemoryStateLike
from jiuwen.core.runtime.store import Store
from jiuwen.core.runtime.callback_manager import CallbackManager
from jiuwen.core.stream.base import BaseStreamMode
from jiuwen.core.stream.emitter import StreamEmitter
from jiuwen.core.stream.manager import StreamWriterManager
from jiuwen.core.tracer.tracer import Tracer


class TaskContext:
    def __init__(self, id: str, store: Store = None, controller_context_manager: Any = None):
        self.__id = id
        self.__global_state = InMemoryStateLike()
        self.__store = store
        self.__controller_context_manager = controller_context_manager
        self.__stream_writer_manager: StreamWriterManager = StreamWriterManager(StreamEmitter(),[BaseStreamMode.TRACE])
        self.__callback_manager = CallbackManager()
        self.__tracer = Tracer()
        self.__tracer.init(self.__stream_writer_manager, self.__callback_manager)

    def state(self) -> StateLike:
        return self.__global_state

    def set_controller_context_manager(self, controller_context_manager: Any):
        self.__controller_context_manager = controller_context_manager

    def controller_context_manager(self) -> Any:
        return self.__controller_context_manager

    def tracer(self) -> Tracer:
        return self.__tracer

    def create_workflow_context(self) -> WorkflowRuntime:
        return WorkflowRuntime(
            state=InMemoryState(InMemoryCommitState(self.__global_state)),
            store=self.__store,
            tracer=self.__tracer,
            config=Config(),
            session_id=self.__id,
            controller_context_manager=self.__controller_context_manager)

    def stream_writer_manager(self) -> StreamWriterManager:
        return self.__stream_writer_manager
