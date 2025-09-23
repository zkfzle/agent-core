from typing import Any

from jiuwen.core.context_engine.base import Context
from jiuwen.core.runtime.agent_state import StateCollection
from jiuwen.core.runtime.callback_manager import CallbackManager
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.interaction.base import Checkpointer
from jiuwen.core.runtime.interaction.checkpointer import default_inmemory_checkpointer
from jiuwen.core.runtime.resource_manager import ResourceMgr
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.runtime.state import State, InMemoryCommitState
from jiuwen.core.runtime.workflow import WorkflowRuntime
from jiuwen.core.runtime.workflow_state import InMemoryState
from jiuwen.core.stream.emitter import StreamEmitter
from jiuwen.core.stream.manager import StreamWriterManager
from jiuwen.core.tracer.tracer import Tracer


class AgentRuntime(BaseRuntime):
    def __init__(self, trace_id: str, context: Context = None):
        self._trace_id = trace_id
        self._state = StateCollection()
        self._stream_writer_manager = StreamWriterManager(StreamEmitter())
        self._callback_manager = CallbackManager()
        tracer = Tracer()
        tracer.init(self._stream_writer_manager, self._callback_manager)
        self._tracer = tracer
        self._context = context
        self._checkpointer = default_inmemory_checkpointer
        self._resource_manager = ResourceMgr()
        self._config = Config()

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
        pass

    def session_id(self) -> str:
        return self._trace_id

    def context(self) -> Context:
        return self._context

    def resource_manager(self):
        return self._resource_manager

    def checkpointer(self) -> Checkpointer:
        return self._checkpointer

    def create_workflow_runtime(self) -> WorkflowRuntime:
        state = self._state.global_state
        return WorkflowRuntime(
            config=self._config,
            state=InMemoryState(InMemoryCommitState(state)),
            tracer=self._tracer,
            context=self._context,
            session_id=self._trace_id)
