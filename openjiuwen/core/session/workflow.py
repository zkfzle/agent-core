# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Any, Union, Optional

from openjiuwen.core.session.callback.callback_manager import CallbackManager
from openjiuwen.core.session.config.base import Config
from openjiuwen.core.session.internal.wrapper import WrappedSession
from openjiuwen.core.session.internal.workflow import  WorkflowSession
from openjiuwen.core.session.state.base import State
from openjiuwen.core.session.stream import StreamWriterManager, StreamWriter, OutputSchema
from openjiuwen.core.session.tracer import Tracer


class Session(WrappedSession):
    """
    Session is the main class for managing the workflow of a session.
    It extends WrappedSession to provide additional functionality.
    """

    def __init__(self, workflow_id: str = '', parent: WrappedSession = None, session_id: str = None,
                 state: State = None, envs: dict[str, Any] = None):
        workflow_session = WorkflowSession(workflow_id=workflow_id,
                                           parent=parent.base() if parent is not None else None, session_id=session_id,
                                           state=state)
        super().__init__(workflow_session)
        self._inner.config().set_envs(envs)

    def set_stream_writer_manager(self, stream_writer_manager: StreamWriterManager) -> None:
        self._inner.set_stream_writer_manager(stream_writer_manager)

    def set_tracer(self, tracer: Tracer) -> None:
        self._inner.set_tracer(tracer)

    def set_actor_manager(self, queue_manager: "ActorManager"):
        self._inner.set_actor_manager(queue_manager)

    def set_workflow_id(self, workflow_id):
        self._inner.set_workflow_id(workflow_id)

    def actor_manager(self) -> "ActorManager":
        return self._inner.actor_manager()

    def config(self) -> Config:
        return self._inner.config()

    def state(self) -> State:
        return self._inner.state()

    def tracer(self) -> Any:
        return self._inner.tracer()

    def stream_writer_manager(self) -> StreamWriterManager:
        return self._inner.stream_writer_manager()

    def callback_manager(self) -> CallbackManager:
        return self._inner.callback_manager()

    def session_id(self) -> str:
        return self._inner.session_id()

    def resource_manager(self) -> "ResourceMgr":
        return self._inner.resource_manager()

    def checkpointer(self):
        return self._inner.checkpointer()

    def workflow_id(self):
        return self._inner.workflow_id()

    def main_workflow_id(self):
        return self._inner.main_workflow_id()

    def workflow_nesting_depth(self):
        return self._inner.workflow_nesting_depth()

    async def close(self):
        await self._inner.close()

    def executable_id(self) -> str:
        raise NotImplementedError

    def update_state(self, data: dict):
        raise NotImplementedError

    def get_state(self, key: Union[str, list, dict] = None) -> Any:
        raise NotImplementedError

    def update_global_state(self, data: dict):
        raise NotImplementedError

    def get_global_state(self, key: Union[str, list, dict] = None) -> Any:
        raise NotImplementedError

    def stream_writer(self) -> Optional[StreamWriter]:
        raise NotImplementedError

    def custom_writer(self) -> Optional[StreamWriter]:
        raise NotImplementedError

    async def write_stream(self, data: Union[dict, OutputSchema]):
        raise NotImplementedError

    async def write_custom_stream(self, data: dict):
        raise NotImplementedError

    async def trace(self, data: dict):
        raise NotImplementedError

    async def trace_error(self, error: Exception):
        raise NotImplementedError

    async def interact(self, value):
        raise NotImplementedError


def create_workflow_session(workflow_id: str = '', parent: WrappedSession = None, session_id: str = None,
                                  state: State = None, envs: dict[str, Any] = None) -> Session:
    return Session(workflow_id=workflow_id, parent=parent, session_id=session_id, state=state, envs=envs)

