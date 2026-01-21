# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import uuid

from openjiuwen.core.session import Config
from openjiuwen.core.session.internal.wrapper import TaskSession
from openjiuwen.core.session.workflow import Session as WorkflowSession


class Session:
    def __init__(self, session_id: str = None, config: Config = None):
        if session_id is None:
            session_id = str(uuid.uuid4())
        self._session_id = session_id
        self._inner = TaskSession(session_id=session_id, config=config)

    def get_session_id(self) -> str:
        return self._session_id

    def get_envs(self):
        return self._inner.get_envs()

    def get_agent_id(self):
        return self._inner.get_agent_config().id

    def get_agent_version(self):
        return self._inner.get_agent_config().version

    def get_agent_description(self):
        return self._inner.get_agent_config().description

    def create_workflow_session(self) -> WorkflowSession:
        return WorkflowSession(parent=self, session_id=self.get_session_id())

    async def interact(self, value):
        await self._inner.interact(value)


def create_agent_session(session_id: str = None, config: Config = None) -> Session:
    return Session(session_id=session_id, config=config)