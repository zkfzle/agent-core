# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import uuid

from openjiuwen.core.session import Config
from openjiuwen.core.session.internal.wrapper import TaskSession


class Session:
    """AgentGroup Session"""

    def __init__(self, session_id: str = None, config: Config = None):
        if session_id is None:
            session_id = str(uuid.uuid4())
        self._session_id = session_id
        self._inner = TaskSession(session_id=session_id, config=config)

    def get_session_id(self) -> str:
        return self._session_id

    def get_envs(self):
        return self._inner.get_envs()


def create_agent_group_session(session_id: str = None, config: Config = None) -> Session:
    """Create AgentGroup Session"""
    return Session(session_id=session_id, config=config)