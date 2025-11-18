#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.


from typing import Optional

from openjiuwen.core.common.constants.constant import INTERACTIVE_INPUT
from openjiuwen.core.runtime.interaction.agent_storage import AgentStorage
from openjiuwen.core.runtime.interaction.base import Checkpointer
from openjiuwen.core.runtime.runtime import BaseRuntime


class AgentInMemoryCheckpointer(Checkpointer):
    def __init__(self):
        self._agent_stores: dict[str, AgentStorage] = {}

    async def pre_workflow_execute(self, runtime: BaseRuntime, inputs):
        pass

    async def post_workflow_execute(self, runtime: BaseRuntime, result, exception):
        pass

    async def pre_agent_execute(self, runtime: BaseRuntime, inputs):
        agent_store = self._agent_stores.setdefault(runtime.session_id(), AgentStorage())
        agent_store.recover(runtime)
        if inputs is not None:
            runtime.state().set_state({INTERACTIVE_INPUT: [inputs]})

    async def interrupt_agent_execute(self, runtime: BaseRuntime):
        agent_store = self._agent_stores.get(runtime.session_id())
        if agent_store is not None:
            agent_store.save(runtime)

    async def post_agent_execute(self, runtime: BaseRuntime):
        agent_store = self._agent_stores.get(runtime.session_id())
        if agent_store is not None:
            agent_store.save(runtime)

    async def release(self, session_id: str, agent_id: Optional[str] = None):
        agent_store = self._agent_stores.get(session_id)
        if agent_store is None:
            return
        if agent_id is not None:
            agent_store.clear(agent_id)
        else:
            self._agent_stores.pop(session_id, None)


default_agent_inmemory_checkpointer: Checkpointer = AgentInMemoryCheckpointer()
