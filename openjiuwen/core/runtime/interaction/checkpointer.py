#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from langgraph._internal._constants import INTERRUPT
from langgraph.checkpoint.memory import InMemorySaver

from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.core.common.constants.constant import INTERACTIVE_INPUT
from openjiuwen.core.runtime.interaction.agent_storage import AgentStorage
from openjiuwen.core.runtime.interaction.base import Checkpointer
from openjiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from openjiuwen.core.runtime.interaction.workflow_storage import WorkflowStorage
from openjiuwen.core.runtime.runtime import BaseRuntime


class InMemoryCheckpointer(Checkpointer):
    def __init__(self):
        self._agent_stores = {}
        self._workflow_stores = {}
        self._graph_state = InMemorySaver()

    async def pre_workflow_execute(self, runtime: BaseRuntime, inputs: InteractiveInput):
        self._workflow_stores.setdefault(runtime.session_id(), WorkflowStorage())
        if isinstance(inputs, InteractiveInput):
            workflow_store = self._workflow_stores.get(runtime.session_id())
            workflow_store.recover(runtime, inputs)


    async def post_workflow_execute(self, runtime: BaseRuntime, result, exception):
        workflow_store = self._workflow_stores.get(runtime.session_id())
        if exception is not None:
            workflow_store.save(runtime)
            raise exception

        if result.get(INTERRUPT) is None:
            await self._graph_state.adelete_thread(Checkpointer.get_thread_id(runtime))
            workflow_store.clear(runtime.workflow_id())
            if runtime.config().get_agent_config() is None:
                self._workflow_stores.pop(runtime.session_id(), None)
        else:
            workflow_store.save(runtime)

    async def pre_agent_execute(self, runtime: BaseRuntime, inputs):
        agent_store = self._agent_stores.setdefault(runtime.session_id(), AgentStorage())
        agent_store.recover(runtime)
        if inputs is not None:
            runtime.state().set_state({INTERACTIVE_INPUT: [inputs]})

    async def interrupt_agent_execute(self, runtime: BaseRuntime):
        agent_store = self._agent_stores.get(runtime.session_id())
        agent_store.save(runtime)

    async def post_agent_execute(self, runtime: BaseRuntime):
        agent_store = self._agent_stores.get(runtime.session_id())
        agent_store.save(runtime)

    async def release(self, session_id: str, agent_id: str = None):
        if agent_id is not None:
            agent_store = self._agent_stores.get(session_id)
            agent_store.clear(agent_id)
        else:
            self._workflow_stores.pop(session_id, None)
            self._agent_stores.pop(session_id, None)

    def graph_checkpointer(self):
        return self._graph_state


default_inmemory_checkpointer: Checkpointer = InMemoryCheckpointer()