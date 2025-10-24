from langgraph._internal._constants import INTERRUPT

from jiuwen.core.common.constants.constant import INTERACTIVE_INPUT
from jiuwen.core.runtime.interaction.agent_storage import AgentStorage
from jiuwen.core.runtime.interaction.base import Checkpointer
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.runtime.interaction.workflow_storage import WorkflowStorage
from jiuwen.core.runtime.runtime import BaseRuntime


class InMemoryCheckpointer(Checkpointer):
    def __init__(self):
        self._agent_store = AgentStorage()
        self._workflow_store = WorkflowStorage()

    async def pre_workflow_execute(self, runtime: BaseRuntime, inputs: InteractiveInput):
        self._workflow_store.recover(runtime, inputs)

    async def post_workflow_execute(self, runtime: BaseRuntime, result, exception):
        session_id = runtime.session_id()
        if exception is not None:
            self._workflow_store.save(runtime)
            raise exception

        if result.get(INTERRUPT) is None:
            await self._workflow_store.graph_checkpointer().adelete_thread(session_id)
            self._workflow_store.clear(session_id)
        else:
            self._workflow_store.save(runtime)

    async def pre_agent_execute(self, runtime: BaseRuntime, inputs):
        self._agent_store.recover(runtime)
        if inputs is not None:
            runtime.state().set_state({INTERACTIVE_INPUT: [inputs]})

    async def interrupt_agent_execute(self, runtime: BaseRuntime):
        self._agent_store.save(runtime)

    async def post_agent_execute(self, runtime: BaseRuntime):
        self._agent_store.save(runtime)

    async def release(self, session_id: str):
        await self._workflow_store.graph_checkpointer().adelete_thread(session_id)
        self._workflow_store.clear(session_id)
        self._agent_store.clear(session_id)

    def graph_checkpointer(self):
        return self._workflow_store.graph_checkpointer()


default_inmemory_checkpointer: Checkpointer = InMemoryCheckpointer()