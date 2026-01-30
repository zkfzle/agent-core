import os.path
import pickle
from typing import Union

from openjiuwen.core.runtime.agent import StaticAgentRuntime
from openjiuwen.core.runtime.resources_manager.resource_manager import ResourceMgr
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.runtime.wrapper import (
    StaticWrappedRuntime,
    TaskRuntime,
    WrappedRuntime
)
from openjiuwen.core.stream.base import OutputSchema


class AgentRuntime(WrappedRuntime, StaticWrappedRuntime):
    """
    deprecated
    """

    def __init__(self, config, working_dir=None, resource_mgr: ResourceMgr = None):
        inner = StaticAgentRuntime(config, resource_mgr=resource_mgr)
        super().__init__(inner)
        self._runtime = inner
        self.state = {}
        self.working_dir = working_dir
        if hasattr(config, "get"):
            agent_name = config.get("name")
            session_id = config.get("session_id")
        else:
            agent_name = config.name
            session_id = config.session_id
        if working_dir:
            if not os.path.exists(os.path.join(self.working_dir, ".agents")):
                os.makedirs(os.path.join(self.working_dir, ".agents"))
            self.checkpoint_path = os.path.join(self.working_dir, ".agents", f"{agent_name}_{session_id}.pkl")

    async def write_stream(self, data: Union[dict, OutputSchema]):
        return await self.write_custom_stream(data)

    async def pre_run(self, **kwargs) -> Runtime:
        session_id = kwargs.get("session_id")
        if session_id is None:
            session_id = kwargs.get("trace_id")
        inputs = kwargs.get("inputs")
        inner = await self._runtime.create_agent_runtime(session_id, inputs)
        return TaskRuntime(inner=inner)

    def resource_mgr(self):
        return self._inner.resource_manager()

    async def release(self, session_id: str):
        await self._runtime.checkpointer().release(session_id)

    def update_state(self, data):
        self.state.update(data)
        if self.working_dir is not None:
            with open(self.checkpoint_path, "wb") as f:
                f.write(pickle.dumps(self.state))

    def get_state(self, key: Union[str, list, dict] = None):
        return self.state.get(key)

    def checkpoint_exists(self):
        return os.path.exists(self.checkpoint_path)

    def load_from_checkpoint(self):
        if self.working_dir is not None:
            with open(self.checkpoint_path, "rb") as f:
                runtime_state = pickle.load(f)
            self.update_state(runtime_state)
        else:
            raise RuntimeError(f"Checkpoint {self.checkpoint_path} does not exist")