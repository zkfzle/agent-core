from typing import Union, Any, Optional

from jiuwen.core.graph.interrupt.interaction import Interaction
from jiuwen.core.runtime.runtime import Runtime, NodeRuntime
from jiuwen.core.stream.writer import StreamWriter, OutputSchema
from jiuwen.core.tracer.workflow_tracer import trace, trace_error


class WrappedNodeRuntime(Runtime):
    def __init__(self, runtime: NodeRuntime):
        self._inner = runtime
        self._interaction = None

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
        await trace(self._inner, data)

    async def trace_error(self, error: Exception):
        await trace_error(self._inner, error)

    async def interaction(self, value):
        if self._interaction is None:
            self._interaction = Interaction(self._inner)
        return await self._interaction.wait_user_inputs(value)

    def base(self) -> NodeRuntime:
        return self._inner
