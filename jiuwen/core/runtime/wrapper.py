from typing import Union, Any, Optional, List, Tuple

from jiuwen.core.graph.interrupt.interaction import Interaction
from jiuwen.core.runtime.runtime import Runtime, NodeRuntime, Workflow
from jiuwen.core.stream.writer import StreamWriter, OutputSchema
from jiuwen.core.tracer.workflow_tracer import trace, trace_error
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import FunctionInfo
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.tool.base import Tool


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

    async def interact(self, value):
        if self._interaction is None:
            self._interaction = Interaction(self._inner)
        return await self._interaction.wait_user_inputs(value)

    def add_prompt(self, template_id: str, template: Template):
        pass

    def add_prompts(self, templates: List[Tuple[str, Template]]):
        pass

    def remove_prompt(self, template_id: str):
        pass

    def get_prompt(self, template_id: str) -> Template:
        pass

    def add_model(self, model_id: str, model: BaseChatModel):
        pass

    def add_models(self, models: List[Tuple[str, BaseChatModel]]):
        pass

    def remove_model(self, model_id: str):
        pass

    def get_model(self, model_id: str) -> BaseChatModel:
        pass

    def add_workflow(self, workflow_id: str, workflow: Workflow):
        pass

    def add_workflows(self, workflows: List[Tuple[str, Workflow]]):
        pass

    def remove_workflow(self, workflow_id: str):
        pass

    def get_workflow(self, workflow_id: str) -> Workflow:
        pass

    def add_tool(self, tool_id: str, tool: Tool):
        pass

    def add_tools(self, tools: List[Tuple[str, Tool]]):
        pass

    def remove_tool(self, tool_id: str):
        pass

    def get_tool(self, tool_id: str) -> Tool:
        pass

    def get_function_info(self, tool_id: List[str], workflow_id: List[str]) -> List[FunctionInfo]:
        pass

    def base(self) -> NodeRuntime:
        return self._inner

    async def close(self):
        pass
