#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import Any, List, Tuple, Union, Optional

from jiuwen.core.runtime.agent import AgentRuntime as Inner
from jiuwen.core.runtime.interaction.interaction import AgentInteraction
from jiuwen.core.runtime.runtime import BaseRuntime, Workflow
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.runtime.workflow import WorkflowRuntime
from jiuwen.core.stream.writer import OutputSchema, StreamWriter
from jiuwen.core.tracer.tracer import Tracer
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import FunctionInfo
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.tool.base import Tool


class AgentRuntime(Runtime):
    def __init__(self, trace_id: str):
        self._inner = Inner(trace_id)
        self._interaction = None

    async def initialize(self):
        await self._inner.checkpointer().pre_agent_execute(self._inner)


    def get_tool(self, tool_id: str) -> Tool:
        pass

    def executable_id(self) -> str:
        return ""

    def trace_id(self) -> str:
        return self._inner.session_id()

    def update_state(self, data: dict):
        self._inner.state().update(data)

    def get_state(self, key: Union[str, list, dict] = None) -> Any:
        return self._inner.state().get(key)

    def update_global_state(self, data: dict):
        self._inner.state().update_global(data)

    def get_global_state(self, key: Union[str, list, dict] = None) -> Any:
        return self._inner.state().get_global(key)

    def stream_writer(self) -> Optional[StreamWriter]:
        manager = self._inner.stream_writer_manager()
        if manager:
            return manager.get_custom_writer()
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
        pass

    async def trace_error(self, error: Exception):
        pass

    async def interact(self, value):
        if self._interaction is None:
            self._interaction = AgentInteraction(self._inner)
        return await self._interaction.wait_user_inputs(value)

    def add_prompt(self, template_id: str, template: Template):
        self._inner.resource_manager().add_prompt(template_id, template)

    def add_prompts(self, templates: List[Tuple[str, Template]]):
        self._inner.resource_manager().add_prompts(templates)

    def remove_prompt(self, template_id: str):
        self._inner.resource_manager().remove_prompt(template_id)

    def get_prompt(self, template_id: str) -> Template:
        return self._inner.resource_manager().get_prompt(template_id)

    def add_model(self, model_id: str, model: BaseChatModel):
        self._inner.resource_manager().add_model(model_id, model)

    def add_models(self, models: List[Tuple[str, BaseChatModel]]):
        self._inner.resource_manager().add_models(models)

    def remove_model(self, model_id: str):
        self._inner.resource_manager().remove_model(model_id)

    def get_model(self, model_id: str) -> BaseChatModel:
        return self._inner.resource_manager().get_model(model_id)

    def add_workflow(self, workflow_id: str, workflow: Workflow):
        self._inner.resource_manager().add_workflow(workflow_id, workflow)

    def add_workflows(self, workflows: List[Tuple[str, Workflow]]):
        self._inner.resource_manager().add_workflows(workflows)

    def remove_workflow(self, workflow_id: str):
        self._inner.resource_manager().remove_workflow(workflow_id)

    def get_workflow(self, workflow_id: str) -> Workflow:
        return self._inner.resource_manager().get_workflow(workflow_id)

    def add_tool(self, tool_id: str, tool: Tool):
        self._inner.resource_manager().add_tool(tool_id, tool)

    def add_tools(self, tools: List[Tuple[str, Tool]]):
        self._inner.resource_manager().add_tools(tools)

    def remove_tool(self, tool_id: str):
        self._inner.resource_manager().remove_tool(tool_id)

    def get_function_info(self, tool_id: List[str], workflow_id: List[str]) -> List[FunctionInfo]:
        pass

    def base(self) -> BaseRuntime:
        return self._inner

    async def close(self):
        await self._inner.checkpointer().post_agent_execute(self.trace_id())
        await self._inner.stream_writer_manager().stream_emitter().close()

    def set_controller_context_manager(self, controller_context_manager: Any):
        self._controller_context_manager = controller_context_manager

    def controller_context_manager(self) -> Any:
        return self._controller_context_manager

    def tracer(self) -> Tracer:
        return self._inner.tracer()

    def create_workflow_runtime(self) -> WorkflowRuntime:
        return self._inner.create_workflow_runtime()
