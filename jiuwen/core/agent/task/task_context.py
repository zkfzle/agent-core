#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import Any, List, Tuple, Union, Optional

from jiuwen.core.runtime.runtime import AgentRuntime as Inner
from jiuwen.core.runtime.runtime import WorkflowRuntime, BaseRuntime, Workflow
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.stream.writer import OutputSchema, StreamWriter
from jiuwen.core.tracer.tracer import Tracer
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import FunctionInfo
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.tool.base import Tool


class AgentRuntime(Runtime):
    def __init__(self, trace_id: str):
        self._inner = Inner(trace_id)

    def get_tool(self, tool_id: str) -> Tool:
        pass

    def executable_id(self) -> str:
        pass

    def trace_id(self) -> str:
        pass

    def update_state(self, data: dict):
        pass

    def get_state(self, key: Union[str, list, dict] = None) -> Any:
        pass

    def update_global_state(self, data: dict):
        pass

    def get_global_state(self, key: Union[str, list, dict] = None) -> Any:
        pass

    def stream_writer(self) -> Optional[StreamWriter]:
        pass

    def custom_writer(self) -> Optional[StreamWriter]:
        pass

    async def write_stream(self, data: Union[dict, OutputSchema]):
        pass

    async def write_custom_stream(self, data: dict):
        pass

    async def trace(self, data: dict):
        pass

    async def trace_error(self, error: Exception):
        pass

    async def interact(self, value):
        pass

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

    def get_function_info(self, tool_id: List[str], workflow_id: List[str]) -> List[FunctionInfo]:
        pass

    def base(self) -> BaseRuntime:
        pass

    async def close(self):
        await self._inner.stream_writer_manager().stream_emitter().close()

    def set_controller_context_manager(self, controller_context_manager: Any):
        self._controller_context_manager = controller_context_manager

    def controller_context_manager(self) -> Any:
        return self._controller_context_manager

    def tracer(self) -> Tracer:
        return self._inner.tracer()

    def create_workflow_runtime(self) -> WorkflowRuntime:
        return self._inner.create_workflow_runtime()
