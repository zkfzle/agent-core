#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""Handler of Agent"""
from typing import Dict, Callable, Any, Awaitable, Union

from pydantic import BaseModel, Field

from jiuwen.agent.common.enum import SubTaskType
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.graph.interrupt.interactive_input import InteractiveInput


class AgentHandlerInputs(BaseModel):
    query: str = Field(default="")
    name: str = Field(default="")
    arguments: Union[dict, InteractiveInput] = Field(default_factory=dict)
    context: Any = Field(default=None)


class AgentHandler:
    def __init__(self, agent_config: AgentConfig):
        self._function_map: Dict[SubTaskType, Callable[[AgentHandlerInputs], Awaitable[dict]]] = {
            SubTaskType.WORKFLOW: self.invoke_workflow,
            SubTaskType.PLUGIN: self.invoke_plugin
        }
        self._config = agent_config

    async def invoke(self, sub_task_type: SubTaskType, inputs: AgentHandlerInputs):
        handler = self._function_map.get(sub_task_type)
        if not handler:
            raise JiuWenBaseException()
        return await handler(inputs)

    async def invoke_workflow(self, inputs: AgentHandlerInputs):
        return dict()

    async def invoke_plugin(self, inputs: AgentHandlerInputs):
        return dict()

    async def invoke_llm(self, inputs: AgentHandlerInputs):
        return dict()

    async def send_message(self, inputs: AgentHandlerInputs):
        return dict()

    def search_workflow_metadata_by_workflow_name(self, workflow_name: str) -> WorkflowSchema:
        pass


class AgentHandlerImpl(AgentHandler):
    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)

    async def invoke(self, sub_task_type: SubTaskType, inputs: AgentHandlerInputs):
        handler = self._function_map.get(sub_task_type)
        if not handler:
            raise JiuWenBaseException()
        return await handler(inputs)

    async def invoke_workflow(self, inputs: AgentHandlerInputs):
        context = inputs.context
        workflow_name = inputs.name
        context_manager = context.controller_context_manager()
        workflow_manager = context_manager.workflow_mgr
        workflow_metadata = self.search_workflow_metadata_by_workflow_name(workflow_name)
        workflow = workflow_manager.find_workflow_by_id_and_version(workflow_metadata.id, workflow_metadata.version)
        workflow_result = await workflow.invoke(inputs.arguments, context.create_workflow_runtime())
        return workflow_result.result

    async def invoke_plugin(self, inputs: AgentHandlerInputs):
        context = inputs.context
        plugin_name = inputs.name
        plugin_args = inputs.arguments

        context_manager = context.controller_context_manager()
        workflow_manager = context_manager.workflow_mgr
        plugin = workflow_manager.find_tool_by_name(plugin_name)
        plugin_result = plugin.invoke(plugin_args)
        return plugin_result

    def search_workflow_metadata_by_workflow_name(self, workflow_name: str) -> WorkflowSchema:
        workflows_config = self._config.workflows
        for item in workflows_config:
            if workflow_name == item.name:
                return item
        raise JiuWenBaseException()