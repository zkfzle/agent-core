# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Optional, Any

from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.foundation.llm import BaseModelClient
from openjiuwen.core.foundation.prompt import PromptTemplate
from openjiuwen.core.foundation.tool import Tool
from openjiuwen.core.session.interaction.interaction import WorkflowInteraction
from openjiuwen.core.session.internal.workflow import NodeSession
from openjiuwen.core.session.internal.wrapper import StateSession
from openjiuwen.core.session.tracer import TracerWorkflowUtils


class Session(StateSession):
    def __init__(self, session: NodeSession, stream_mode: bool = False):
        super().__init__(session)
        self._interaction = None
        self._stream_mode = stream_mode

    def workflow_id(self):
        return self._inner.workflow_id()

    def state(self):
        return self._inner.state()

    async def trace(self, data: dict):
        await TracerWorkflowUtils.trace(self._inner, data)


    async def trace_error(self, error: Exception):
        await TracerWorkflowUtils.trace_error(self._inner, error)


    async def interact(self, value):
        if self._stream_mode:
            raise JiuWenBaseException(
                StatusCode.WORKFLOW_STREAM_NOT_SUPPORT.code,
                StatusCode.WORKFLOW_STREAM_NOT_SUPPORT.errmsg,
            )
        if self._interaction is None:
            self._interaction = WorkflowInteraction(self._inner)
        return await self._interaction.wait_user_inputs(value)


    # todo: resource interface will be deleted when resource_mgr supports tag feature
    def get_prompt(self, template_id: str) -> PromptTemplate:
        return self._inner.resource_manager()._resource_registry.prompt().get_prompt(template_id)


    def get_model(self, model_id: str) -> BaseModelClient:
        return self._inner.resource_manager()._resource_registry.model().get_model(model_id)


    async def get_workflow(self, workflow_id: str) -> "Workflow":
        return await self._inner.resource_manager()._resource_registry.workflow().get_workflow(workflow_id)


    def get_workflow_sync(self, workflow_id: str) -> "Workflow":
        return self._inner.resource_manager()._resource_registry.workflow().get_workflow_sync(workflow_id)


    def get_tool(self, tool_id: str) -> Tool:
        return self._inner.resource_manager()._resource_registry.tool().get_tool(tool_id)


    def get_current_workflow_config(self):
        return self._inner.config().get_workflow_config(self._inner.workflow_id())


    def add_workflow_config(self, workflow_id, workflow_config):
        return self._inner.config().add_workflow_config(workflow_id, workflow_config)


    def get_workflow_config(self, workflow_id):
        return self._inner.config().get_workflow_config(workflow_id)


    def get_agent_config(self):
        return self._inner.config().get_agent_config()


    def get_env(self, key) -> Optional[Any]:
        return self._inner.config().get_env(key)