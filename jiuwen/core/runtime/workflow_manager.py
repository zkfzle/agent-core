#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import List, Tuple, TypeVar, Optional, Union

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.runtime.thread_safe_dict import ThreadSafeDict
from jiuwen.core.tracer.decorator import decrate_workflow_with_trace
from jiuwen.core.utils.llm.messages import ToolInfo
from jiuwen.core.workflow.workflow_config import WorkflowInputsSchema

Workflow = TypeVar("Workflow", contravariant=True)


def generate_workflow_key(workflow_id: str, workflow_version: str) -> str:
    return f"{workflow_id}_{workflow_version}"

WorkflowProvider = lambda: Workflow

class WorkflowMgr:
    def __init__(self):
        self._workflows: ThreadSafeDict[str, Workflow] = ThreadSafeDict()
        self._workflow_providers: ThreadSafeDict[str, WorkflowProvider] = ThreadSafeDict()
        self._workflow_tool_infos: ThreadSafeDict[str, ToolInfo] = ThreadSafeDict()
        self._workflow_schema: ThreadSafeDict[str, WorkflowInputsSchema] = ThreadSafeDict()

    def add_workflow(self, workflow_id: str, workflow: Union[Workflow, WorkflowProvider]) -> None:
        if workflow_id is None or workflow_id.strip() == "":
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_ADD_FAILED.code,
                                      StatusCode.RUNTIME_WORKFLOW_ADD_FAILED.errmsg.format(
                                          reason="workflow_id is invalid, can not be None or empty"))
        if workflow is None:
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_ADD_FAILED.code,
                                      StatusCode.RUNTIME_WORKFLOW_ADD_FAILED.errmsg.format(
                                          reason="workflow is invalid, can not be None"))
        if callable(workflow):
            self._workflow_providers[workflow_id] = workflow
        else:
            self._workflows[workflow_id] = workflow
            self._workflow_tool_infos[workflow_id] = workflow.get_tool_info()

    def add_workflows(self, workflows: List[Tuple[str, Union[Workflow, WorkflowProvider]]]):
        if not workflows:
            return
        for key, workflow in workflows:
            self.add_workflow(key, workflow)

    def get_workflow(self, workflow_id: str, runtime=None) -> Workflow:
        if workflow_id is None or workflow_id.strip() == "":
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_GET_FAILED.code,
                                      StatusCode.RUNTIME_WORKFLOW_GET_FAILED.errmsg.format(
                                          reason="workflow_id is invalid, can not be None or empty"))
        workflow = self.find_workflow_by_id_and_version(workflow_id)
        if not workflow or not runtime or not runtime.tracer():
            return workflow
        return decrate_workflow_with_trace(WrappedWorkflow(workflow), runtime)

    def find_workflow_by_id_and_version(self, workflow_id: str):
        if workflow_id is None:
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_GET_FAILED.code,
                                      StatusCode.RUNTIME_WORKFLOW_GET_FAILED.errmsg.format(
                                          reason="workflow_id is invalid, can not be None"))
        workflow = self._workflows.get(workflow_id)
        if workflow:
            return workflow
        provider = self._workflow_providers.get(workflow_id)
        if provider:
            try:
                workflow = provider()
                # 添加类型验证
                if not hasattr(workflow, "get_tool_info"):
                    raise TypeError(f"Workflow must have get_tool_info method")
                self._workflows[workflow_id] = workflow
                self._workflow_tool_infos[workflow_id] = workflow.get_tool_info()
                return workflow
            except Exception as e:
                raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_GET_FAILED.code,
                                          StatusCode.RUNTIME_WORKFLOW_GET_FAILED.errmsg.format(
                                              reason=str(e)))
        return None

    def remove_workflow(self, workflow_id: str) -> Optional[Workflow]:
        if workflow_id is None or workflow_id.strip() == "":
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_REMOVE_FAILED.code,
                                      StatusCode.RUNTIME_WORKFLOW_REMOVE_FAILED.errmsg.format(
                                          reason="workflow_id is invalid, can not be empty"))
        try:
            workflow = self._workflows.pop(workflow_id, None)
            self._workflow_providers.pop(workflow_id, None)
            self._workflow_tool_infos.pop(workflow_id, None)
            return workflow
        except Exception as e:
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_REMOVE_FAILED.code,
                                      StatusCode.RUNTIME_WORKFLOW_REMOVE_FAILED.errmsg.format(
                                          reason=str(e)))

    def get_tool_infos(self, workflow_ids: List[str]):
        try:
            if not workflow_ids:
                return [info for info in self._workflow_tool_infos.values()]
            infos = []
            for workflow_id in workflow_ids:
                if workflow_id is None or workflow_id.strip() == "":
                    raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_TOOL_INFO_GET_FAILED.code,
                                              StatusCode.RUNTIME_WORKFLOW_TOOL_INFO_GET_FAILED.errmsg.format(
                                                  reason="workflow_id is invalid, can not be None or empty"))
                infos.append(self._workflow_tool_infos.get(workflow_id))
            return infos
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_TOOL_INFO_GET_FAILED.code,
                                      StatusCode.RUNTIME_WORKFLOW_TOOL_INFO_GET_FAILED.errmsg.format(
                                          reason=f"Failed to get workflow tool infos: {str(e)}"))



class WrappedWorkflow:
    def __init__(self, workflow):
        self.inner = workflow

    async def invoke(self, inputs, runtime, context=None):
        return await self.inner.invoke(inputs, runtime, context)

    async def stream(self, inputs, runtime, context=None, stream_modes=None):
        result = self.inner.stream(inputs, runtime, context, stream_modes)
        async for item in result:
            yield item

    def get_tool_info(self):
        return self.inner.get_tool_info()

    async def sub_invoke(self, inputs, runtime, config=None):
        return await self.inner.sub_invoke(inputs, runtime, config)

    def get_workflow_metadata(self):
        metadata = self.inner.config().metadata if self.inner.config() else {}
        return dict(metadata)
