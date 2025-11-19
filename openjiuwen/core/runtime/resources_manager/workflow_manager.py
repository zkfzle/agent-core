#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import List, Tuple, TypeVar, Optional, Union, Callable

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runtime.resources_manager.abstract_manager import AbstractManager
from openjiuwen.core.tracer.decorator import decorate_workflow_with_trace
from openjiuwen.core.utils.tool.schema import ToolInfo

Workflow = TypeVar("Workflow", contravariant=True)


def generate_workflow_key(workflow_id: str, workflow_version: str) -> str:
    return f"{workflow_id}_{workflow_version}"

WorkflowProvider = Callable[[], Workflow]

class WorkflowMgr(AbstractManager[Workflow]):
    def __init__(self):
        super().__init__()
        self._workflow_tool_infos: dict[str, ToolInfo] = {}

    def add_workflow(self, workflow_id: str, workflow: Union[Workflow, WorkflowProvider]) -> None:
        self._validate_id(workflow_id, StatusCode.RUNTIME_WORKFLOW_ADD_FAILED, "workflow")
        self._validate_resource(workflow, StatusCode.RUNTIME_WORKFLOW_ADD_FAILED,
                                "workflow is invalid, can not be None")
        
        # Define validation function for non-callable workflows
        def validate_workflow(workflow_obj):
            self._workflow_tool_infos[workflow_id] = workflow_obj.get_tool_info()
            return workflow_obj
        
        self._add_resource(workflow_id, workflow, StatusCode.RUNTIME_WORKFLOW_ADD_FAILED, validate_workflow)

    def add_workflows(self, workflows: List[Tuple[str, Union[Workflow, WorkflowProvider]]]):
        if not workflows:
            return
        for key, workflow in workflows:
            self.add_workflow(key, workflow)

    def get_workflow(self, workflow_id: str, runtime=None) -> Workflow:
        # Validate ID using base class method
        self._validate_id(workflow_id, StatusCode.RUNTIME_WORKFLOW_GET_FAILED, "workflow")
        
        try:
            workflow = self.find_workflow_by_id_and_version(workflow_id)
            return decorate_workflow_with_trace(workflow, runtime)
        except JiuWenBaseException:
            raise
        except Exception as e:
            self._handle_exception(e, StatusCode.RUNTIME_WORKFLOW_GET_FAILED, "get")

    def find_workflow_by_id_and_version(self, workflow_id: str):
        # Validate ID using base class method
        self._validate_id(workflow_id, StatusCode.RUNTIME_WORKFLOW_GET_FAILED, "workflow")
        
        # Define function to create workflow from provider
        def create_workflow_from_provider(provider):
            workflow = provider()
            if not hasattr(workflow, "get_tool_info"):
                raise TypeError(f"Workflow must have get_tool_info method")
            self._workflow_tool_infos[workflow_id] = workflow.get_tool_info()
            return workflow
        
        return self._get_resource(workflow_id, StatusCode.RUNTIME_WORKFLOW_GET_FAILED, create_workflow_from_provider)

    def remove_workflow(self, workflow_id: str) -> Optional[Workflow]:
        self._validate_id(workflow_id, StatusCode.RUNTIME_WORKFLOW_REMOVE_FAILED, "workflow")
        
        try:
            workflow = self._remove_resource(workflow_id, StatusCode.RUNTIME_WORKFLOW_REMOVE_FAILED)
            self._workflow_tool_infos.pop(workflow_id, None)
            return workflow
        except Exception as e:
            self._handle_exception(e, StatusCode.RUNTIME_WORKFLOW_REMOVE_FAILED, "remove")

    def get_tool_infos(self, workflow_ids: List[str]):
        try:
            if not workflow_ids:
                return [info for info in self._workflow_tool_infos.values()]
            
            infos = []
            for workflow_id in workflow_ids:
                self._validate_id(workflow_id, StatusCode.RUNTIME_WORKFLOW_TOOL_INFO_GET_FAILED, "workflow")
                infos.append(self._workflow_tool_infos.get(workflow_id))
            return infos
        except JiuWenBaseException:
            raise
        except Exception as e:
            self._handle_exception(e, StatusCode.RUNTIME_WORKFLOW_TOOL_INFO_GET_FAILED, "get_tool_info")