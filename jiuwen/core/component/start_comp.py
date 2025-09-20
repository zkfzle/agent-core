#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from copy import deepcopy
from typing import TypedDict

from jiuwen.core.context_engine.base import Context
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.graph.executable import Input, Output


class Start(ComponentExecutable, WorkflowComponent):
    def __init__(self, conf: dict = None):
        super().__init__()
        self.conf = conf

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        self._validate_inputs(inputs)
        return self._fill_default_values(deepcopy(inputs))

    def _fill_default_values(self, inputs: Input):
        if not self.conf:
            return inputs
        defined_variables = self.conf.get("inputs", [])
        default_maps = {var["id"]: var["default_value"]
                        for var in defined_variables
                        if "id" in var and "default_value" in var and var["default_value"] is not None}
        return default_maps | inputs

    def _validate_inputs(self, inputs: Input):
        defined_variables = self.conf.get("inputs", {})
        variables_not_given = []
        for variable in [var for var in defined_variables if var.get("required", False)]:
            variable_name = variable.get("id", "")
            if variable_name not in inputs:
                variables_not_given.append(variable_name)
            if variables_not_given:
                raise JiuWenBaseException(error_code=StatusCode.WORKFLOW_START_MISSING_GLOBAL_VARIABLE_VALUE.code,
                                          message=StatusCode.WORKFLOW_START_MISSING_GLOBAL_VARIABLE_VALUE.errmsg.format(
                                              variable_names=variables_not_given))


class StartInputSchema(TypedDict):
    query: str
    dialogueHistory: list
    conversationHistory: list


class StartOutputSchema(TypedDict):
    query: str
    dialogueHistory: list
    conversationHistory: list
