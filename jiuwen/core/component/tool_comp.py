#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.component.base import ComponentConfig, WorkflowComponent
from jiuwen.core.context_engine.base import Context
from jiuwen.core.graph.executable import Executable, Input, Output
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.utils.tool import constant
from jiuwen.core.utils.tool.base import Tool


@dataclass
class ToolComponentConfig(ComponentConfig):
    pass


class ToolComponentInput(BaseModel):
    model_config = ConfigDict(extra='allow')   # 允许任意额外字段


class ToolComponentOutput(BaseModel):
    error_code: int = Field(default=-1, alias=constant.ERR_CODE)
    error_message: str = Field(default="", alias=constant.ERR_MESSAGE)
    data: str = Field(default="", alias=constant.RESTFUL_DATA)


class ToolExecutable(ComponentExecutable):

    def __init__(self, config: ToolComponentConfig):
        super().__init__()
        self._config = config
        self._tool: Tool = None

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        if self._tool is None:
            self._tool = self.get_tool(runtime)
        tool_inputs = ToolComponentInput(**inputs).model_dump()
        formatted_inputs = prepare_inputs(tool_inputs, self.get_tool_param())
        try:
            response = self._tool.invoke(formatted_inputs)
            return self._create_output(response)
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.TOOL_COMPONENT_EXECUTE_ERROR.code,
                message='tool component execution error'
            ) from e

    def _create_output(self, response):
        return ToolComponentOutput(**response).model_dump()

    def get_tool(self, runtime: Runtime) -> Tool:
        pass

    def set_tool(self, tool: Tool):
        self._tool = tool
        return self

    def get_tool_param(self):
        return self._tool.params

    def validate_require_params(self, user_field):
        require_params = self.get_tool_param()
        params_dict = {param.name: param.descrpition for param in require_params}
        missing_params = {param for param in params_dict if param not in user_field}
        if missing_params:
            missing_params_dict = {param: params_dict[param] for param in missing_params}
            interrupt_message = {
                'type': 'MessageSubTypes.PLUGIN_PARAM_MISS.value',
                'tool_name': self._tool.name,
                'missing_params': missing_params_dict,
            }
            self.interrupt(interrupt_message)


TYPE_CASTER = {
    "str": str,
    "integer": int,
    "number": float,
    "bool": bool
}


def _transform_type(value, expected_type, key):
    expected_type = expected_type.lower()
    caster = TYPE_CASTER.get(expected_type)
    if caster:
        try:
            return caster(value)
        except(TypeError, ValueError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.TOOL_COMPONENT_PARAM_CHECK_ERROR.code,
                message=f'{StatusCode.TOOL_COMPONENT_PARAM_CHECK_ERROR.errmsg}'
                        f'param name is {key}, expected type: {expected_type}'
            ) from e
    return value


def prepare_inputs(user_field, defined_param) -> dict:
    define_dict = {}
    formatted_inputs = {}
    for param in defined_param:
        define_dict[param.name] = param
    for k, v in user_field.items():
        if define_dict.get(k):
            param = define_dict.get(k)
            expected_type = param.type
            formatted_inputs[k] = _transform_type(v, expected_type, k)
        else:
            raise JiuWenBaseException(
                error_code=StatusCode.TOOL_COMPONENT_INPUTS_ERROR.code,
                message=f'{StatusCode.TOOL_COMPONENT_INPUTS_ERROR.errmsg}, param is {k}'
            )
    return formatted_inputs

class ToolComponent(WorkflowComponent):

    def __init__(self, config: ToolComponentConfig):
        super().__init__()
        self._config = config
        self._tool = None

    def to_executable(self) -> Executable:
        return ToolExecutable(self._config).set_tool(self._tool)

    def bind_tool(self, tool: Tool):
        self._tool = tool
        return self
