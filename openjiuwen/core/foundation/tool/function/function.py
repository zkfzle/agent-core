#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import inspect
from typing import Callable, List, AsyncIterator

from openjiuwen.core.foundation.tool.schema import ToolInfo
from openjiuwen.core.foundation.tool.base import Tool
from openjiuwen.core.foundation.tool.constant import Input, Output
from openjiuwen.core.foundation.tool.param import Param
from openjiuwen.core.foundation.tool.param_util import ParamUtil
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode


class LocalFunction(Tool):

    def __init__(self, name: str, description: str, params: List[Param] = None, func: Callable = None):
        super().__init__()
        self.name = name
        self.description = description
        self.params = params
        self._func = func

    async def invoke(self, inputs: Input, **kwargs) -> Output:
        inputs = ParamUtil.format_input_with_default_when_required(self.params, inputs)
        if inspect.isgeneratorfunction(self._func) or inspect.isasyncgenfunction(self._func):
            raise JiuWenBaseException(
                error_code=StatusCode.PLUGIN_UNEXPECTED_ERROR.code, message="invoke function not support generator"
            )
        if inspect.iscoroutinefunction(self._func):
            res = await self._func(**inputs)
        else:
            res = self._func(**inputs)
        return res

    async def stream(self, inputs: Input, **kwargs) -> AsyncIterator[Output]:
        inputs = ParamUtil.format_input_with_default_when_required(self.params, inputs)
        if inspect.isasyncgenfunction(self._func):
            async for item in self._func(**inputs):
                yield item
        else:
            raise JiuWenBaseException(
                error_code=StatusCode.PLUGIN_UNEXPECTED_ERROR.code, message="stream function need aysnc generator"
            )

    def get_tool_info(self) -> ToolInfo:
        tool_info_dict = Param.format_functions(self)
        tool_info = ToolInfo(**tool_info_dict)
        return tool_info
