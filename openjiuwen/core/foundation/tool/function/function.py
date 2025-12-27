#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import inspect
from typing import Callable, List, AsyncIterator

from openjiuwen.core.foundation.tool.schema import ToolInfo, ToolCard
from openjiuwen.core.foundation.tool.base import Tool
from openjiuwen.core.foundation.tool.constant import Input, Output
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode


class LocalFunction(Tool):

    def __init__(self, card: ToolCard, func: Callable = None):
        super().__init__(card)
        self._func = func

    async def invoke(self, inputs: Input, **kwargs) -> Output:
        inputs = self.card.validate_inputs(inputs)
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
        inputs = self.card.validate_inputs(inputs)
        if inspect.isasyncgenfunction(self._func):
            async for item in self._func(**inputs):
                yield item
        else:
            raise JiuWenBaseException(
                error_code=StatusCode.PLUGIN_UNEXPECTED_ERROR.code, message="stream function need aysnc generator"
            )
