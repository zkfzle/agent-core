#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import AsyncIterator, Iterator, Any

from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.context.context import Context
from jiuwen.core.context.utils import extract_origin_key, is_ref_path, NESTED_PATH_SPLIT
from jiuwen.core.graph.executable import Executable, Input, Output


class SetVariableComponent(WorkflowComponent, Executable):

    def __init__(self, variable_mapping: dict[str, Any]):
        super().__init__()
        self._variable_mapping = variable_mapping

    async def invoke(self, inputs: Input, context: Context) -> Output:
        result = {}
        parent_context = context.parent_context()
        for left, right in self._variable_mapping.items():
            left_ref_str = extract_origin_key(left)
            if left_ref_str == "":
                left_ref_str = left
            if isinstance(right, str) and is_ref_path(right):
                ref_str = extract_origin_key(right)
                result[left_ref_str] = context.state().get_global(ref_str)
                continue
            result[left_ref_str] = right
        parent_context.state().update(result)
        parent_context.state().set_outputs(result)
        return None

    async def stream(self, inputs: Input, context: Context) -> Iterator[Output]:
        yield self.invoke(inputs, context)

    def interrupt(self, message: dict):
        pass

    def to_executable(self) -> Executable:
        return self

    async def collect(self, inputs: AsyncIterator[Input], contex: Context) -> Output:
        pass

    async def transform(self, inputs: AsyncIterator[Input], context: Context) -> AsyncIterator[Output]:
        pass
