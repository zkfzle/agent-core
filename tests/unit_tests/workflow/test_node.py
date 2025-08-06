#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import AsyncIterator, Iterator

from jiuwen.core.common.logging import logger
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.context.context import Context, NodeContext
from jiuwen.core.context.utils import NESTED_PATH_SPLIT
from jiuwen.core.graph.executable import Executable, Input, Output


class CommonNode(Executable, WorkflowComponent):

    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id

    async def invoke(self, inputs: Input, context: Context) -> Output:
        return inputs

    async def stream(self, inputs: Input, context: Context) -> AsyncIterator[Output]:
        yield await self.invoke(inputs, context)

    async def collect(self, inputs: AsyncIterator[Input], contex: Context) -> Output:
        pass

    async def transform(self, inputs: AsyncIterator[Input], context: Context) -> AsyncIterator[Output]:
        pass

    async def interrupt(self, message: dict):
        pass

    def to_executable(self) -> Executable:
        return self


class AddTenNode(Executable, WorkflowComponent):

    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id

    async def invoke(self, inputs: Input, context: Context) -> Output:
        return {"result": inputs["source"] + 10}

    async def stream(self, inputs: Input, context: Context) -> AsyncIterator[Output]:
        yield await self.invoke(inputs, context)

    async def collect(self, inputs: AsyncIterator[Input], contex: Context) -> Output:
        pass

    async def transform(self, inputs: AsyncIterator[Input], context: Context) -> AsyncIterator[Output]:
        pass

    async def interrupt(self, message: dict):
        pass

    def to_executable(self) -> Executable:
        return self
