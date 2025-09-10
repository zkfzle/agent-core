#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import AsyncIterator

from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime


class CommonNode(ComponentExecutable, WorkflowComponent):

    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id

    async def invoke(self, inputs: Input, context: Runtime) -> Output:
        return inputs

    async def stream(self, inputs: Input, context: Runtime) -> AsyncIterator[Output]:
        yield await self.invoke(inputs, context)


class AddTenNode(ComponentExecutable, WorkflowComponent):

    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id

    async def invoke(self, inputs: Input, context: Runtime) -> Output:
        return {"result": inputs["source"] + 10}
