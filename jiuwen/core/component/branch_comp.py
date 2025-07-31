#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from contextvars import Context
from typing import Callable, Union, Hashable, Iterator, AsyncIterator

from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.component.branch_router import BranchRouter
from jiuwen.core.component.condition.condition import Condition
from jiuwen.core.graph.base import Graph
from jiuwen.core.graph.executable import Executable, Input, Output


class BranchComponent(WorkflowComponent, Executable):

    def __init__(self, executable: Executable = None):
        super().__init__()
        self._router = BranchRouter()
        self._executable = executable

    def add_branch(self, condition: Union[str, Callable[[], bool], Condition], target: Union[str, list[str]],
                   branch_id: str = None):
        if isinstance(target, str):
            target = [target]
        self._router.add_branch(condition, target, branch_id=branch_id)

    def router(self) -> Callable[..., Union[Hashable, list[Hashable]]]:
        return self._router

    def to_executable(self) -> Executable:
        return self

    async def invoke(self, inputs: Input, context: Context) -> Output:
        self._router.set_context(context)
        if self._executable:
            return self._executable.invoke(inputs, context)
        return inputs

    async def stream(self, inputs: Input, context: Context) -> Iterator[Output]:
        yield self.invoke(inputs, context)

    def add_component(self, graph: Graph, node_id: str, wait_for_all: bool = False):
        graph.add_node(node_id, self.to_executable(), wait_for_all=wait_for_all)
        graph.add_conditional_edges(node_id, self.router())

    async def collect(self, inputs: AsyncIterator[Input], contex: Context) -> Output:
        pass

    async def transform(self, inputs: AsyncIterator[Input], context: Context) -> AsyncIterator[Output]:
        pass

    def interrupt(self, message: dict):
        pass
