#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import AsyncIterator, Self, Union, Callable, Any

from langgraph.constants import END, START

from jiuwen.core.common.constants.constant import INDEX
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.component.break_comp import BreakComponent, LoopController
from jiuwen.core.component.condition.condition import Condition, AlwaysTrue, FuncCondition
from jiuwen.core.component.condition.expression import ExpressionCondition
from jiuwen.core.component.loop_callback.loop_callback import LoopCallback, END_ROUND, START_ROUND, OUT_LOOP, FIRST_LOOP
from jiuwen.core.component.loop_callback.loop_id import LoopIdCallback
from jiuwen.core.graph.atomic_node import AtomicNode
from jiuwen.core.graph.base import Graph, INPUTS_KEY
from jiuwen.core.graph.executable import Output, Input, Executable
from jiuwen.core.runtime.config import WorkflowConfig
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.workflow.base import BaseWorkFlow


class EmptyExecutable(Executable):
    async def on_invoke(self, inputs: Input, runtime: BaseRuntime) -> Output:
        pass

    def skip_trace(self) -> bool:
        return True


class LoopGroup(BaseWorkFlow, Executable):

    def __init__(self, workflow_config: WorkflowConfig, new_graph: Graph):
        super().__init__(workflow_config, new_graph)
        self.compiled = None
        self.group_input_schema = {}

    def start_nodes(self, nodes: list[str]) -> Self:
        for node in nodes:
            self.start_comp(node)
        return self

    def end_nodes(self, nodes: list[str]) -> Self:
        for node in nodes:
            self.end_comp(node)
        return self

    async def on_invoke(self, inputs: Input, runtime: BaseRuntime) -> Output:
        if self.compiled is None:
            raise JiuWenBaseException(-1, "loop graph is not compiled")
        await self.compiled.invoke(inputs, runtime)
        return None

    def skip_trace(self) -> bool:
        return True

    def graph_invoker(self) -> bool:
        return True


BROKEN = "_broken"
FIRST_IN_LOOP = "_first_in_loop"

CONDITION_NODE_ID = "condition"
BODY_NODE_ID = "body"


class LoopComponent(WorkflowComponent, LoopController, Executable, AtomicNode):

    def __init__(self, node_id: str, body: Executable, new_graph: Graph,
                 condition: Union[str, Callable[[], bool], Condition] = None, break_nodes: list[BreakComponent] = None,
                 callbacks: list[LoopCallback] = None):
        super().__init__()
        self._node_id = node_id
        self._body = body

        self._condition: Condition
        if condition is None:
            self._condition = AlwaysTrue()
        elif isinstance(condition, Condition):
            self._condition = condition
        elif isinstance(condition, Callable):
            self._condition = FuncCondition(condition)
        elif isinstance(condition, str):
            self._condition = ExpressionCondition(condition)

        if break_nodes:
            for break_node in break_nodes:
                break_node.set_controller(self)

        loop_id_callback = LoopIdCallback(node_id)

        self._callbacks: list[LoopCallback] = []

        self.register_callback(loop_id_callback)
        if callbacks:
            for callback in callbacks:
                self.register_callback(callback)

        self._graph = new_graph
        self._graph.add_node(BODY_NODE_ID, self._body)
        self._graph.add_node(CONDITION_NODE_ID, EmptyExecutable())
        self._graph.add_edge(START, CONDITION_NODE_ID)
        self._graph.add_edge(BODY_NODE_ID, CONDITION_NODE_ID)
        self._graph.add_conditional_edges(CONDITION_NODE_ID, self)

        self._in_loop = [BODY_NODE_ID]
        self._out_loop = [END]
        self._runtime = None

    def to_executable(self) -> Executable:
        return self

    def register_callback(self, callback: LoopCallback):
        self._callbacks.append(callback)

    def __call__(self, *args, **kwargs) -> list[str]:
        return self.atomic_invoke(runtime=self._runtime)

    def _atomic_invoke(self, **kwargs) -> Any:
        inputs = self._runtime.state().get_inputs(self._node_id)
        outputs = self._condition_invoke(inputs=inputs, runtime=self._runtime)
        self._runtime.state().set_outputs({self._node_id: outputs[1]})
        return outputs[0]

    def _condition_invoke(self, inputs: Input, runtime: BaseRuntime) -> Output:
        index = self._runtime.state().get(INDEX)
        runtime.state().update(inputs)
        if index is None:
            raise JiuWenBaseException(-1, 'inner error, loop index is not set')
        continue_loop = False if self.is_broken() else self._condition(runtime=runtime)
        for callback in self._callbacks:
            if index < 0:
                callback(FIRST_LOOP, runtime)
            else:
                callback(END_ROUND, runtime)
            if continue_loop:
                callback(START_ROUND, runtime)
            else:
                callback(OUT_LOOP, runtime)
        index = index + 1 if continue_loop else -1
        runtime.state().update({INDEX: index})
        if not continue_loop:
            runtime.state().update({INDEX: -1, BROKEN: False})
        return self._in_loop if continue_loop else self._out_loop, {INDEX: index}

    def is_broken(self) -> bool:
        _is_broken = self._runtime.state().get(BROKEN)
        if isinstance(_is_broken, bool):
            return _is_broken
        return False

    def break_loop(self):
        self._runtime.state().update({BROKEN: True})

    async def on_invoke(self, inputs: Input, runtime: BaseRuntime) -> Output:
        self._runtime = runtime
        # set loop graph inputs
        self._runtime.state().update(inputs.get(INPUTS_KEY) if INPUTS_KEY in inputs else inputs)
        index = self._runtime.state().get(INDEX)
        if index is None:
            self._runtime.state().update({BROKEN: False, INDEX: -1})
        if self._runtime.tracer() is not None:
            self._runtime.tracer().register_workflow_span_manager(self._runtime.executable_id())
        compiled = self._graph.compile(self._runtime)
        if isinstance(self._body, LoopGroup):
            self._body.compiled = self._body.compile(self._runtime)
            return await compiled.invoke(inputs, self._runtime)
        return None

    def graph_invoker(self) -> bool:
        return True
