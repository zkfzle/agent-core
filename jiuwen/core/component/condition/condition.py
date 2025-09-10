#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import abstractmethod
from typing import Callable, Any

from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.atomic_node import AtomicNode
from jiuwen.core.graph.executable import Input, Output

INDEX = "index"


class Condition(AtomicNode):
    def __init__(self, input_schema: Any = None):
        self._input_schema = input_schema

    def __call__(self, context: BaseRuntime) -> bool:
        return self.atomic_invoke(context=context)

    def _atomic_invoke(self, **kwargs) -> Any:
        context: BaseRuntime = kwargs["context"]
        inputs = context.state().get_inputs(self._input_schema) if self._input_schema is not None else {}
        result = self.invoke(inputs=inputs, context=context)
        if isinstance(result, tuple):
            context.state().set_outputs(result[1])
            result = result[0]
        return result

    @abstractmethod
    def invoke(self, inputs: Input, context: BaseRuntime) -> Output:
        pass


class FuncCondition(Condition):
    def __init__(self, func: Callable[[], bool]):
        super().__init__()
        self._func = func

    def invoke(self, inputs: Input, context: BaseRuntime) -> Output:
        return self._func()


class AlwaysTrue(Condition):
    def invoke(self, inputs: Input, context: BaseRuntime) -> Output:
        return True
