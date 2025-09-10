#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import abstractmethod
from typing import Any

from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.atomic_node import AtomicNode
from jiuwen.core.graph.executable import Output

FIRST_LOOP = "first_in_loop"
START_ROUND = "start_round"
END_ROUND = "end_round"
OUT_LOOP = "out_loop"


class LoopCallback(AtomicNode):
    def __call__(self, input: str, context: BaseRuntime) -> None:
        self.atomic_invoke(input=input, context=context)

    def _atomic_invoke(self, **kwargs) -> Any:
        input = kwargs.get("input")
        context = kwargs.get("context")
        if input == FIRST_LOOP:
            output = self.first_in_loop(context)
        elif input == START_ROUND:
            output = self.start_round(context)
        elif input == END_ROUND:
            output = self.end_round(context)
        else:
            output = self.out_loop(context)
        if output is not None:
            context.state().set_outputs(output)
        return None

    @abstractmethod
    def first_in_loop(self, context: BaseRuntime) -> Output:
        raise NotImplementedError

    @abstractmethod
    def out_loop(self, context: BaseRuntime) -> Output:
        raise NotImplementedError

    @abstractmethod
    def start_round(self, context: BaseRuntime) -> Output:
        raise NotImplementedError

    @abstractmethod
    def end_round(self, context: BaseRuntime) -> Output:
        raise NotImplementedError
