#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Union, Any

from jiuwen.core.component.loop_callback.loop_callback import LoopCallback
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.executable import Output


class IntermediateLoopVarCallback(LoopCallback):
    def __init__(self, intermediate_loop_var: dict[str, Union[str, Any]],
                 intermediate_loop_var_root: str = "intermediateLoopVar"):
        self.intermediate_loop_var = intermediate_loop_var
        self.intermediate_loop_var_root = intermediate_loop_var_root

    def first_in_loop(self, runtime: BaseRuntime) -> Output:
        vars = runtime.state().get(self.intermediate_loop_var)
        runtime.state().update({self.intermediate_loop_var_root: vars})
        return {self.intermediate_loop_var_root: vars}

    def out_loop(self, runtime: BaseRuntime) -> Output:
        runtime.state().update({self.intermediate_loop_var_root: None})
        return {self.intermediate_loop_var_root: None}

    def start_round(self, runtime: BaseRuntime) -> Output:
        return None

    def end_round(self, runtime: BaseRuntime) -> Output:
        return None
