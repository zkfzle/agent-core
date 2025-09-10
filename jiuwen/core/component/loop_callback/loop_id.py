#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from jiuwen.core.component.loop_callback.loop_callback import LoopCallback
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.executable import Output
from jiuwen.core.component.condition.condition import INDEX

LOOP_ID = "__sys_loop_id"


class LoopIdCallback(LoopCallback):
    def __init__(self, node_id: str):
        self._node_id = node_id

    def first_in_loop(self, context: BaseRuntime) -> Output:
        context.state().update_global({self._node_id + "." + INDEX: context.state().get(INDEX) + 1})
        context.state().update_global({LOOP_ID: self._node_id})
        return None

    def out_loop(self, context: BaseRuntime) -> Output:
        context.state().update_global({LOOP_ID: None})
        return None

    def start_round(self, context: BaseRuntime) -> Output:
        context.state().update_global({self._node_id + "." + INDEX: context.state().get(INDEX) + 1})
        return None

    def end_round(self, context: BaseRuntime) -> Output:
        return None
