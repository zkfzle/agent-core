#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from jiuwen.core.component.loop_callback.loop_callback import LoopCallback
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.executable import Output
from jiuwen.core.common.constants.constant import INDEX, LOOP_ID


class LoopIdCallback(LoopCallback):
    def __init__(self, node_id: str):
        self._node_id = node_id

    def first_in_loop(self, runtime: BaseRuntime) -> Output:
        runtime.state().update_global({self._node_id + "." + INDEX: runtime.state().get(INDEX) + 1})
        runtime.state().update_global({LOOP_ID: self._node_id})
        return None

    def out_loop(self, runtime: BaseRuntime) -> Output:
        runtime.state().update_global({LOOP_ID: None})
        return None

    def start_round(self, runtime: BaseRuntime) -> Output:
        runtime.state().update_global({self._node_id + "." + INDEX: runtime.state().get(INDEX) + 1})
        return None

    def end_round(self, runtime: BaseRuntime) -> Output:
        return None
