#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Union, Any

from jiuwen.core.component.condition.condition import Condition
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.common.constants.constant import INDEX

DEFAULT_MAX_LOOP_NUMBER = 1000
DEFAULT_PATH_ARRAY_LOOP_VAR = "arrLoopVar"


class ArrayCondition(Condition):
    def __init__(self, node_id: str, arrays: dict[str, Union[str, list[Any]]]):
        super().__init__(arrays)
        self._node_id = node_id
        self._arrays = arrays

    def invoke(self, inputs: Input, runtime: BaseRuntime) -> Output:
        current_idx = runtime.state().get(INDEX) + 1
        min_length = DEFAULT_MAX_LOOP_NUMBER
        updates: dict[str, Any] = {}
        for key, array_info in self._arrays.items():
            arr = inputs.get(key, [])
            min_length = min(len(arr), min_length)
            if current_idx >= min_length:
                return False
            updates[key] = arr[current_idx]
        runtime.state().update({self._node_id: updates})
        return True, {self._node_id: updates}
