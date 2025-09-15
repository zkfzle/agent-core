#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Union

from jiuwen.core.component.condition.condition import Condition
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.common.constants.constant import INDEX


class NumberCondition(Condition):
    def __init__(self, limit: Union[str, int], index_path: str = None):
        super().__init__(limit)
        self._index_path = index_path if index_path else INDEX
        self._limit = limit

    def invoke(self, inputs: Input, runtime: BaseRuntime) -> Output:
        current_idx = runtime.state().get(self._index_path) + 1
        limit_num = inputs
        return current_idx < limit_num
