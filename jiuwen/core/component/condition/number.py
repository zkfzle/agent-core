#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Union, Any

from jiuwen.core.component.condition.condition import Condition, INDEX
from jiuwen.core.context.context import Context
from jiuwen.core.graph.executable import Input, Output


class NumberCondition(Condition):
    def __init__(self, limit: Union[str, int], index_path: str = None):
        super().__init__(limit)
        self._index_path = index_path if index_path else INDEX
        self._limit = limit

    def invoke(self, inputs: Input, context: Context) -> Output:
        current_idx = context.state().get(self._index_path) + 1
        limit_num = inputs
        return current_idx < limit_num
