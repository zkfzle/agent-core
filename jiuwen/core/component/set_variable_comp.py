#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any

from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.context_engine.base import Context
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.runtime.utils import extract_origin_key, is_ref_path


class SetVariableComponent(WorkflowComponent, ComponentExecutable):

    def __init__(self, variable_mapping: dict[str, Any]):
        super().__init__()
        self._variable_mapping = variable_mapping

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        result = {}
        parent_runtime = runtime.base().parent()
        for left, right in self._variable_mapping.items():
            left_ref_str = extract_origin_key(left)
            if left_ref_str == "":
                left_ref_str = left
            if isinstance(right, str) and is_ref_path(right):
                ref_str = extract_origin_key(right)
                result[left_ref_str] = runtime.get_global_state(ref_str)
                continue
            result[left_ref_str] = right
        parent_runtime.state().update(result)
        parent_runtime.state().set_outputs(result)
        return None
