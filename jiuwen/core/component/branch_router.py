#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Callable, Union

from jiuwen.core.component.condition.condition import Condition, FuncCondition
from jiuwen.core.component.condition.expression import ExpressionCondition
from jiuwen.core.runtime.runtime import BaseRuntime, Runtime
from jiuwen.core.common.exception.exception import JiuWenBaseException


class Branch:
    def __init__(self, condition: Union[str, Callable[[], bool], Condition], target: list[str],
                 branch_id: str = None):
        super().__init__()
        self.branch_id = branch_id
        if isinstance(condition, Condition):
            self._condition = condition
        elif isinstance(condition, str):
            self._condition = ExpressionCondition(condition)
        elif isinstance(condition, Callable):
            self._condition = FuncCondition(condition)
        else:
            raise JiuWenBaseException(-1, "condition must be either a string or a callable")
        self.target = target

    def evaluate(self, runtime: BaseRuntime) -> bool:
        return self._condition(runtime)


class BranchRouter:
    def __init__(self):
        super().__init__()
        self._branches: list[Branch] = []
        self._runtime: Runtime = None

    def add_branch(self, condition: Union[str, Callable[[], bool], Condition], target: list[str],
                   branch_id: str = None):
        self._branches.append(Branch(condition, target, branch_id))

    def set_runtime(self, runtime: Runtime):
        self._runtime = runtime

    def __call__(self, *args, **kwargs) -> list[str]:
        for branch in self._branches:
            if branch.evaluate(self._runtime.base()):
                return branch.target
        return []
