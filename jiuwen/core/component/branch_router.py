#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Callable, Union

from jiuwen.core.component.condition.condition import Condition, FuncCondition
from jiuwen.core.component.condition.expression import ExpressionCondition
from jiuwen.core.runtime.runtime import BaseRuntime, Runtime, NodeRuntime
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

    def trace_info(self) -> str:
        return self._condition.trace_info()


class BranchRouter:
    def __init__(self, report_trace: bool = False):
        super().__init__()
        self._branches: list[Branch] = []
        self._runtime: Runtime = None
        self.report_trace = report_trace

    def add_branch(self, condition: Union[str, Callable[[], bool], Condition], target: list[str],
                   branch_id: str = None):
        self._branches.append(Branch(condition, target, branch_id))

    def set_runtime(self, runtime: Runtime):
        self._runtime = runtime

    async def __call__(self, *args, **kwargs) -> list[str]:
        runtime = self._runtime
        if self.report_trace:
            branches = []
            for branch in self._branches:
                branches.append({
                    "branch_id": branch.branch_id,
                    "condition": branch.trace_info()
                })
            await runtime.base().trace_inputs({
                "branches": branches
            })
        for branch in self._branches:
            if branch.evaluate(self._runtime.base()):
                if self.report_trace:
                    await runtime.base().trace_outputs({
                        "branch_id": branch.branch_id
                    })
                return branch.target
        return []
