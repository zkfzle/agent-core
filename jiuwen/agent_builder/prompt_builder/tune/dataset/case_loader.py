#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List, Set

from jiuwen.agent_builder.prompt_builder.tune.base import Case, EvaluatedCase


class CaseLoader:
    def __init__(self, cases: List[Case]):
        self._cases = cases

    @staticmethod
    def shuffle(cases: List[Case]) -> List[Case]:
        pass

    def size(self) -> int:
        return len(self._cases)

    def get_cases(self) -> List[Case]:
        return self._cases

    def get_variable_keys(self) -> Set[str]:
        variable_keys = set()
        for case in self._cases:
            variable_keys.update(case.inputs.keys())
        return variable_keys