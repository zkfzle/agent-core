#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List
import random

from jiuwen.agent_builder.prompt_builder.tune.base import Case


class CaseLoader:
    def __init__(self, cases: List[Case]):
        self._cases = cases

    def shuffle(self):
        random.shuffle(self._cases)

    def size(self) -> int:
        return len(self._cases)

    def get_cases(self) -> List[Case]:
        return self._cases
