#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Dict, List, Optional

from jiuwen.core.context_engine.base import ContextVariable


class VariableManager:
    def __init__(self):
        self.__variables: Dict[str, ContextVariable] = {}

    def __contains__(self, name: str):
        return name in self.__variables

    def get(self, var_name: str) -> Optional[ContextVariable]:
        return self.__variables.get(var_name)

    def set(self, var_name: str, value: ContextVariable):
        self.__variables[var_name] = value

    def update(self, var_name: str, value: ContextVariable):
        if var_name in self.__variables:
            self.__variables[var_name] = value

    def set_batch(self, variables: List[ContextVariable]):
        for var in variables:
            self.__variables[var.name] = var

    def get_all(self) -> Dict[str, ContextVariable]:
        return self.__variables.copy()
