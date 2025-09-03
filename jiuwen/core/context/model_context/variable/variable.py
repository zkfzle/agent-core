#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Dict, Any, Optional

from jiuwen.core.context.model_context.base import ContextVariable, Serializable
from jiuwen.core.context_engine.base import EngineOutput


class VariableManager(Serializable):
    def __init__(self):
        self.__variables: Dict[str, ContextVariable] = {}

    def get(self, var_name: str) -> Optional[ContextVariable]:
        return self.__variables.get(var_name)

    def set(self, var_name: str, value: ContextVariable):
        self.__variables[var_name] = value

    def serialize(self) -> Dict[str, Any]:
        return dict(zip(list(self.__variables.keys()),
                        [var.model_dump() for var in self.__variables.values()]))

    def deserialize(self, data: Dict[str, Any]):
        self.__variables.update(data)

    """ async update processing"""
    def update_variable(self, update_output: EngineOutput):
        for var_name, var_value in update_output.variables.items():
            if var_name in self.__variables:
                self.__variables[var_name].value = var_value