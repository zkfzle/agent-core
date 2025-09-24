#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List

from jiuwen.core.utils.tool.param import Param
from jiuwen.core.utils.tool.function.function import LocalFunction


def tool(*,
         name:str = None,
         descripton: str = None,
         params: List[Param] = None) ->LocalFunction:

    def decorator(func):
        last_description = descripton or func.__doc__
        last_name = name or func.__name__
        return LocalFunction(name=last_name, description = last_description, params=params, func = func)

    return decorator
