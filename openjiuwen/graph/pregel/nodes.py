#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Any, Callable

from openjiuwen.graph.pregel.router import IRouter


class PregelNode:
    def __init__(self, name: str, func: Callable[[Any], Any], routers: list[IRouter]):
        self.name = name
        self.func = func
        self.routers = routers
