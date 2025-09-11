#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Generic

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, V

from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.executable import Input

class BaseCheckpointer(BaseCheckpointSaver[V], Generic[V], ABC):

    def __init__(self):
        super().__init__()
        self.ctx: BaseRuntime = None
        self.input: Input = None

    def register_runtime(self, runtime: BaseRuntime):
        self.ctx = runtime

    def register_input(self, input: Input):
        self.input = input

    @abstractmethod
    def recover(self, config: RunnableConfig):
        raise NotImplementedError

    @abstractmethod
    def save(self, config: RunnableConfig):
        raise NotImplementedError
