#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from jiuwen.core.context_engine.execute.asynch.pipeline import AsyncProcessPipeline
from jiuwen.core.context_engine.execute.online.pipeline import OnlineProcessPipeline
from jiuwen.core.context_engine.base import EngineInput, EngineOutput


class ContextEngine:
    def __init__(self, config=None):
        self.__config = config
        self.__online_pipeline = OnlineProcessPipeline(self.__config)
        self.__async_pipeline = AsyncProcessPipeline(self.__config)

    def process(self, data: EngineInput) -> EngineOutput:
        pass