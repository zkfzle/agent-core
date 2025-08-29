#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import Optional

from jiuwen.core.context_engine.config import ContextEngineConfig
from jiuwen.core.context_engine.execute.asynch.pipeline import AsyncProcessPipeline
from jiuwen.core.context_engine.execute.online.pipeline import OnlineProcessPipeline
from jiuwen.core.context_engine.base import EngineInput, EngineOutput


class ContextEngine:
    def __init__(self, config: ContextEngineConfig = None):
        self.__config = config
        self.__online_pipeline: Optional[OnlineProcessPipeline] = None
        self.__async_pipeline: Optional[AsyncProcessPipeline] = None
        self.build_from_config(config)

    def build_from_config(self, config: ContextEngineConfig):
        self.__config = config
        online_process_config = config.online_process if config else None
        async_process_config = config.async_process if config else None
        self.__online_pipeline = OnlineProcessPipeline(online_process_config)
        self.__async_pipeline = AsyncProcessPipeline(async_process_config)

    def process(self, data: EngineInput) -> EngineOutput:
        return self.__online_pipeline.run(data)