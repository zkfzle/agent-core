#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List

from jiuwen.core.context_engine.base import EngineInput, EngineOutput
from jiuwen.core.context_engine.processor.base import BaseProcessor
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.context_engine.config import OnlineExecuteConfig


class OnlineProcessPipeline:
    def __init__(self, config: OnlineExecuteConfig):
        self.__execute_pipeline: List[BaseProcessor] = []
        self.__config: OnlineExecuteConfig = config

    def build_from_config(self, config: OnlineExecuteConfig):
        if not config:
            return
        for processor_config in config.processors:
            processor = ProcessorFactory().create_processor(processor_config)
            if processor:
                self.__execute_pipeline.append(processor)

    def run(self, input: EngineInput) -> EngineOutput:
        for processor in self.__execute_pipeline:
            pass
