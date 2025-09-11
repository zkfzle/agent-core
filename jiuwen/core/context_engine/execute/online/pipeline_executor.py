#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List, Optional

from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.context_engine.base import ContextWindow
from jiuwen.core.context_engine.processor.base import BaseContextProcessor
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.context_engine.config import ContextEngineConfig


class PipelineContextExecutor:
    def __init__(self, config: Optional[ContextEngineConfig] = None):
        self._processor_pipeline: List[BaseContextProcessor] = []
        self._config: Optional[ContextEngineConfig] = config

    def build_from_config(self, config: ContextEngineConfig, llm: Optional[BaseChatModel] = None):
        if not config:
            return

        for processor_config in config.processors:
            processor = ProcessorFactory().create_processor(processor_config, llm)
            if not processor:
                logger.warning(f"preprocessor type error: {processor_config.processor_type}")
                continue
            self._processor_pipeline.append(processor)

    def run(self, context_window: ContextWindow) -> ContextWindow:
        for processor in self._processor_pipeline:
            context_window: ContextWindow = processor.run(context_window)
        return context_window
