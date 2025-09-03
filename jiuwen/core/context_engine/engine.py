#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import Optional, Callable, Dict

from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.context_engine.config import ContextEngineConfig
from jiuwen.core.context_engine.execute.asynch.pipeline import AsyncProcessPipeline
from jiuwen.core.context_engine.execute.online.pipeline import OnlineProcessPipeline
from jiuwen.core.context_engine.base import EngineInput, EngineOutput


class ContextEngine:
    def __init__(self,
                 config: ContextEngineConfig = None,
                 get_context_func: Callable = None,
                 update_context_callbacks: Dict[str, Callable] = None,
                 ):
        self.__config = config
        self.__online_pipeline: OnlineProcessPipeline = OnlineProcessPipeline(None)
        self.__async_pipeline: AsyncProcessPipeline = AsyncProcessPipeline(get_context_func, None,
                                                                           update_context_callbacks)
        self.__llm: Optional[BaseChatModel] = None
        self.build_from_config(config, get_context_func, update_context_callbacks)

    def build_from_config(self,
                          config: ContextEngineConfig,
                          get_context_func: Callable,
                          update_context_callbacks: Dict[str, Callable]):
        self.__config = config
        online_process_config = config.online_process if config else None
        async_process_config = config.async_process if config else None
        self.__llm = self.__init_model()
        self.__online_pipeline = OnlineProcessPipeline(online_process_config)
        self.__online_pipeline.build_from_config(online_process_config, self.__llm)
        self.__async_pipeline = AsyncProcessPipeline(get_context_func, async_process_config, update_context_callbacks)
        self.__async_pipeline.build_from_config(async_process_config, self.__llm)

    def process(self, data: EngineInput) -> EngineOutput:
        return self.__online_pipeline.run(data)

    def __init_model(self):
        if not self.__config or not self.__config.model_provider or not self.__config.model_info:
            return None
        try:
            return ModelFactory().get_model(model_provider=self.__config.model_provider,
                                            model_info=self.__config.model_info)
        except Exception:
            logger.error("Failed to init model while loading context engine")
            return None
