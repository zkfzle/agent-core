#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import asyncio
from typing import List, Callable, Optional, Dict

from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.context_engine.processor.asynch.base import AsyncProcessStage, AsyncProcessCallbacks
from jiuwen.core.context_engine.config import AsyncExecuteConfig

DEFAULT_TASK_SCAN_PERIOD = 0.2


class AsyncProcessPipeline:
    def __init__(self,
                 get_context_window_func: Callable,
                 config: Optional[AsyncExecuteConfig],
                 update_data_callbacks: Dict[str, Callable],
                 ):
        self.__config = config
        self.__pending_processors: List[AsyncProcessStage] = []
        self.__async_task_pipeline = []
        self.__get_context_window_func = get_context_window_func
        self.__update_data_callbacks = update_data_callbacks

    def build_from_config(self, config: AsyncExecuteConfig, llm: Optional[BaseChatModel] = None):
        if not config:
            return

        for async_processor_config in config.processors:
            async_processor = ProcessorFactory().create_processor(async_processor_config)
            if not async_processor or not isinstance(async_processor, AsyncProcessStage):
                logger.warning(f"async processor type error: {async_processor_config.processor_type}")
                continue
            if async_processor.need_llm():
                async_processor.bind_llm(llm)
            async_processor.set_callbacks(AsyncProcessCallbacks(
                finished_callback=self.__process_finished,
                exception_callback=self.__process_abnormal,
                data_update_callback=self.__update_data_callbacks.get(async_processor.update_strategy(), None),
            ))
            self.__pending_processors.append(async_processor)
        if self.__pending_processors:
            asyncio.create_task(self.__start())

    async def __start(self):
        while True:
            pending_processors = self.__pending_processors
            self.__pending_processors = []
            for async_processor in pending_processors:
                context_window = self.__get_context_window_func()
                if not async_processor.is_ready(context_window):
                    self.__pending_processors.append(async_processor)
                    continue
                self.__async_task_pipeline.append(asyncio.create_task(
                    async_processor(context_window)
                ))

            await asyncio.sleep(DEFAULT_TASK_SCAN_PERIOD)

    def __process_finished(self, async_processor: AsyncProcessStage):
        self.__pending_processors.append(async_processor)

    def __process_abnormal(self, async_processor: AsyncProcessStage):
        self.__pending_processors.append(async_processor)