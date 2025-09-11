#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import asyncio
from functools import partial
from typing import List, Callable, Optional

from jiuwen.core.common.logging import logger
from jiuwen.core.context_engine.accessor.accessor import ContextAccessor
from jiuwen.core.context_engine.base import ContextWindow
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.context_engine.processor.asynch.base import AsyncContextProcess, AsyncProcessCallbacks
from jiuwen.core.context_engine.config import ContextEngineConfig

DEFAULT_TASK_SCAN_PERIOD = 0.2

GET_CONTEXT_WINDOW_FUNC_TYPE = Callable[[], ContextWindow]
UPDATE_CONTEXT_FUNC_TYPE = Callable[[ContextWindow], None]


class AsyncContextExecutor:
    def __init__(self,
                 config: Optional[ContextEngineConfig],
                 accessor: ContextAccessor):
        self._config = config
        self._pending_processors: List[AsyncContextProcess] = []
        self._async_task_pipeline = []
        self._accessor = accessor

    def build_from_config(self, config: ContextEngineConfig, llm: Optional[BaseChatModel] = None):
        if not config:
            return

        for async_processor_config in config.async_processors:
            async_processor = ProcessorFactory().create_processor(async_processor_config, llm)
            if not async_processor or not isinstance(async_processor, AsyncContextProcess):
                logger.warning(f"async processor type error: {async_processor_config.processor_type}")
                continue
            async_processor.set_callbacks(AsyncProcessCallbacks(
                finished_callback=self._process_finished,
                exception_callback=self._process_abnormal,
                data_update_callback=partial(self._accessor.update_context_by_type, async_processor.update_strategy()),
            ))
            self._pending_processors.append(async_processor)

        self._start_async_task_loop()

    async def _start(self):
        while True:
            pending_processors = self._pending_processors
            self._pending_processors = []
            for async_processor in pending_processors:
                context_window = self._accessor.create_context_window()
                if not async_processor.is_ready(context_window):
                    self._pending_processors.append(async_processor)
                    continue
                self._async_task_pipeline.append(asyncio.create_task(
                    async_processor(context_window)
                ))

            await asyncio.sleep(DEFAULT_TASK_SCAN_PERIOD)

    def _start_async_task_loop(self):
        if not self._pending_processors:
            return
        try:
            asyncio.create_task(self._start())
        except RuntimeError as e:
            logger.warning(f"cannot start async context processing: {e}")

    def _process_finished(self, async_processor: AsyncContextProcess):
        self._pending_processors.append(async_processor)

    def _process_abnormal(self, async_processor: AsyncContextProcess):
        self._pending_processors.append(async_processor)