#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import abstractmethod
from typing import Optional, Callable
from enum import Enum
import asyncio

from jiuwen.core.common.logging import logger
from jiuwen.core.context_engine.base import EngineInput, EngineOutput
from jiuwen.core.context_engine.config import BaseAsyncProcessorConfig
from jiuwen.core.context_engine.processor.base import ProcessStage

DEFAULT_ASYNC_PROCESS_TIMEOUT: float = 5.0


class AsyncProcessStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    FINISHED = "finished"


class AsyncProcessCallbacks:
    def __init__(self,
                 finished_callback: Callable,
                 data_update_callback: Callable = None,
                 exception_callback: Callable = None):
        self.finished_callback: Callable[["AsyncProcessStage"], None] = finished_callback
        self.data_update_callback: Optional[Callable[[EngineInput], None]] =(
                data_update_callback or AsyncProcessCallbacks.default_data_update_callback)
        self.exception_callback: Callable[["AsyncProcessStage"], None] = exception_callback

    @staticmethod
    def default_data_update_callback(output: EngineOutput):
        pass


class AsyncProcessStage(ProcessStage):
    def __init__(self, config: BaseAsyncProcessorConfig):
        super().__init__(config)
        self.__statue: AsyncProcessStatus = AsyncProcessStatus.PENDING
        self.__callbacks: Optional[AsyncProcessCallbacks] = None

    async def __call__(self, input_data: EngineInput):
        try:
            output = await asyncio.wait_for(self.arun(input_data), DEFAULT_ASYNC_PROCESS_TIMEOUT)
        except TimeoutError as e:
            logger.error(f"Async processor timed out: {e}")
            self.__callbacks.exception_callback(self)
            return
        except Exception as e:
            logger.error(f"Async processor error: {e}")
            self.__callbacks.exception_callback(self)
            return
        finally:
            if self._llm:
                self._llm.close()
        self.__callbacks.finished_callback(self)
        self.__callbacks.data_update_callback(output)

    def set_callbacks(self, callbacks: AsyncProcessCallbacks):
        self.__callbacks = callbacks

    @abstractmethod
    async def arun(self, input_data: EngineInput) -> Optional[EngineOutput]:
        pass

    @abstractmethod
    def is_ready(self, input_data: EngineInput) -> bool:
        pass

    @abstractmethod
    def update_strategy(self) -> str:
        pass
