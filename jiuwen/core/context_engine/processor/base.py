#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import ABC, abstractmethod
from typing import Optional

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.context_engine.config import BaseProcessorConfig
from jiuwen.core.context_engine.base import ContextWindow


class BaseContextProcessor(ABC):
    def __init__(self, config: BaseProcessorConfig):
        self._config = config
        self._llm: Optional[BaseChatModel] = None

    @abstractmethod
    def run(self, context: ContextWindow) -> Optional[ContextWindow]:
        pass

    def bind_llm(self, llm: BaseChatModel):
        self._llm = llm

    @property
    def config(self) -> BaseProcessorConfig:
        return self._config
