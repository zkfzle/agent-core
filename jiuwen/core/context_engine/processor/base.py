#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import ABC
from typing import Optional

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.context_engine.config import BaseProcessorConfig


class ProcessStage(ABC):
    def __init__(self, config: BaseProcessorConfig):
        self._config = config
        self._llm: Optional[BaseChatModel] = None

    @staticmethod
    def need_llm() -> bool:
        return False

    def bind_llm(self, llm: BaseChatModel):
        self._llm = llm

    @property
    def config(self):
        return self._config
