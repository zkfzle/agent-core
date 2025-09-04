#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import abstractmethod, ABC

from jiuwen.core.context_engine.config import BaseProcessorConfig


class ProcessStage(ABC):
    def __init__(self, config: BaseProcessorConfig):
        self._BaseProcessor__config = config

    def run(self, input_data):
        pass
