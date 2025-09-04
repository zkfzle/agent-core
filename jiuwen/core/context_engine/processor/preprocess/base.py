#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import abstractmethod
from pydantic import BaseModel

from jiuwen.core.context_engine.base import EngineInput, EngineOutput
from jiuwen.core.context_engine.config import BaseProcessorConfig
from jiuwen.core.context_engine.processor.base import ProcessStage


class PreprocessStage(ProcessStage):
    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def run(self, input_data: EngineInput) -> EngineOutput:
        pass


