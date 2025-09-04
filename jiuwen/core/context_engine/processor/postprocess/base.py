#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import abstractmethod, ABC

from jiuwen.core.context_engine.base import EngineInput, EngineOutput
from jiuwen.core.context_engine.processor.base import ProcessStage


class PostprocessStage(ProcessStage):
    @abstractmethod
    def run(self, input_data: EngineInput) -> EngineOutput:
        pass
