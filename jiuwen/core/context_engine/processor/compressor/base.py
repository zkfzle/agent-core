#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from pydantic import Field

from jiuwen.core.context_engine.processor.base import BaseProcessor
from jiuwen.core.context_engine.config import BaseProcessorConfig


class CompressorConfig(BaseProcessorConfig):
    compress_ratio: float = Field(default=0.8, ge=0, le=1.0)


class BaseCompressor(BaseProcessor):
    def __init__(self, config: CompressorConfig):
        super(BaseCompressor, self).__init__(config)