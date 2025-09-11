#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from pydantic import BaseModel, Field
from typing import Dict, Union, List
from enum import Enum

from jiuwen.core.context_engine.base import ContextVariable


class AsyncRunStrategy(Enum):
    ONCE = "once"
    LOOP = "loop"


class BaseProcessorConfig(BaseModel):
    processor_type: str = Field(default="")


class BaseAsyncProcessorConfig(BaseProcessorConfig):
    run_strategy: str = Field(default=AsyncRunStrategy.LOOP.value)
    run_interval: float = Field(default=0.5, gt=0.1)


class ContextEngineConfig(BaseModel):
    conversation_history_length: int = Field(default=20, ge=0)
    """variable config"""
    variables: List[ContextVariable] = Field(default=[])
    """online processing config"""
    processors: List[Union[BaseProcessorConfig, Dict]] = Field(default=[])
    """async processing config"""
    schedule_interval: str = Field(default=0.5, gt=0.1)
    async_processors: List[Union[BaseProcessorConfig, Dict]] = Field(default=[])
