#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from pydantic import BaseModel, Field
from typing import Dict, Union, Optional, List

from jiuwen.core.utils.llm.base import BaseModelInfo
from jiuwen.core.context_engine.base import AsyncUpdateType

class BaseProcessorConfig(BaseModel):
    processor_type: str = Field(default="")


class BaseAsyncProcessorConfig(BaseProcessorConfig):
    output_type: str = Field(default=AsyncUpdateType.UPDATE_NOTHING.value)


class AsyncExecuteConfig(BaseModel):
    processors: List[Union[BaseProcessorConfig, Dict]] = Field(default=[])


class OnlineExecuteConfig(BaseModel):
    preprocess_stage: List[Union[BaseProcessorConfig, Dict]] = Field(default=[])
    assemble_stage: List[Union[BaseProcessorConfig, Dict]] = Field(default=[])
    postprocess_stage: List[Union[BaseProcessorConfig, Dict]] = Field(default=[])


class ContextEngineConfig(BaseModel):
    node_id: str
    model_provider: Optional[str] = Field(default=None)
    model_info: Optional[BaseModelInfo] = Field(default=None)
    online_process: OnlineExecuteConfig = Field(default=OnlineExecuteConfig())
    async_process: AsyncExecuteConfig = Field(default=AsyncExecuteConfig())
