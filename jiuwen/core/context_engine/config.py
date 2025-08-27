#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from pydantic import BaseModel, Field
from typing import Dict, List, Union


class BaseProcessorConfig(BaseModel):
    processor_type: str = Field(default="")


class AsyncExecuteConfig(BaseModel):
    processors: List[Union[BaseProcessorConfig, Dict]] = Field(default=[])


class OnlineExecuteConfig(BaseModel):
    processors: List[Union[BaseProcessorConfig, Dict]] = Field(default=[])


class ContextEngineConfig(BaseModel):
    node_id: str
    online_process: OnlineExecuteConfig = Field(default=OnlineExecuteConfig())
    async_process: AsyncExecuteConfig = Field(default=AsyncExecuteConfig())
