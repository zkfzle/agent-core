#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from enum import Enum

from pydantic import BaseModel, Field
from typing import Union, Dict, Any, Optional

from jiuwen.core.utils.prompt.template.template import Template

class ProcessorType(Enum):
    COMPRESSOR = "compressor"
    ASSEMBLER = "assembler"


class EngineInput(BaseModel):
    user_input: Union[str, Dict] = Field(default="")
    system_prompt: Union[str, Template] = Field(default="")
    variables: Dict[str, Dict] = Field(default={})
    chat_history: Union[str, Dict] = Field(default="")
    memory: Optional[Any] = Field(default=None)
    tools: Union[str, Dict] = Field(default="")


class EngineOutput(BaseModel):
    output: Union[str, Dict] = Field(default="")


class ProcessorInput(BaseModel):
    engine_input: EngineInput


class ProcessorOutput(BaseModel):
    last_processer_type: ProcessorType
    compressed_context: Dict[str, Dict] = Field(default={})
    variables: Dict[str, Dict] = Field(default={})