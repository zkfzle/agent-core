#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from enum import Enum

from pydantic import BaseModel, Field
from typing import Union, Dict, Any, Optional, List

from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.utils.prompt.template.template import Template

class ProcessorType(Enum):
    COMPRESSOR = "compressor"
    ASSEMBLER = "assembler"


class ProcessStage(Enum):
    PREPROCESSING = "preprocessing"
    ASSEMBLING = "assembling"
    POSTPROCESSING = "postprocessing"
    ASYNC_PROCESSING = "async_processing"


class AsyncUpdateType(Enum):
    UPDATE_NOTHING = "update_nothing"


class ContextType(Enum):
    USER_INPUT = "user_input"
    SYSTEM_PROMPT = "system_prompt"
    VARIABLES = "variables"
    USER_VARIABLES = "user_variables"
    CHAT_HISTORY = "chat_history"
    MEMORY = "memory"
    TOOLS = "tools"


class EngineInput(BaseModel):
    user_input: Union[str, Dict] = Field(default="")
    system_prompt: Union[str, Template] = Field(default="")
    variables: Dict[str, Any] = Field(default={})
    user_variables: Dict[str, Dict] = Field(default={})
    chat_history: Union[str, List[BaseMessage]] = Field(default="")
    memory: Optional[Any] = Field(default=None)
    tools: Union[str, Dict] = Field(default="")


class EngineOutput(EngineInput):
    full_output: str = Field(default="")

    @classmethod
    def from_input(cls, input: EngineInput) -> "EngineOutput":
        if isinstance(input, EngineOutput):
            return input
        return cls(**input.model_dump())