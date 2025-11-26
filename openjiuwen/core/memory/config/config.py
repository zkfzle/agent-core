#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from pydantic import BaseModel, Field

class MemoryConfig(BaseModel):
    mem_variables: dict[str, str] = Field(default_factory=dict)
    enable_long_term_mem: bool = Field(default=False)

class RealtimeConfig(BaseModel):
    window_size: int = 5
    user_profile_custom_define: dict[str, str] = {}
    history_message_length_limit: int = 50

class Config(BaseModel):
    variables_key: dict[str, list] = {}
    model_name: str | None = ""
    language: str | None = "en" # or "zh-CN"
    realtime_process_config: RealtimeConfig = Field(default_factory=RealtimeConfig)
