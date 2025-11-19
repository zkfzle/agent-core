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

class Config(BaseModel):
    temperature: float | None = 0.01
    top_p: float | None = 0.95
    timeout: float | None = 60.0
    variables_key: dict[str, list] = {}
    topic: list[str] | None = []
    model_api_base: str | None = ""
    model_api_key: str | None = ""
    model_name: str | None = ""
    model_provider: str | None = ""
    strategy: list[str] = []
    language: str | None = "en" # or "zh-CN"
    enable_long_term_mem: bool | None = False # generate long-term memory or not
    enable_session_summary: bool | None = False # generate session summary or not
    realtime_process_config: RealtimeConfig = Field(default_factory=RealtimeConfig)
