#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from pydantic import BaseModel, Field


class SysMemConfig(BaseModel):
    record_message: bool = Field(default=True)  # record message or not
    ai_msg_gen_max_len: int = Field(default=64)  # max length of AI message generation memory
    history_window_size_to_gen_mem: int = Field(default=5)  # history window size to generate memory


class MemoryConfig(BaseModel):
    mem_variables: dict[str, str] = Field(default_factory=dict)  # memory variables config
    enable_long_term_mem: bool = Field(default=True)  # enable long term memory or not
