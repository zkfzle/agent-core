#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from pydantic import BaseModel, Field, field_validator
from openjiuwen.core.memory.common.crypto import IV_LENGTH


class SysMemConfig(BaseModel):
    record_message: bool = Field(default=True)  # record message or not
    ai_msg_gen_max_len: int = Field(default=64)  # max length of AI message generation memory
    history_window_size_to_gen_mem: int = Field(default=5)  # history window size to generate memory
    crypto_key: str = Field(default="")  # aes key, utf-8 bytes length must be 16, not enable encrypto memory if empty

    @field_validator('crypto_key')
    @classmethod
    def check_crypto_key(cls, v: str) -> str:
        if v == "":
            return ""

        if len(v.encode(encoding="utf-8")) == IV_LENGTH:
            return v

        raise ValueError(f"Invalid crypto_key, must be empty or {IV_LENGTH} bytes(utf-8 length)")


class MemoryConfig(BaseModel):
    mem_variables: dict[str, str] = Field(default_factory=dict)  # memory variables config
    enable_long_term_mem: bool = Field(default=True)  # enable long term memory or not
