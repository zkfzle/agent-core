#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from pydantic import BaseModel, Field, field_validator
from openjiuwen.core.memory.common.crypto import AES_KEY_LENGTH


class SysMemConfig(BaseModel):
    record_message: bool = Field(default=True)  # record message or not
    ai_msg_gen_max_len: int = Field(default=256)  # max length of AI message generation memory
    history_window_size_to_gen_mem: int = Field(default=5)  # history window size to generate memory
    history_summary_to_gen_mem: bool = Field(default=True) # history summary to generate memory or not
    crypto_key: bytes = Field(default=b'')  # aes key, length must be 32, not enable encrypt memory if empty

    @field_validator('crypto_key')
    @classmethod
    def check_crypto_key(cls, v: bytes) -> bytes:
        if len(v) == 0:
            return b''

        if len(v) == AES_KEY_LENGTH:
            return v

        raise ValueError(f"Invalid crypto_key, must be empty or {AES_KEY_LENGTH} bytes length")


class SummaryConfig(BaseModel):
    max_token: int = Field(default=128)
    fraction: float = Field(default=0.3)
    threshold: int = Field(default=0)

    @field_validator('max_token')
    @classmethod
    def check_max_token(cls, v: int) -> int:
        if v > 0:
            return v
        raise ValueError(f"Invalid max_token, must be positive")

    @field_validator('fraction')
    @classmethod
    def check_fraction(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"Invalid fraction, must be between 0 and 1")

        return v

    @field_validator('threshold')
    @classmethod
    def check_threshold(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Invalid threshold, must be positive")
        return v


class MemoryConfig(BaseModel):
    mem_variables: dict[str, str] = Field(default_factory=dict)  # memory variables config
    enable_long_term_mem: bool = Field(default=True)  # enable long term memory or not
    enable_query_decompose: bool = Field(default=True)  # enable search with query decomposition or not
    summary_config: SummaryConfig = Field(default_factory=SummaryConfig)  # summary config
