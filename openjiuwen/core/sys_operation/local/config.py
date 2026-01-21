#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from pydantic import BaseModel, Field


class LocalWorkConfig(BaseModel):
    """Local working configuration"""
    work_dir: str = Field(description="Local working directory path")
