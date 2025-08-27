#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from jiuwen.core.context_engine.config import AsyncExecuteConfig


class AsyncProcessPipeline:
    def __init__(self, config: AsyncExecuteConfig):
        self.__config = config