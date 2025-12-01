#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from sqlalchemy.ext.asyncio import AsyncEngine
from openjiuwen.core.memory.store.base_db_store import BaseDbStore


class DefaultDbStore(BaseDbStore):
    def __init__(self, async_engine: AsyncEngine):
        self.async_engine = async_engine

    def get_async_engine(self) -> AsyncEngine:
        return self.async_engine
