#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import asyncio
from openjiuwen.core.memory.store.base_kv_store import BaseKVStore


class DataIdManager:
    SEPARATOR = "/"
    ID_KEY = f'id_manager{SEPARATOR}next_id'

    def __init__(self, kv_store: BaseKVStore):
        self.kv_store = kv_store
        self.async_lock = asyncio.Lock()

    async def generate_next_id(self) -> int:
        """async generate a unique ID and store it in the KV"""
        async with self.async_lock:
            current_id = await self.kv_store.get(DataIdManager.ID_KEY)
            if not current_id:
                current_id = 0
            current_id = int(current_id)

            next_id = current_id + 1
            await self.kv_store.set(DataIdManager.ID_KEY, str(next_id))
            return current_id
