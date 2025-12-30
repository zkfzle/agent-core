#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from datetime import datetime, timedelta
from typing import List

from openjiuwen.core.memory.store.base_kv_store import BaseKVStore


class MockKVStore(BaseKVStore):
    def __init__(self):
        self.store: dict[str, str] = {}
        self.expire: dict[str, datetime] = {}

    async def set(self, key: str, value: str):
        self.store[key] = value

    async def exclusive_set(self, key: str, value: str, expiry: int | None = None) -> bool:
        if key in self.store and (key not in self.expire or self.expire[key] > datetime.now()):
            return False
        self.store[key] = value
        if expiry:
            self.expire[key] = datetime.now() + timedelta(seconds=expiry)
        return True

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def exists(self, key: str) -> bool:
        return key in self.store

    async def delete(self, key: str):
        if key in self.store:
            del self.store[key]

    async def get_by_prefix(self, prefix: str) -> dict[str, str]:
        return {k: v for k, v in self.store.items() if k.startswith(prefix)}

    async def delete_by_prefix(self, prefix: str):
        del_keys = [k for k in self.store if k.startswith(prefix)]
        for k in del_keys:
            del self.store[k]

    async def mget(self, keys: List[str]) -> List[str | None]:
        return [self.store.get(k) for k in keys]
