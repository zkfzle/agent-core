#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import plyvel
from openjiuwen.core.memory.store.base_kv_store import BaseKVStore
from typing import Optional, List, Any
from openjiuwen.core.memory.config.config import Config
import re

class LeveldbKVStore(BaseKVStore):

    def __init__(self, config: Config):
        path = config.kv_store_dir
        self.db = plyvel.DB(path, create_if_missing=True,
                            lru_cache_size=16 * 1024 * 1024,
                            write_buffer_size=16 * 1024 * 1024,
                            bloom_filter_bits=10)

        def _enc(self, s: str) -> bytes:
            return s.encode("utf-8")

        def _dec(self, s: str) -> str | None:
            return None if s is None else s.decode("utf-8")

        def set(self, key: str, value: str):
            self.db.put(self._enc(key), self._enc(value))

        def get(self, key: str, default: Any = None) -> str:
            val = self.db.get(self._enc(key))
            if val is None:
                return default
            return self._dec(val)

        def exists(self, key: str) -> bool:
            return self.db.get(self._enc(key)) is not None

        def delete(self, key: str):
            self.db.delete(self._enc(key))

        def get_by_regex(self, regex_str: str) -> dict[str, Any]:
            pattern = re.compile(regex_str)
            result = {}
            for key_bytes, v_bytes in self.db:
                if not isinstance(key_bytes, (bytes, bytearray)):
                    continue
                try:
                    key_str = self._dec(key_bytes)
                except:
                    continue
                if pattern.search(key_str):
                    if v_bytes is None:
                        result[key_str] = None
                    try:
                        result[key_str] = self._dec(v_bytes)
                    except:
                        result[key_str] = v_bytes
            return result

        def delete_by_regex(self, regex_str: str):
            pattern = re.compile(regex_str)
            batch = self.db.write_batch()
            deleted = 0
            count = 0
            AUTO_BATCH = 1000

            for key_bytes, _ in self.db:
                if not isinstance(key_bytes, (bytes, bytearray)):
                    continue
                try:
                    key_str = self._dec(key_bytes)
                except:
                    continue
                if pattern.search(key_str):
                    batch.delete(key_bytes)
                    deleted += 1
                    count += 1

                    if count >= AUTO_BATCH:
                        batch.write()
                        batch = self.db.write_batch()
                        count = 0
            if count > 0:
                batch.write()

        def mget(self, keys: List[str], default: Any = None) -> List[str]:
            vals: List[Any] = []
            for k in keys:
                v = self.db.get(self._enc(k))
                vals.append(v if v is not None else default)
            return vals
