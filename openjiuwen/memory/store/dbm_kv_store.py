import dbm
import re
from functools import lru_cache
from typing import List, Any
from openjiuwen.core.memory.store.base_kv_store import BaseKVStore

class DbmKVStore(BaseKVStore):
    """
    KV store based on dbm + lru_cache
    """
    def __init__(self, filename: str, cache_size: int = 256):
        self.db = dbm.open(filename, "c")
        self._cached_get = lru_cache(maxsize=cache_size)(self._raw_get)

    def _raw_get(self, key: str):
        key_b = key.encode("utf-8")
        return self.db.get(key_b, None)

    def set(self, key: str, value: str):
        self.db[key.encode()] = value.encode()
        self._cached_get.cache_clear()

    def get(self, key: str, default: Any = None) -> str:
        v = self._cached_get(key)
        if v is None:
            return default
        return v.decode("utf-8")

    def exists(self, key: str) -> bool:
        return self._cached_get(key) is not None

    def delete(self, key: str):
        key_b = key.encode()
        if key_b in self.db:
            del self.db[key_b]
            self._cached_get.cache_clear()

    def get_by_regex(self, regex_str: str):
        pat = re.compile(regex_str)
        result = {}
        for key_b in self.db.keys():
            k = key_b.decode()
            if pat.search(k):
                result[k] = self.db[key_b].decode()
        return result

    def delete_by_regex(self, regex_str: str):
        pat = re.compile(regex_str)
        delete_keys = []
        for key_b in self.db.keys():
            k = key_b.decode()
            if pat.search(k):
                delete_keys.append(key_b)
        if delete_keys:
            for key_b in delete_keys:
                del self.db[key_b]
            self._cached_get.cache_clear()

    def mget(self, keys: List[str], default: Any = None) -> List[str]:
        result = []
        for k in keys:
            v = self._cached_get(k)
            result.append(v.decode() if v else default)
        return result

    def close(self):
        self.db.close()
