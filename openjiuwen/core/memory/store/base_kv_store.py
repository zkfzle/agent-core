#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import List, Any

class BaseKVStore(ABC):

    @abstractmethod
    def set(self, key: str, value: str):
        pass

    @abstractmethod
    def get(self, key: str, default: Any = None) -> str:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    def delete(self, key: str):
        pass

    @abstractmethod
    def get_by_regex(self, regex_str: str):
        pass

    @abstractmethod
    def delete_by_regex(self, regex_str: str):
        pass

    @abstractmethod
    def mget(self, keys: List[str], default: Any = None) -> List[str]:
        pass