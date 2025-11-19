#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from abc import abstractmethod, ABC
from typing import Any

from openjiuwen.core.memory.mem_unit.memory_unit import BaseMemoryUnit


class BaseMemoryManager(ABC):
    """
    Simplified abstract base class for memory manager implementations.
    Managing a specific type of memory data.
    """

    @abstractmethod
    def add(self, memory: BaseMemoryUnit):
        """add memory."""
        pass

    @abstractmethod
    def update(self, mem_id: str, new_memory: str, **kwargs):
        """update memory by its id."""
        pass

    @abstractmethod
    def delete(self, mem_id: str, **kwargs):
        """delete memory by its id."""
        pass

    @abstractmethod
    def delete_by_user_id(self, user_id: str, app_id: str):
        """delete memory by user id and app id."""
        pass

    @abstractmethod
    def get(self, mem_id: str) -> dict[str, Any] | None:
        """get memory by its id."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int, **kwargs):
        """query memory, return top k results"""
        pass