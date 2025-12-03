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
    async def add(self, memory: BaseMemoryUnit):
        """async add memory."""
        pass

    @abstractmethod
    async def update(self, user_id: str, group_id: str, mem_id: str, new_memory: str, **kwargs):
        """async update memory by its id."""
        pass

    @abstractmethod
    async def delete(self, user_id: str, group_id: str, mem_id: str, **kwargs):
        """async delete memory by its id."""
        pass

    @abstractmethod
    async def delete_by_user_id(self, user_id: str, group_id: str):
        """async delete memory by user id and app id."""
        pass

    @abstractmethod
    async def get(self, user_id: str, group_id: str, mem_id: str) -> dict[str, Any] | None:
        """async get memory by its id."""
        pass

    @abstractmethod
    async def search(self, user_id: str, group_id: str, query: str, top_k: int, **kwargs):
        """async query memory, return top k results"""
        pass