#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Any

from openjiuwen.core.memory.manage.base_memory_manager import BaseMemoryManager
from openjiuwen.core.memory.mem_unit.memory_unit import BaseMemoryUnit
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.store.user_mem_store import UserMemStore


class WriteManager:
    def __init__(self, managers: dict[str, BaseMemoryManager], mem_store: UserMemStore):
        self.managers = managers
        self.mem_store = mem_store

    def add_mem(self, mem_units: list[BaseMemoryUnit]):
        for mem_unit in mem_units:
            mem_type = mem_unit.mem_type.value
            if mem_type in self.managers:
                self.managers[mem_type].add(mem_unit)
            else:
                logger.warning(f"Unsupported memory type: {mem_type}")

    def update_mem_by_id(self, mem_id: str, memory: str):
        mem_type = self.__get_mem_type_from_store(mem_id)
        if mem_type is None:
            logger.error(f"Failed to update mem for mem_id: {mem_id}, because get memory type failed")
            return
        self.managers[mem_type].update(mem_id, memory)

    def delete_mem_by_id(self, mem_id: str):
        mem_type = self.__get_mem_type_from_store(mem_id)
        if mem_type is None:
            logger.error(f"Failed to delete mem for mem_id: {mem_id}, because get memory type failed")
            return
        self.managers[mem_type].delete(mem_id)

    def delete_mem_by_user_id(self, user_id: str, app_id: str):
        for manager in self.managers:
            self.managers[manager].delete_by_user_id(user_id=user_id, app_id=app_id)

    def __get_mem_type_from_store(self, mem_id: str) -> str | None:
        data = None
        try:
            data = self.mem_store.get_by_id(mem_id=mem_id)
        except Exception as e:
            logger.error(f"Failed to get memory: {e}")
            return None
        if data is None:
            logger.error(f"Failed to get memory from store for mem id: {mem_id}")
            return None
        if "mem_type" not in data:
            logger.error(f"mem_type must exist in store for mem id: {mem_id}")
            return None
        if data['mem_type'] not in self.managers:
            logger.error(f"Unsupported memory type: {data['mem_type']} for mem id {mem_id}")
            return None
        return data['mem_type']