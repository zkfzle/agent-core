#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Any

from openjiuwen.core.memory.manage.base_memory_manager import BaseMemoryManager
from openjiuwen.core.memory.mem_unit.memory_unit import BaseMemoryUnit
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.store.user_mem_store import UserMemStore


class WriteManager:
    def __init__(self, managers: dict[str, BaseMemoryManager], memStore: UserMemStore):
        self.managers = managers
        self.memStore = memStore

    def add_mem(self, mem_units: list[BaseMemoryUnit]):
        for mem_unit in mem_units:
            mem_type = mem_unit.mem_type.value
            if mem_type in self.managers:
                self.managers[mem_type].add(mem_unit)
            else:
                logger.warning(f"Unsupported memory type: {mem_type}")

    def update_mem_by_id(self, mem_id: str, memory: str):
        context_data = self.__get_context_data_from_db(mem_id)
        if context_data is None:
            logger.error(f"Failed to update mem for mem_id: {mem_id}, because get context data failed")
            return
        self.managers[context_data["mem_type"]].update(mem_id, memory)

    def delete_mem_by_id(self, mem_id: str):
        context_data = self.__get_context_data_from_db(mem_id)
        if context_data is None:
            logger.error(f"Failed to delete mem for mem_id: {mem_id}, because get context data failed")
            return
        self.managers[context_data["mem_type"]].delete(mem_id)

    def delete_mem_by_user_id(self, user_id: str, app_id: str):
        for manager in self.managers:
            self.managers[manager].delete_by_user_id(user_id=user_id, app_id=app_id)

    def __get_context_data_from_db(self, mem_id: str) -> dict[str, Any] | None:
        context_data = None
        try:
            context_data = self.memStore.get_by_id(mem_id=mem_id)
        except Exception as e:
            logger.error(f"Failed to get context memory: {e}")
            return None
        if context_data is None:
            logger.error(f"Failed to get context memory from db store for mem id: {mem_id}")
            return None
        if "mem_type" not in context_data or "user_id" not in context_data or "app_id" not in context_data:
            logger.error(f"mem_type|user_id|app_id must exist in db store for mem id: {mem_id}")
            return None
        if context_data['mem_type'] not in self.managers:
            logger.error(f"Unsupported memory type: {context_data['mem_type']} for mem id {mem_id}")
            return None
        return context_data