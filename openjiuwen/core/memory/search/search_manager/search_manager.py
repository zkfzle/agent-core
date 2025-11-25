#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Optional, Any

from openjiuwen.core.memory.manage.base_memory_manager import BaseMemoryManager
from openjiuwen.core.memory.manage.user_profile_manager import UserProfileManager
from openjiuwen.core.memory.manage.variable_manager import VariableManager
from openjiuwen.core.memory.mem_unit.memory_unit import MemoryType
from openjiuwen.core.memory.store.user_mem_store import UserMemStore


class SearchManager:
    user_mem_manager_list = [MemoryType.USER_PROFILE.value]
    all_mem_manager_list = [item.value for item in MemoryType]
    def __init__(self, managers: dict[str, BaseMemoryManager], user_mem_store: UserMemStore):
        self.managers = managers
        self.mem_store = user_mem_store

    def search(self, query: str, top_k: int = 5, threshold: float = 0.3, search_type: Optional[str] = None, **kwargs) \
        -> list[dict[str, Any]] | None:
        result = []
        if search_type is None:
            for mem_type, manager in self.managers.items():
                if mem_type in self.user_mem_manager_list:
                    res = manager.search(query=query, top_k=top_k, **kwargs)
                    if res is not None:
                        result.extend(res)
        elif search_type in self.all_mem_manager_list:
            if self.managers.get(search_type, None):
                res = self.managers[search_type].search(query=query, top_k=top_k, **kwargs)
                if res is not None:
                    result = res
            else:
                raise ValueError(f"{search_type} memory manager not inited")
        else:
            raise ValueError(f"{search_type} is not a valid search type")
        filtered_res = [item for item in result if item["score"] >= threshold]
        return filtered_res

    def list_user_mem(self, user_id: str, app_id: str, nums: int, pages: int) -> list[dict[str, Any]] | None:
        data = self.mem_store.get_all(user_id=user_id, app_id=app_id)
        if len(data) <= nums * (pages - 1):
            return None
        if len(data) > nums * pages:
            return data[(nums * (pages - 1)):(nums * pages)]
        return data[(nums * (pages - 1)):len(data)]

    def list_user_profile(self, user_id: str, app_id: str, profile_type: Optional[str] = None) -> list[dict]:
        if not isinstance(self.managers[MemoryType.USER_PROFILE.value], UserProfileManager):
            raise ValueError(f"{MemoryType.USER_PROFILE.value} manager class is not UserProfileManager")
        return self.managers[MemoryType.USER_PROFILE.value].list_user_profile(user_id=user_id,
                                                                           app_id=app_id, profile_type=profile_type)

    def get_user_variable(self, user_id: str, app_id: str, var_name: str) -> str | None:
        if not isinstance(self.managers[MemoryType.VARIABLE.value], VariableManager):
            raise ValueError(f"{MemoryType.VARIABLE.value} manager class is not VariableManager")
        res = self.managers[MemoryType.VARIABLE.value].query_variable(user_id=user_id,
                                                                    app_id=app_id, name=var_name)
        if res is None:
            return None
        return res[var_name]

    def get_session_variable(self, user_id: str, app_id: str, session_id: str, var_name: str) -> str | None:
        if not isinstance(self.managers[MemoryType.VARIABLE.value], VariableManager):
            raise ValueError(f"{MemoryType.VARIABLE.value} manager class is not VariableManager")
        res = self.managers[MemoryType.VARIABLE.value].query_variable(user_id=user_id, session_id = session_id,
                                                                    app_id=app_id, name=var_name)
        if res is None:
            return None
        return res[var_name]

    def get_all_user_variable(self, user_id: str, app_id: str) -> dict[str, Any]:
        if not isinstance(self.managers[MemoryType.VARIABLE.value], VariableManager):
            raise ValueError(f"{MemoryType.VARIABLE.value} manager class is not VariableManager")
        return self.managers[MemoryType.VARIABLE.value].query_variable(user_id=user_id, app_id=app_id)
