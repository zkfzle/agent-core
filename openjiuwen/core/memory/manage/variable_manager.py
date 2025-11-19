#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from re import escape
from typing import Any, Optional

from openjiuwen.core.memory.manage.base_memory_manager import BaseMemoryManager
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.mem_unit.memory_unit import VariableUnit
from openjiuwen.core.memory.store.base_kv_store import BaseKVStore


class VariableManager(BaseMemoryManager):
    SEPARATOR = "\x1F"
    def __init__(self, kv_store: BaseKVStore):
        self.kv_store = kv_store

    def add(self, memory: VariableUnit):
        """add Variable memory"""
        if self.kv_store is None:
            logger.error("kv_store cannot be None")
        key, value = self._make_variable_pairs(
            memory.user_id,
            False,
            memory.app_id,
            memory.variable_name,
            None,
            memory.variable_mem,
            None
        )
        self.kv_store.set(key, value)

    def update(self, mem_id: str, new_memory: str, **kwargs):
        pass

    def update_user_variable(self, user_id: str, app_id: str, var_name: str, var_mem: str):
        if self.kv_store is None:
            logger.error("kv_store cannot be None")
        key, value = self._make_variable_pairs(usr_id=user_id, for_deletion=False,
                                app_id=app_id, var_name=var_name, user_var_value=var_mem)
        self.kv_store.set(key, value)

    def delete(self, mem_id: str, **kwargs):
        pass

    def delete_by_user_id(self, user_id: str, app_id: str):
        if self.kv_store is None:
            logger.error("kv_store cannot be None")
        user_key = f"^user_var{self.SEPARATOR}{escape(user_id)}{self.SEPARATOR}{escape(app_id)}{self.SEPARATOR}.*$"
        session_key = f"^session_var{self.SEPARATOR}{escape(user_id)}{self.SEPARATOR}{escape(app_id)}{self.SEPARATOR}.*$"
        self.kv_store.delete_by_regex(user_key)
        self.kv_store.delete_by_regex(session_key)

    def delete_user_variable(self, user_id: str, app_id: str, var_name: str):
        if self.kv_store is None:
            logger.error("kv_store cannot be None")
        key, _ = self._make_variable_pairs(usr_id=user_id, for_deletion=False,
                                app_id=app_id, var_name=var_name)
        self.kv_store.delete(key)

    def get(self, mem_id: str) -> dict[str, Any] | None:
        pass

    def search(self, query: str, top_k: int, **kwargs):
        pass

    def query_variable(self, user_id: str, app_id: str, name:Optional[str] = None,
                       session_id: Optional[str] = None) -> dict[str, Any]:
        """query variable by user_id, app_id, variable_name return variable mem."""
        self._check_user_and_app_id(user_id, app_id, "Search")
        if not name or not name.strip():
            regex_str = f"^user_var{self.SEPARATOR}{escape(user_id)}{self.SEPARATOR}{escape(app_id)}{self.SEPARATOR}.*$"
            return {k.split(f"{self.SEPARATOR}")[-1]: v for k, v in self.kv_store.get_by_regex(regex_str).items()}
        if session_id:
            key = f"session_var{self.SEPARATOR}{user_id}{self.SEPARATOR}{app_id}{self.SEPARATOR}{session_id}{self.SEPARATOR}{name}"
        else:
            key = f"user_var{self.SEPARATOR}{user_id}{self.SEPARATOR}{app_id}{self.SEPARATOR}{name}"
        return {name: self.kv_store.get(key)}

    @staticmethod
    def _make_variable_pairs(
            usr_id: str,
            for_deletion: bool,
            app_id: str,
            var_name: Optional[str] = None,
            session_id: Optional[str] = None,
            user_var_value: Optional[Any] = None,
            session_var_value: Optional[Any] = None
    ) -> (str, str):
        key, value = "", ""
        if var_name is not None:
            # 1) user_var
            if session_id is None:
                key = (
                    f"user_var{VariableManager.SEPARATOR}{usr_id}"
                    f"{VariableManager.SEPARATOR}{app_id}"
                    f"{VariableManager.SEPARATOR}{var_name}"
                )
                value = None if for_deletion else user_var_value
            # 2# session_var
            else:
                key = (
                    f"session_var{VariableManager.SEPARATOR}{usr_id}"
                    f"{VariableManager.SEPARATOR}{app_id}"
                    f"{VariableManager.SEPARATOR}{session_id}"
                    f"{VariableManager.SEPARATOR}{var_name}"
                )
                value = None if for_deletion else session_var_value
        return key, value

    @staticmethod
    def _check_user_and_app_id(user_id, app_id, context="Operation"):
        if not user_id or not user_id.strip():
            logger.error(f"{context} failed, user ID is empty")
        if not app_id or not app_id.strip():
            logger.error(f"{context} failed, app ID is empty")