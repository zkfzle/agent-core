#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
import re
import struct
from typing import Any
from openjiuwen.core.memory.store.base_kv_store import BaseKVStore
from openjiuwen.core.memory.common.read_write_lock import ReadWriteLock
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.mem_unit.memory_unit import MemoryType


class UserMemStore:

    HEX_NUM_PER_INT: int = 8
    IDS_STR: str = "ids"
    USER_PROFILE_TOPIC_STR: str = "UPT"
    KEY_PREFIX_STR = "UMD"
    MEM_TYPE_FIELD_KEY: str = "mem_type"
    TOPIC_FIELD_KEY: str = "profile_type"
    SEPARATOR = "\x1F"

    def __init__(self, kv_store_instance: BaseKVStore):
        if kv_store_instance is None:
            raise ValueError("store instance is None in UserMemStore")
        self.kv_store = kv_store_instance
        self._lock = ReadWriteLock()

    def write(self, user_id: str, app_id: str, mem_id: str, data: dict[str, Any]) -> bool:
        """write data to store"""
        if not data:
            logger.error(f"write failed, because data is empty")
            return False
        user_mem_key = self.__get_user_mem_key(user_id, app_id, mem_id)
        with self._lock.write_lock():
            if self.kv_store.exists(user_mem_key):
                logger.error(f"write failed, user memory already exists for user_id={user_id}, app_id={app_id}, "
                             f"mem_id={mem_id}")
                return False
            if UserMemStore.MEM_TYPE_FIELD_KEY in data.keys():
                user_mem_ids_key = self.__get_user_ids_key(user_id, app_id, data[UserMemStore.MEM_TYPE_FIELD_KEY])
                user_mem_ids_value = self.kv_store.get(user_mem_ids_key, "")
                self.kv_store.set(user_mem_ids_key, self.__write_int(user_mem_ids_value, int(mem_id)))

                #Append id to user profile topic ids
                if (data[UserMemStore.MEM_TYPE_FIELD_KEY] == MemoryType.USER_PROFILE.value and
                    UserMemStore.TOPIC_FIELD_KEY in data.keys() and
                    data[UserMemStore.TOPIC_FIELD_KEY] is not None):
                    user_mem_topic_key = self.__get_concatenation_key([user_id, app_id,
                                                                       UserMemStore.USER_PROFILE_TOPIC_STR,
                                                                       data[UserMemStore.TOPIC_FIELD_KEY]])
                    user_mem_topic_value = self.kv_store.get(user_mem_topic_key, "")
                    self.kv_store.set(user_mem_topic_key, self.__write_int(user_mem_topic_value, int(mem_id)))

            # Append id to user ids
            user_ids_key = self.__get_user_ids_key(user_id, app_id)
            user_ids_value = self.kv_store.get(user_ids_key, "")
            self.kv_store.set(user_ids_key, self.__write_int(user_ids_value, int(mem_id)))

            # Set user mem id
            self.kv_store.set(user_mem_key, json.dumps(data))
            return True

    def update(self, user_id: str, app_id: str, mem_id: str, data: dict[str, Any]) -> bool:
        """update the data of given id"""
        user_mem_key = self.__get_user_mem_key(user_id, app_id, mem_id)
        with self._lock.write_lock():
            if not self.kv_store.exists(user_mem_key):
                logger.error(f"update failed, user memory does not exists for user_id={user_id}, app_id={app_id}, "
                             f"mem_id={mem_id}")
                return False
            old_data = self.kv_store.get(user_mem_key, "")
            if not old_data:
                self.kv_store.set(user_mem_key, json.dumps(data))
                return True
            dict_value = json.loads(old_data)
            for new_key, new_value in data.items():
                dict_value[new_key] = new_value
            self.kv_store.set(user_mem_key, json.dumps(dict_value))
            return True
    
    def delete(self, user_id: str, app_id: str, mem_id: str):
        """delete data by given id"""
        user_mem_key = self.__get_user_mem_key(user_id, app_id, mem_id)
        with self._lock.write_lock():
            if not self.kv_store.exists(user_mem_key):
                logger.warning(f"delete failed, user memory does not exists for user_id={user_id}, app_id={app_id}, "
                               f"mem_id={mem_id}")
                return True
            data = self.kv_store.get(user_mem_key, "")
            if data:
                # Delete user mem_type ids
                dict_value = json.loads(data)
                if UserMemStore.MEM_TYPE_FIELD_KEY in dict_value:
                    user_mem_ids_key = self.__get_user_ids_key(user_id, app_id,
                                                               dict_value[UserMemStore.MEM_TYPE_FIELD_KEY])
                    self.__delete_mem_id(user_mem_ids_key, mem_id)

                    #Delete user profile topic ids
                    if (dict_value[UserMemStore.MEM_TYPE_FIELD_KEY] == MemoryType.USER_PROFILE.value and
                        UserMemStore.TOPIC_FIELD_KEY in dict_value and
                        dict_value[UserMemStore.TOPIC_FIELD_KEY] is not None):
                        user_mem_topic_key = self.__get_concatenation_key([user_id, app_id,
                                                                           UserMemStore.USER_PROFILE_TOPIC_STR,
                                                                           dict_value[UserMemStore.TOPIC_FIELD_KEY]])
                        self.__delete_mem_id(user_mem_topic_key, mem_id)
            
            # Delete user ids
            user_ids_key = self.__get_user_ids_key(user_id, app_id)
            self.__delete_mem_id(user_ids_key, mem_id)

            # Delete user mem
            self.kv_store.delete(user_mem_key)
            return True
    
    def delete_by_user(self, user_id: str, app_id: str) -> bool:
        """delete all data under the given user_id and app_id"""
        delete_regex_key = self.__get_concatenation_key([re.escape(user_id), re.escape(app_id), ".*"])
        with self._lock.write_lock():
            self.kv_store.delete_by_regex(delete_regex_key)
        return True
    
    def get(self, user_id: str, app_id: str, mem_id: str) -> dict[str, Any] | None:
        """get data from given id"""
        user_mem_key = self.__get_user_mem_key(user_id, app_id, mem_id)
        with self._lock.read_lock():
            return self.__get(user_mem_key)
    
    def batch_get(self, user_id: str, app_id: str, mem_ids: list[str]) -> list[dict[str, Any]] | None:
        """get data from given ids"""
        with self._lock.read_lock():
            return [self.__get(self.__get_user_mem_key(user_id, app_id, mem_id)) for mem_id in mem_ids]
    
    def get_all(self, user_id: str, app_id: str, mem_type: str = None) -> list[dict[str, Any]] | None:
        """get data from given user_id|app_id|mem_type"""
        user_ids_key = self.__get_user_ids_key(user_id, app_id, mem_type)
        with self._lock.read_lock():
            if not self.kv_store.exists(user_ids_key):
                return None
            user_ids_value = self.kv_store.get(user_ids_key, "")
            if not user_ids_value:
                return None
            all_ids = self.__get_all_ints(user_ids_value)
            mem_ids = [str(mem_id) for mem_id in all_ids]
            return self.batch_get(user_id, app_id, mem_ids)
    
    def get_by_topic(self, user_id: str, app_id: str, topic: str) -> list[dict[str, Any]] | None:
        """get data from given user_id|app_id|topic"""
        user_mem_topic_key = self.__get_concatenation_key([user_id, app_id, UserMemStore.USER_PROFILE_TOPIC_STR, topic])
        with self._lock.read_lock():
            if not self.kv_store.exists(user_mem_topic_key):
                return None
            user_mem_topic_value = self.kv_store.get(user_mem_topic_key, "")
            if not user_mem_topic_value:
                return None
            all_ids = self.__get_all_ints(user_mem_topic_value)
            mem_ids = [str(mem_id) for mem_id in all_ids]
            return self.batch_get(user_id, app_id, mem_ids)

    def get_by_id(self, mem_id: str) -> dict[str, Any] | None:
        """get data from given id"""
        regex_id_key = self.__get_concatenation_key([rf"[^{self.SEPARATOR}]+",
                                                     rf"[^{self.SEPARATOR}]+",
                                                     re.escape(mem_id)])
        with self._lock.read_lock():
            mem_value = self.kv_store.get_by_regex(regex_id_key)
            if len(mem_value) != 1:
                logger.error(f"Failed to get correct number, result len is: {len(mem_value)}")
                return None
            return json.loads(list(mem_value.values())[0])

    def __get_user_ids_key(self, user_id: str, app_id: str, mem_type: str = None) -> str:
        if mem_type is None:
            return self.__get_concatenation_key([user_id, app_id, self.IDS_STR])
        else:
            return self.__get_concatenation_key([user_id, app_id, mem_type, self.IDS_STR])

    def __get_user_mem_key(self, user_id: str, app_id: str, mem_id: str) -> str:
        return self.__get_concatenation_key([user_id, app_id, mem_id])

    def __get_concatenation_key(self, fields: list[str]) -> str:
        key_str = UserMemStore.KEY_PREFIX_STR
        for field in fields:
            key_str += f"{self.SEPARATOR}{field}"
        return key_str

    def __delete_mem_id(self, ids_key: str, mem_id: str):
        if self.kv_store.exists(ids_key):
            ids_value = self.kv_store.get(ids_key, "")
            new_ids_value = self.__delete_int_by_value(ids_value, int(mem_id))
            if new_ids_value != "":
                self.kv_store.set(ids_key, new_ids_value)
            else:
                self.kv_store.delete(ids_key)

    def __get(self, mem_key: str):
        mem_value = self.kv_store.get(mem_key, "")
        if not mem_value:
            return None
        return json.loads(mem_value)

    def __write_int(self, data_list: str, num: int) -> str:
        """append an integer to data_list"""
        return data_list + struct.pack('i', num).hex()

    def __delete_int_by_value(self, data_list: str, value: int) -> str:
        """delete the integer by value."""
        total = len(data_list) // self.HEX_NUM_PER_INT
        for i in range(total):
            bytes_chunk = bytes.fromhex(data_list[i * self.HEX_NUM_PER_INT:(i+1) * self.HEX_NUM_PER_INT])
            num = struct.unpack('i', bytes_chunk)[0]
            if num == value:
                return data_list[:i * self.HEX_NUM_PER_INT] + data_list[(i+1) * self.HEX_NUM_PER_INT:]
        return data_list

    def __get_all_ints(self, data_list: str) -> list[int]:
        """return all integers in data_list."""
        ints = []
        total = len(data_list) // self.HEX_NUM_PER_INT
        for i in range(total):
            bytes_chunk = bytes.fromhex(data_list[i * self.HEX_NUM_PER_INT:(i+1) * self.HEX_NUM_PER_INT])
            value = struct.unpack('i', bytes_chunk)[0]
            ints.append(value)
        return ints

    def __delete_int_by_idx(self, data_list: str, idx: int) -> str:
        total = len(data_list) // self.HEX_NUM_PER_INT
        if 0 <= idx < total:
            return data_list[:idx * self.HEX_NUM_PER_INT] + data_list[(idx+1) * self.HEX_NUM_PER_INT:]
        return data_list

    def __get_int_by_idx(self, data_list: str, idx: int) -> int | None:
        total = len(data_list) // self.HEX_NUM_PER_INT
        if 0 <= idx < total:
            bytes_chunk = bytes.fromhex(data_list[idx * self.HEX_NUM_PER_INT:(idx+1) * self.HEX_NUM_PER_INT])
            return struct.unpack('i', bytes_chunk)[0]
        return None
