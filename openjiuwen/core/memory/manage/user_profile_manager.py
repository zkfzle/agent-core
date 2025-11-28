#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import threading
from datetime import datetime, timezone
from typing import Any, List, Optional

from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore
from openjiuwen.core.memory.common.base import parse_memory_hit_infos
from openjiuwen.core.memory.manage.base_memory_manager import BaseMemoryManager
from openjiuwen.core.memory.manage.data_id_manager import DataIdManager
from openjiuwen.core.memory.mem_unit.memory_unit import UserProfileUnit, MemoryType, ConflictType, BaseMemoryUnit
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.store.user_mem_store import UserMemStore


class UserProfileManager(BaseMemoryManager):
    def __init__(self, semantic_recall_instance: BaseSemanticStore,
                 user_mem_store: UserMemStore,
                 data_id_generator: DataIdManager):
        self.mem_store = user_mem_store
        self.semantic_recall = semantic_recall_instance
        self.date_user_profile_id = data_id_generator
        self.lock = threading.Lock()

    def add(self, memory: BaseMemoryUnit):
        if not isinstance(memory, UserProfileUnit):
            raise ValueError('user profile add Must pass UserProfileUnit class.')
        if not memory.user_id:
            raise ValueError('Must pass user_id')
        if not memory.app_id:
            raise ValueError('Must pass app_id')
        if not memory.profile_mem:
            raise ValueError('Must pass profile_mem')
        if not memory.profile_type:
            raise ValueError('Must profile_type')
        for conflict in memory.conflict_info:
            conf_id = conflict['id']
            conf_mem = conflict['text']
            conf_event = conflict['event']
            if not conf_mem or conf_mem == "":
                continue
            if conf_id == "-1" and conf_event == ConflictType.ADD.value:
                mem_id = self._add_user_profile_memory(user_id=memory.user_id,
                                                       app_id=memory.app_id,
                                                       profile_type=memory.profile_type,
                                                       profile_mem=conf_mem,
                                                       source_id=memory.message_mem_id)
                self._add_vector_user_profile_memory(user_id=memory.user_id,
                                                     app_id=memory.app_id,
                                                     memory_id=[mem_id],
                                                     mem=[conf_mem])
            elif conf_event == ConflictType.NONE.value:
                logger.info(f"none conflict info: {conflict}")
            elif conf_event ==ConflictType.UPDATE.value:
                logger.info(f"update conflict info: {conflict}, update: {memory.profile_mem}")
                self.update(conf_id, memory.profile_mem)
            elif conf_event == ConflictType.DELETE.value:
                logger.info(f"delete conflict info: {conflict}")
                self.delete(conf_id)

    def update(self, mem_id: str, new_memory: str, **kwargs) -> bool:
        data = self.mem_store.get_by_id(mem_id=mem_id)
        if data is None:
            logger.error("Update User_profile failed, the data corresponding to mem_id, does not exist")
            return False
        time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        new_data = {'mem': new_memory, 'time': time}
        new_data.update(kwargs)
        with self.lock:
            self.mem_store.update(mem_id=mem_id, user_id=data['user_id'], app_id=data['app_id'], data=new_data)
        self.semantic_recall.remove(ids=[mem_id], user_id=data['user_id'],
                                    app_id=data['app_id'], mem_type=MemoryType.USER_PROFILE.value)
        self.semantic_recall.add(mem=[new_memory], memory_id=[mem_id], user_id=data['user_id'],
                                    app_id=data['app_id'], mem_type=MemoryType.USER_PROFILE.value)
        return True

    def search(self, query: str, top_k: int, **kwargs):
        app_id = kwargs.get("app_id", "")
        user_id = kwargs.get("user_id", "")
        if app_id == "":
            raise ValueError("Must pass app_id")
        if user_id == "":
            raise ValueError("Must pass user_id")
        mem_type = kwargs.get("mem_type", MemoryType.USER_PROFILE.value)
        mem_ids, scores = self._recall_by_vector(query, user_id, app_id, top_k, mem_type)
        retrieve_res = self.mem_store.batch_get(user_id=user_id, app_id=app_id, mem_ids=mem_ids)
        if retrieve_res is None:
            return None
        for item in retrieve_res:
            item["score"] = scores.get(item['id'], 0)
        retrieve_res.sort(key=lambda x: scores.get(x["id"], 0), reverse=True)
        return retrieve_res

    def get(self, mem_id: str) -> dict[str, Any] | None:
        data = self.mem_store.get_by_id(mem_id=mem_id)
        return data

    def delete(self, mem_id: str, **kwargs):
        data = self.mem_store.get_by_id(mem_id=mem_id)
        if data is None:
            logger.error(f"Delete user_profile in db failed, the mem of mem_id({mem_id}) is not exist.")
            return False
        mem_type = kwargs.get("mem_type", MemoryType.USER_PROFILE.value)
        with self.lock:
            self.mem_store.delete(mem_id=mem_id, user_id=data['user_id'], app_id=data['app_id'])
            self._delete_vector_user_profile_memory(memory_id=[mem_id], user_id=data['user_id'],
                                                    app_id=data['app_id'], mem_type=mem_type)
        return True

    def delete_by_user_id(self, user_id: str, app_id: str):
        data = self.mem_store.get_all(user_id=user_id, app_id=app_id)
        if data is None:
            logger.error(f"Delete user_profile in db failed, the mem of user_id({user_id}) is not exist.")
            return False
        mem_ids = [item['id'] for item in data]
        with self.lock:
            self.mem_store.delete_by_user(user_id=user_id, app_id=app_id)
            self._delete_vector_user_profile_memory(memory_id=mem_ids, user_id=user_id,
                                                    app_id=app_id, mem_type=MemoryType.USER_PROFILE.value)
        return True

    def list_user_profile(self, user_id: str, app_id: str, profile_type: Optional[str] = None,
                          mem_type=MemoryType.USER_PROFILE) -> List[UserProfileUnit]:
        datas = self.mem_store.get_all(user_id=user_id, app_id=app_id, mem_type=mem_type.value)
        if not datas:
            logger.debug(f"End to get user profile, result is None, "
                         f"params user_id:{user_id}, app_id:{app_id}, mem_type:{mem_type}")
            return []
        new_datas = []
        if profile_type is not None:
            for data in datas:
                if data['profile_type'] == profile_type:
                    new_datas.append(data)
        else:
            new_datas = datas
        new_datas.sort(key=lambda x: (x['mem'], x['timestamp']), reverse=True)
        return new_datas

    def _recall_by_vector(self, query: str, user_id: str, app_id: str, top_k: int = 5,
                          mem_type=MemoryType.USER_PROFILE.value) -> tuple[List[str], dict[str, float]]:
        """向量召回"""
        memory_hit_info = self.semantic_recall.search(query=[query], user_id=user_id, app_id=app_id,
                                                      mem_type=mem_type, top_k=top_k)
        return parse_memory_hit_infos(memory_hit_info)

    def _add_user_profile_memory(
            self,
            is_implicit: bool = False,
            mem_type: str = MemoryType.USER_PROFILE.value,
            user_id: Optional[str] = None,
            app_id: Optional[str] = None,
            profile_type: Optional[str] = None,
            profile_mem: Optional[str] = None,
            source_id: Optional[str] = None,
            reasoning: Optional[str] = None,
            context_summary: Optional[str] = ""
    ) -> str:
        """
        向用户画像表中添加数据
        """
        if not user_id:
            raise ValueError('Must pass user_id')
        if not app_id:
            raise ValueError('Must pass app_id')
        if not profile_mem:
            raise ValueError('Must pass profile_mem')
        if not profile_type:
            raise ValueError('Must profile_type')
        mem_id = str(self.date_user_profile_id.generate_next_id())
        time = datetime.now(timezone.utc)
        data = {
            'id': mem_id,
            'user_id': user_id or '',
            'app_id': app_id or '',
            'is_implicit': is_implicit,
            'profile_type': profile_type,
            'mem': profile_mem,
            'source_id': source_id,
            'reasoning': reasoning,
            'context_summary': context_summary,
            'mem_type': mem_type,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.mem_store.write(user_id=user_id, app_id=app_id, mem_id=mem_id, data=data)
        return mem_id

    def _add_vector_user_profile_memory(
            self, user_id: str, app_id: str, memory_id: List[str],
            mem: List[str], mem_type: str = MemoryType.USER_PROFILE.value):
        """
        向向量数据库中添加用户画像的向量数据
        """
        dimension = len(mem[0])
        if dimension == 0:
            raise ValueError('dimension must not be zero')
        if self.semantic_recall:
            self.semantic_recall.add(mem=mem, memory_id=memory_id, user_id=user_id,
                                     app_id=app_id, mem_type=mem_type)
        else:
            raise ValueError('vector store must not be None')

    def _delete_vector_user_profile_memory(
            self, user_id: str, app_id: str,
            memory_id: List[str], mem_type: str = MemoryType.USER_PROFILE.value):
        if self.semantic_recall:
            self.semantic_recall.remove(ids=memory_id, user_id=user_id,
                                     app_id=app_id, mem_type=mem_type)
        else:
            raise ValueError('vector store must not be None')
