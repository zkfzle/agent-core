#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from datetime import datetime, timezone
from typing import Any, List, Tuple

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.common.base import generate_idx_name, parse_memory_hit_infos
from openjiuwen.core.memory.manage.base_memory_manager import BaseMemoryManager
from openjiuwen.core.memory.mem_unit.memory_unit import EpisodicMemoryUnit, MemoryType, BaseMemoryUnit
from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore
from openjiuwen.core.memory.store.user_mem_store import UserMemStore
from openjiuwen.core.utils.llm.base import BaseModelClient


class EpisodicMemoryManager(BaseMemoryManager):
    def __init__(self,
                 semantic_store: BaseSemanticStore,
                 user_mem_store: UserMemStore,
                 crypto_key: bytes):
        self.mem_store = user_mem_store
        self.semantic_store = semantic_store
        self.crypto_key = crypto_key

    async def add(self, memory: BaseMemoryUnit, llm: Tuple[str, BaseModelClient] | None = None):
        if not isinstance(memory, EpisodicMemoryUnit):
            raise ValueError('invalid memory unit during adding episodic memory.')
        if not memory.user_id:
            raise ValueError('episodic_memory_manager add operation must pass user_id')
        if not memory.group_id:
            raise ValueError('episodic_memory_manager add operation must pass group_id')
        mem_id = await self._add_episodic_memory_memory(memory=memory)
        await self._add_vector_episodic_memory_memory(user_id=memory.user_id,
                                                      group_id=memory.group_id,
                                                      mem_id=mem_id,
                                                      mem=memory.content)

    async def update(self, user_id: str, group_id: str, mem_id: str, new_memory: str, **kwargs) -> bool:
        time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        encrypt_new_memory = BaseMemoryManager.encrypt_memory_if_needed(key=self.crypto_key, plaintext=new_memory)
        new_data = {'mem': encrypt_new_memory, 'time': time}
        await self.mem_store.update(mem_id=mem_id, user_id=user_id, group_id=group_id, data=new_data)
        table_name = generate_idx_name(user_id, group_id, MemoryType.EPISODIC_MEMORY.value)
        await self.semantic_store.delete_docs([mem_id], table_name)
        # semantic memory embedding must not encrypt
        await self.semantic_store.add_docs([(mem_id, new_memory)], table_name)
        return True

    async def search(self, user_id: str, group_id: str, query: str, top_k: int, **kwargs):
        mem_ids, scores = await self._recall_by_vector(query, user_id, group_id, top_k)
        retrieve_res = await self.mem_store.batch_get(user_id=user_id, group_id=group_id, mem_ids=mem_ids)
        if retrieve_res is None:
            return None
        for item in retrieve_res:
            item["score"] = scores.get(item['id'], 0)
            item["mem"] = BaseMemoryManager.decrypt_memory_if_needed(key=self.crypto_key, ciphertext=item["mem"])
        retrieve_res.sort(key=lambda x: scores.get(x["id"], 0), reverse=True)
        return retrieve_res

    async def get(self, user_id: str, group_id: str, mem_id: str) -> dict[str, Any] | None:
        retrieve_res = await self.mem_store.get(user_id=user_id, group_id=group_id, mem_id=mem_id)
        retrieve_res["mem"] = BaseMemoryManager.decrypt_memory_if_needed(key=self.crypto_key,
                                                                         ciphertext=retrieve_res["mem"])
        return retrieve_res

    async def delete(self, user_id: str, group_id: str, mem_id: str, **kwargs):
        data = await self.mem_store.get(user_id=user_id, group_id=group_id, mem_id=mem_id)
        if data is None:
            logger.error(f"Delete episodic_memory in store failed, the mem of mem_id({mem_id}) is not exist.")
            return False
        await self.mem_store.delete(mem_id=mem_id, user_id=user_id, group_id=group_id)
        await self._delete_vector_episodic_memory_memory(user_id=user_id, group_id=group_id, mem_ids=[mem_id],
                                                         mem_type=MemoryType.EPISODIC_MEMORY.value)
        return True

    async def delete_by_user_id(self, user_id: str, group_id: str):
        data = await self.mem_store.get_all(user_id=user_id, group_id=group_id,
                                            mem_type=MemoryType.EPISODIC_MEMORY.value)
        if data is None:
            logger.error(f"Delete episodic_memory in store failed, the mem of user_id({user_id}) is not exist.")
            return False
        mem_ids = [item['id'] for item in data]
        await self.mem_store.batch_delete(user_id=user_id, group_id=group_id, mem_ids=mem_ids)
        await self._delete_vector_episodic_memory_memory(user_id=user_id, group_id=group_id, mem_ids=mem_ids,
                                                         mem_type=MemoryType.EPISODIC_MEMORY.value)
        return True

    async def list_episodic_memory(self, user_id: str, group_id: str) -> list[dict[str, Any]]:
        results = await self.mem_store.get_all(user_id=user_id, group_id=group_id,
                                               mem_type=MemoryType.EPISODIC_MEMORY.value)
        if not results:
            logger.debug(f"End to get episodic memory, result is None, user_id:{user_id}, group_id:{group_id}")
            return []
        for data in results:
            data["mem"] = BaseMemoryManager.decrypt_memory_if_needed(key=self.crypto_key, ciphertext=data["mem"])
        return results

    async def _recall_by_vector(self, query: str, user_id: str, group_id: str,
                                top_k: int = 5) -> tuple[List[str], dict[str, float]]:
        table_name = generate_idx_name(user_id, group_id, MemoryType.EPISODIC_MEMORY.value)
        memory_hit_info = await self.semantic_store.search(query, table_name, top_k)
        return parse_memory_hit_infos(memory_hit_info)

    async def _add_episodic_memory_memory(
            self,
            memory: EpisodicMemoryUnit
    ) -> str:
        content = BaseMemoryManager.encrypt_memory_if_needed(key=self.crypto_key, plaintext=memory.content)
        data = {
            'id': memory.mem_id,
            'user_id': memory.user_id or '',
            'group_id': memory.group_id or '',
            'mem': content,
            'source_id': memory.user_id,
            'mem_type': MemoryType.EPISODIC_MEMORY.value,
            'timestamp': memory.timestamp,
        }
        await self.mem_store.write(user_id=memory.user_id, group_id=memory.group_id,
                                   mem_id=memory.mem_id, data=data)
        return memory.mem_id

    async def _add_vector_episodic_memory_memory(
            self, user_id: str, group_id: str, mem_id: str, mem: str, mem_type: str = MemoryType.EPISODIC_MEMORY.value):
        if self.semantic_store:
            table_name = generate_idx_name(user_id, group_id, mem_type)
            await self.semantic_store.add_docs([(mem_id, mem)], table_name)
        else:
            raise ValueError('semantic store must not be None')

    async def _delete_vector_episodic_memory_memory(
            self, user_id: str, group_id: str, mem_ids: List[str], mem_type: str = MemoryType.EPISODIC_MEMORY.value):
        if self.semantic_store:
            table_name = generate_idx_name(user_id, group_id, mem_type)
            await self.semantic_store.delete_docs(mem_ids, table_name)
        else:
            raise ValueError('semantic store must not be None')
