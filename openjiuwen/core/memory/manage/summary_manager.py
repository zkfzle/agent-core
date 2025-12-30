from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore
from openjiuwen.core.memory.common.base import generate_idx_name, parse_memory_hit_infos
from openjiuwen.core.memory.generation.conflict_resolution import ConflictResolution
from openjiuwen.core.memory.manage.base_memory_manager import BaseMemoryManager
from openjiuwen.core.memory.manage.data_id_manager import DataIdManager
from openjiuwen.core.memory.mem_unit.memory_unit import SummaryUnit, BaseMemoryUnit, MemoryType
from openjiuwen.core.memory.store.user_mem_store import UserMemStore


class SummaryManager(BaseMemoryManager):
    def __init__(self,
                 semantic_recall_instance: BaseSemanticStore,
                 user_mem_store: UserMemStore,
                 crypto_key: bytes):
        self.mem_store = user_mem_store
        self.semantic_recall = semantic_recall_instance
        self.crypto_key = crypto_key

    async def add(self, memory: BaseMemoryUnit, llm: Tuple[str, BaseModelClient] | None = None):
        """add memory."""
        if not isinstance(memory, SummaryUnit):
            raise ValueError('summary manager add Must pass SummaryUnit class.')

        await self._add_summary_memory_to_vector(memory)
        await self._add_summary_memory_to_mem_store(memory)

    async def update(self, user_id: str, group_id: str, mem_id: str, new_memory: str, **kwargs):
        """update memory by its id."""
        time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        encrypt_new_memory = BaseMemoryManager.encrypt_memory_if_needed(key=self.crypto_key, plaintext=new_memory)
        new_data = {'mem': encrypt_new_memory, 'time': time}
        await self.mem_store.update(mem_id=mem_id, user_id=user_id, group_id=group_id, data=new_data)
        table_name = generate_idx_name(user_id, group_id, MemoryType.SUMMARY.value)
        await self.semantic_recall.delete_docs([mem_id], table_name)
        # semantic memory embedding must not encrypt
        await self.semantic_recall.add_docs([(mem_id, new_memory)], table_name)
        return True

    async def delete(self, user_id: str, group_id: str, mem_id: str, **kwargs):
        """delete memory by its id."""
        data = await self.mem_store.get(user_id=user_id, group_id=group_id, mem_id=mem_id)
        if data is None:
            logger.error(f"Delete summary in store failed, the mem of mem_id({mem_id}) is not exist.")
            return False
        await self.mem_store.delete(mem_id=mem_id, user_id=user_id, group_id=group_id)
        await self._delete_vector_summary_memory(memory_id=[mem_id],
                                                      user_id=user_id,
                                                      group_id=group_id)
        return True

    async def delete_by_user_id(self, user_id: str, group_id: str):
        """delete memory by user id and app id."""
        data = await self.mem_store.get_all(user_id=user_id, group_id=group_id, mem_type=MemoryType.USER_PROFILE.value)
        if data is None:
            logger.error(f"Delete summary in store failed, the mem of user_id({user_id}) is not exist.")
            return False
        mem_ids = [item['id'] for item in data]
        await self.mem_store.batch_delete(user_id=user_id, group_id=group_id, mem_ids=mem_ids)
        await self._delete_vector_summary_memory(memory_id=mem_ids,
                                                 user_id=user_id,
                                                group_id=group_id)
        return True

    async def get(self, user_id: str, group_id: str, mem_id: str) -> dict[str, Any] | None:
        """get memory by its id."""
        retrieve_res = await self.mem_store.get(user_id=user_id, group_id=group_id, mem_id=mem_id)
        retrieve_res["mem"] = BaseMemoryManager.decrypt_memory_if_needed(key=self.crypto_key,
                                                                         ciphertext=retrieve_res["mem"])
        return retrieve_res

    async def search(self, user_id: str, group_id: str, query: str, top_k: int, **kwargs):
        """query memory, return top k results"""
        mem_ids, scores = await self._recall_by_vector(query, user_id, group_id, top_k)
        retrieve_res = await self.mem_store.batch_get(user_id=user_id, group_id=group_id, mem_ids=mem_ids)
        if retrieve_res is None:
            return None
        for item in retrieve_res:
            item["score"] = scores.get(item['id'], 0)
            item["mem"] = BaseMemoryManager.decrypt_memory_if_needed(key=self.crypto_key, ciphertext=item["mem"])

        retrieve_res.sort(key=lambda x: x["score"], reverse=True)
        return retrieve_res

    async def _add_summary_memory_to_mem_store(self, summary_unit: SummaryUnit):
        time = datetime.now(timezone.utc)
        mem = BaseMemoryManager.encrypt_memory_if_needed(key=self.crypto_key,
                                                         plaintext=summary_unit.summary)
        data = {
            'id': summary_unit.mem_id,
            'user_id': summary_unit.user_id or '',
            'group_id': summary_unit.group_id or '',
            'mem': mem,
            'source_id': summary_unit.message_mem_id,
            'mem_type': MemoryType.SUMMARY.value,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            # not used
            'is_implicit': False,
            'profile_type': "",
            'reasoning': "",
            'context_summary': "",
        }
        await self.mem_store.write(user_id=summary_unit.user_id,
                                   group_id=summary_unit.group_id,
                                   mem_id=summary_unit.mem_id,
                                   data=data)

    async def _add_summary_memory_to_vector(self, summary_unit: SummaryUnit):
        if self.semantic_recall:
            table_name = generate_idx_name(user_id=summary_unit.user_id,
                                           group_id=summary_unit.group_id,
                                           mem_type=MemoryType.SUMMARY.value)
            await self.semantic_recall.add_docs([(summary_unit.mem_id, summary_unit.summary)], table_name)
        else:
            raise ValueError('vector store must not be None')

    async def _delete_vector_summary_memory(
            self,
            user_id: str,
            group_id: str,
            memory_id: List[str]):
        if self.semantic_recall:
            table_name = generate_idx_name(user_id, group_id, MemoryType.SUMMARY.value)
            await self.semantic_recall.delete_docs(memory_id, table_name)
        else:
            raise ValueError('vector store must not be None')

    async def _recall_by_vector(
            self,
            query: str,
            user_id: str,
            group_id: str,
            top_k: int = 5) -> tuple[List[str], dict[str, float]]:
        table_name = generate_idx_name(user_id, group_id, MemoryType.SUMMARY.value)
        memory_hit_info = await self.semantic_recall.search(query, table_name, top_k)
        return parse_memory_hit_infos(memory_hit_info)
