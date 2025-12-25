#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import List, Tuple

import pytest

from openjiuwen.core.memory.manage.data_id_manager import DataIdManager
from openjiuwen.core.memory.manage.episodic_memory_manager import EpisodicMemoryManager
from openjiuwen.core.memory.mem_unit.memory_unit import EpisodicMemoryUnit, MemoryType
from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore
from openjiuwen.core.memory.store.user_mem_store import UserMemStore
from tests.unit_tests.core.memory.store.mock_kv_store import MockKVStore


# Mock语义存储实现，避免实际模型加载
class MockSemanticStore(BaseSemanticStore):
    """Mock语义存储，用于测试环境，不依赖实际模型"""

    def __init__(self, config, model_config):
        self.memory_store = {}
        self.config = config
        self.model_config = model_config

    async def add_docs(self, docs: List[Tuple[str, str]], table_name: str) -> bool:
        """模拟添加记忆"""
        if table_name not in self.memory_store:
            self.memory_store[table_name] = {}

        for mid, m in docs:
            self.memory_store[table_name][mid] = {
                'content': m
            }
        return True

    async def delete_docs(self, ids: List[str], table_name: str) -> bool:
        """模拟删除记忆"""
        if table_name in self.memory_store:
            for id_to_remove in ids:
                self.memory_store[table_name].pop(id_to_remove, None)
        return True

    async def search(self, query: str, table_name: str, top_k: int) -> List[Tuple[str, float]]:
        """模拟搜索功能，返回匹配的记忆"""
        if table_name not in self.memory_store:
            return []

        # 简单的文本匹配搜索
        results: List[Tuple[str, float]] = []
        for memory_id, memory_data in self.memory_store[table_name].items():
            content = memory_data['content']
            # 简单的关键词匹配
            if any(q in content for q in query):
                # 模拟返回SearchHit对象
                results.append((memory_id, 0.0))

        # 返回top_k个结果
        return results[-5:]

    async def delete_table(self, table_name: str) -> bool:
        """模拟删除索引功能"""
        if table_name in self.memory_store:
            del self.memory_store[table_name]
        return True


class TestEpisodicMemoryManager:
    @pytest.mark.asyncio
    async def test_basic(self):
        mock_semantic_store = MockSemanticStore(None, None)
        mock_kv_store = MockKVStore()
        mock_mem_store = UserMemStore(mock_kv_store)
        data_id_manager = DataIdManager()
        episodic_memory_manager = EpisodicMemoryManager(
            semantic_store=mock_semantic_store,
            user_mem_store=mock_mem_store,
            data_id_manager=data_id_manager,
            crypto_key=bytes()
        )
        data1 = EpisodicMemoryUnit(
            mem_type=MemoryType.EPISODIC_MEMORY,
            user_id="user1",
            group_id="group1",
            content="用户的职业是软件工程师",
            message_mem_id="message_id_1"
        )
        await episodic_memory_manager.add(data1)
        res = await episodic_memory_manager.search("user1", "group1", "用户的职业", 5)
        print(f"res = {res}")
