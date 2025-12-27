# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
索引管理器抽象基类测试用例
"""
import pytest
from unittest.mock import AsyncMock

from openjiuwen.core.retrieval.indexing.indexer.base import Indexer
from openjiuwen.core.retrieval.common.config import IndexConfig
from openjiuwen.core.retrieval.common.document import TextChunk


class ConcreteIndexer(Indexer):
    """具体索引管理器实现，用于测试抽象基类"""

    async def build_index(self, chunks, config, embed_model=None, **kwargs):
        return True

    async def update_index(self, chunks, doc_id, config, embed_model=None, **kwargs):
        return True

    async def delete_index(self, doc_id, index_name, **kwargs):
        return True

    async def index_exists(self, index_name):
        return True

    async def get_index_info(self, index_name):
        return {"count": 10}


class TestIndexer:
    """索引管理器抽象基类测试"""

    @pytest.mark.asyncio
    async def test_build_index(self):
        """测试构建索引"""
        indexer = ConcreteIndexer()
        chunks = [
            TextChunk(id_="1", text="chunk 1", doc_id="doc_1"),
            TextChunk(id_="2", text="chunk 2", doc_id="doc_1"),
        ]
        config = IndexConfig(index_name="test_index", index_type="vector")
        result = await indexer.build_index(chunks, config)
        assert result is True

    @pytest.mark.asyncio
    async def test_update_index(self):
        """测试更新索引"""
        indexer = ConcreteIndexer()
        chunks = [TextChunk(id_="1", text="updated chunk", doc_id="doc_1")]
        config = IndexConfig(index_name="test_index", index_type="vector")
        result = await indexer.update_index(chunks, "doc_1", config)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_index(self):
        """测试删除索引"""
        indexer = ConcreteIndexer()
        result = await indexer.delete_index("doc_1", "test_index")
        assert result is True

    @pytest.mark.asyncio
    async def test_index_exists(self):
        """测试检查索引是否存在"""
        indexer = ConcreteIndexer()
        result = await indexer.index_exists("test_index")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_index_info(self):
        """测试获取索引信息"""
        indexer = ConcreteIndexer()
        info = await indexer.get_index_info("test_index")
        assert "count" in info
        assert info["count"] == 10

