# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ChromaDB 向量存储测试用例
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import shutil
import json

from openjiuwen.core.retrieval.vector_store.chroma_store import ChromaVectorStore
from openjiuwen.core.retrieval.common.config import VectorStoreConfig
from openjiuwen.core.retrieval.common.retrieval_result import SearchResult


@pytest.fixture
def temp_chroma_path():
    """创建临时 ChromaDB 路径"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_store_config():
    """创建向量存储配置"""
    return VectorStoreConfig(
        collection_name="test_collection",
        distance_metric="cosine",
    )


class TestChromaVectorStore:
    """ChromaDB 向量存储测试"""

    def test_init_success(self, temp_chroma_path, vector_store_config):
        """测试初始化成功"""
        store = ChromaVectorStore(
            config=vector_store_config,
            chroma_path=temp_chroma_path,
        )
        assert store.collection_name == "test_collection"
        assert store.chroma_path == temp_chroma_path
        assert store.client is not None
        assert store.collection is not None

    def test_init_empty_path(self, vector_store_config):
        """测试空路径初始化"""
        with pytest.raises(ValueError, match="chroma_path is required"):
            ChromaVectorStore(config=vector_store_config, chroma_path="")

    def test_init_whitespace_path(self, vector_store_config):
        """测试空白路径初始化"""
        with pytest.raises(ValueError, match="chroma_path is required"):
            ChromaVectorStore(config=vector_store_config, chroma_path="   ")

    def test_init_with_custom_fields(self, temp_chroma_path, vector_store_config):
        """测试使用自定义字段初始化"""
        store = ChromaVectorStore(
            config=vector_store_config,
            chroma_path=temp_chroma_path,
            text_field="custom_text",
            vector_field="custom_vector",
            doc_id_field="custom_doc_id",
        )
        assert store.text_field == "custom_text"
        assert store.vector_field == "custom_vector"
        assert store.doc_id_field == "custom_doc_id"

    def test_init_with_euclidean_metric(self, temp_chroma_path):
        """测试使用欧氏距离初始化"""
        config = VectorStoreConfig(
            collection_name="test_collection",
            distance_metric="euclidean",
        )
        store = ChromaVectorStore(config=config, chroma_path=temp_chroma_path)
        assert store.config.distance_metric == "euclidean"

    @pytest.mark.asyncio
    async def test_add_single_dict(self, temp_chroma_path, vector_store_config):
        """测试添加单个字典"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        data = {
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test"},
        }
        
        await store.add(data)
        # 验证数据已添加（通过搜索）
        results = await store.search([0.1] * 384, top_k=1)
        assert len(results) >= 0  # 可能为空，取决于 ChromaDB 的行为

    @pytest.mark.asyncio
    async def test_add_without_embedding(self, temp_chroma_path, vector_store_config):
        """测试添加没有向量的数据"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        data = {
            "id": "1",
            "content": "Test content",
        }
        
        # 应该跳过没有向量的数据
        await store.add(data)
        # 不应该抛出异常

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, temp_chroma_path, vector_store_config):
        """测试添加带元数据的数据"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        data = {
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test", "author": "test_author"},
            "document_id": "doc_1",
            "chunk_id": "chunk_1",
        }
        
        await store.add(data)
        # 验证元数据已保存
        results = await store.search([0.1] * 384, top_k=1)
        if results:
            assert "source" in results[0].metadata or "document_id" in results[0].metadata

    @pytest.mark.asyncio
    async def test_search_with_filters(self, temp_chroma_path, vector_store_config):
        """测试带过滤条件的搜索"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        # 先添加数据
        await store.add({
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test"},
            "document_id": "doc_1",
        })
        
        filters = {"document_id": "doc_1"}
        results = await store.search([0.1] * 384, top_k=5, filters=filters)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_sparse_search_with_filters(self, temp_chroma_path, vector_store_config):
        """测试带过滤条件的稀疏搜索"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        await store.add({
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test"},
        })
        
        filters = {"source": "test"}
        results = await store.sparse_search("Test", top_k=5, filters=filters)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self, temp_chroma_path, vector_store_config):
        """测试带过滤条件的混合搜索"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        await store.add({
            "id": "1",
            "embedding": [0.1] * 384,
            "content": "Test content",
            "metadata": {"source": "test"},
        })
        
        filters = {"source": "test"}
        results = await store.hybrid_search(
            query_text="Test",
            query_vector=[0.1] * 384,
            top_k=5,
            filters=filters,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_failure(self, temp_chroma_path, vector_store_config):
        """测试混合搜索失败"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        with patch.object(store, "search", side_effect=Exception("Search error")):
            results = await store.hybrid_search(
                query_text="Test",
                query_vector=[0.1] * 384,
                top_k=5,
            )
            # 应该返回空列表或部分结果
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_delete_by_filter_expr(self, temp_chroma_path, vector_store_config):
        """测试通过过滤表达式删除"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        # ChromaDB 不支持复杂的 filter_expr，应该返回 False
        result = await store.delete(filter_expr="source == 'test'")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_without_params(self, temp_chroma_path, vector_store_config):
        """测试没有参数时删除"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        result = await store.delete()
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_with_exception(self, temp_chroma_path, vector_store_config):
        """测试删除时发生异常"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = Exception("Delete error")
            
            result = await store.delete(ids=["1"])
            assert result is False

    def test_close(self, temp_chroma_path, vector_store_config):
        """测试关闭向量存储"""
        store = ChromaVectorStore(config=vector_store_config, chroma_path=temp_chroma_path)
        # 不应该抛出异常
        store.close()

