# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
PGVector store test cases - Comprehensive Coverage
"""

from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
from sqlalchemy import Table, Column, String, Text, MetaData, delete
from sqlalchemy.dialects.postgresql import JSONB

from pgvector.sqlalchemy import Vector

from openjiuwen.core.retrieval.common.config import VectorStoreConfig
from openjiuwen.core.retrieval.vector_store.pg_store import PGVectorStore
from openjiuwen.core.retrieval.indexing.vector_fields.pg_fields import PGVectorField


@pytest.fixture
def vector_store_config():
    return VectorStoreConfig(
        collection_name="test_collection",
        distance_metric="euclidean",
    )

@pytest.fixture
def mock_session_factory():
    """Creates a mock session with async context manager support"""
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock()
    
    # Mock begin() transaction
    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=None)
    mock_session.begin.return_value = mock_transaction
    
    return mock_session


class TestPGVectorStore:

    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    def test_init(self, mock_sessionmaker, mock_create_engine, vector_store_config):
        """UT-001: Test initialization"""
        store = PGVectorStore(
            config=vector_store_config,
            pg_uri="postgresql+asyncpg://user:pass@localhost/db",
        )
        assert store.pg_uri == "postgresql+asyncpg://user:pass@localhost/db"
        assert store.collection_name == "test_collection"
        mock_create_engine.assert_called_once()
        
        # Test metric mapping logic
        config_cosine = VectorStoreConfig(collection_name="c", distance_metric="cosine")
        store_cosine = PGVectorStore(config=config_cosine, pg_uri="uri")
        assert store_cosine._distance_metric == "cosine"

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    async def test_add_batch_and_upsert(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """UT-002 & Functional B: Test data adding with batching and upsert logic"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        mock_create_engine.return_value = AsyncMock()
        
        store = PGVectorStore(config=vector_store_config, pg_uri="sqlite+aiosqlite:///:memory:")
        
        # Mock table with real columns to avoid KeyError during access
        metadata = MetaData()
        mock_table = Table(
            "test_collection", metadata,
            Column("id", String, primary_key=True),
            Column("content", Text),
            Column("metadata", JSONB),
            Column("embedding", Vector(2))
        )
        store._table = mock_table
        
        # Create 150 items to test batching (default batch 128)
        data = [{"id": str(i), "content": f"text{i}", "embedding": [1.0, 2.0], "metadata": {"i": i}} for i in range(150)]
        
        with patch.object(store, '_get_or_create_table', new_callable=AsyncMock) as mock_get_table:
            mock_get_table.return_value = mock_table
            
            # Use small batch size to verify multiple batches
            await store.add(data, batch_size=100)
            
            # Should be called twice (100 + 50)
            assert mock_session_factory.execute.call_count == 2

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    async def test_search_metric_handling(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """UT-003 & Functional C: Test search with different metrics"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        mock_create_engine.return_value = AsyncMock()
        
        # Mock result
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.id = "1"
        mock_row.content = "res"
        mock_row.metadata = {}
        mock_row.distance = 0.5
        mock_result.fetchall.return_value = [mock_row]
        
        # Make execute return our result mock
        mock_session_factory.execute = AsyncMock(return_value=mock_result)
        
        # Define real table to avoid sqlalchemy coercion error with MagicMock
        metadata = MetaData()
        mock_table = Table(
            "test_collection", metadata,
            Column("id", String, primary_key=True),
            Column("content", Text),
            Column("metadata", JSONB),
            Column("embedding", Vector(2))
        )
        
        # We need to spy on the column methods or mock them
        # Since Vector type handles this, we can just check if logic executes without error
        # Or wrap the Vector type to spy on it.
        # Simpler: just verify that search runs for each metric type.
        
        # Test L2
        store = PGVectorStore(config=vector_store_config, pg_uri="uri")
        store._table = mock_table
        
        with patch.object(store, '_get_or_create_table', new_callable=AsyncMock) as mget:
            mget.return_value = mock_table
            await store.search([1.0, 0.0])
            # If we reached here, SQL generation worked for L2

        # Test Cosine
        config_cos = VectorStoreConfig(collection_name="c", distance_metric="cosine")
        store_cos = PGVectorStore(config=config_cos, pg_uri="uri")
        store_cos._table = mock_table
        
        with patch.object(store_cos, '_get_or_create_table', new_callable=AsyncMock) as mget:
            mget.return_value = mock_table
            await store_cos.search([1.0, 0.0])
            # If we reached here, SQL generation worked for Cosine

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    async def test_sparse_search(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """Functional C: Test sparse search generation"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        
        store = PGVectorStore(config=vector_store_config, pg_uri="uri")
        
        mock_table = Table("t", MetaData(), Column("content", Text))
        store._table = mock_table
        
        mock_row = MagicMock()
        mock_row.id = "1"
        mock_row.content = "text"
        mock_row.rank = 0.9
        mock_row.metadata = {}
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        
        # Set return value for execute
        mock_session_factory.execute = AsyncMock(return_value=mock_result)

        with patch.object(store, '_get_or_create_table', new_callable=AsyncMock) as mget:
            mget.return_value = mock_table
            results = await store.sparse_search("query")
            
            assert len(results) == 1
            assert results[0].score == 0.9
            assert mock_session_factory.execute.called

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    async def test_hybrid_search(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """Functional C: Test hybrid search fusion"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        
        store = PGVectorStore(config=vector_store_config, pg_uri="uri")
        
        # Mock search and sparse_search results
        from openjiuwen.core.retrieval.common.retrieval_result import SearchResult
        
        res_vec = [SearchResult(id="1", text="t1", score=0.9, metadata={"id": "1"})]
        res_sparse = [SearchResult(id="1", text="t1", score=0.8, metadata={"id": "1"}), SearchResult(id="2", text="t2", score=0.7, metadata={"id": "2"})]
        
        with patch.object(store, 'search', new_callable=AsyncMock) as mock_search, \
             patch.object(store, 'sparse_search', new_callable=AsyncMock) as mock_sparse:
            
            mock_search.return_value = res_vec
            mock_sparse.return_value = res_sparse
            
            results = await store.hybrid_search("q", [0.1])
            
            assert mock_search.called
            assert mock_sparse.called
            # ID "1" is in both, should be ranked high. "2" is only in sparse.
            ids = [r.id for r in results]
            assert "1" in ids
            assert "2" in ids

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    async def test_delete_by_ids(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """Functional B: Test delete by IDs"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        store = PGVectorStore(config=vector_store_config, pg_uri="uri")
        
        mock_table = Table("t", MetaData(), Column("id", String, primary_key=True))
        store._table = mock_table
        
        with patch.object(store, '_get_or_create_table', new_callable=AsyncMock) as mget:
            mget.return_value = mock_table
            
            # Success case
            res = await store.delete(ids=["1", "2"])
            assert res is True
            assert mock_session_factory.execute.called
            
            # Fail case (no table)
            mget.return_value = None
            res = await store.delete(ids=["1"])
            assert res is False

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    async def test_crud_lifecycle(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """Functional B: Full CRUD lifecycle test (Create, Read, Update, Delete)"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        
        store = PGVectorStore(config=vector_store_config, pg_uri="uri")
        
        # Mock table
        metadata = MetaData()
        mock_table = Table(
            "test_collection", metadata,
            Column("id", String, primary_key=True),
            Column("content", Text),
            Column("metadata", JSONB),
            Column("embedding", Vector(2))
        )
        store._table = mock_table
        
        with patch.object(store, '_get_or_create_table', new_callable=AsyncMock) as mock_get_table:
            mock_get_table.return_value = mock_table
            
            # 1. CREATE (Add)
            data = [{"id": "1", "content": "initial", "embedding": [0.1, 0.2]}]
            await store.add(data)
            assert mock_session_factory.execute.called
            mock_session_factory.execute.reset_mock()
            
            # 2. READ (Search with ID filter)
            # Mock return for search
            mock_result = MagicMock()
            mock_row = MagicMock()
            mock_row.id = "1"
            mock_row.content = "initial"
            mock_row.distance = 0.0
            mock_row.metadata = {}
            mock_result.fetchall.return_value = [mock_row]
            mock_session_factory.execute.return_value = mock_result
            
            # Use side effect to return mock column for embedding, allow others to pass
            # We don't need to mock table.c! Real Vector column returns valid SQL expression.
            # Just let it run.
            
            results = await store.search(query_vector=[0.1, 0.2], filters={"id": "1"})
            assert len(results) == 1
            assert results[0].text == "initial"
            assert mock_session_factory.execute.called
            mock_session_factory.execute.reset_mock()
            
            # 3. UPDATE (Add same ID)

            data_update = [{"id": "1", "content": "updated", "embedding": [0.1, 0.3]}]
            await store.add(data_update)
            assert mock_session_factory.execute.called
            # In a real DB, this uses ON CONFLICT DO UPDATE. 
            # With mocks, we verify the execute is called again.
            mock_session_factory.execute.reset_mock()
            
            # 4. DELETE
            await store.delete(ids=["1"])
            assert mock_session_factory.execute.called

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    async def test_filter_building(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """Functional C: Test metadata filter construction"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        store = PGVectorStore(config=vector_store_config, pg_uri="uri")
        
        # Define real table to avoid mocking issues with ColumnCollection
        metadata = MetaData()
        mock_table = Table(
            "test_collection", metadata,
            Column("id", String, primary_key=True),
            Column("metadata", JSONB)
        )
        store._table = mock_table
        
        filters = {"category": "news", "id": "123"}
        
        conds = store._build_filters(mock_table, filters)
        
        assert len(conds) == 2
        # One should be direct column comparison (id)
        # One should be JSON containment (category)

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.inspect")
    async def test_table_management(self, mock_inspect, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """Functional A: Test table existence check and deletion"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        # Mock engine connect/begin for table ops
        mock_conn = AsyncMock()
        mock_create_engine.return_value.connect.return_value.__aenter__.return_value = mock_conn
        mock_create_engine.return_value.begin.return_value.__aenter__.return_value = mock_conn
        
        store = PGVectorStore(config=vector_store_config, pg_uri="uri")
        
        # 1. Test table_exists
        # Mock run_sync to execute the lambda we pass to it
        async def side_effect_run_sync(fn):
            return fn(MagicMock()) # execute lambda with mock sync_conn
        mock_conn.run_sync.side_effect = side_effect_run_sync
        
        mock_inspector = MagicMock()
        mock_inspector.has_table.return_value = True
        mock_inspect.return_value = mock_inspector
        
        exists = await store.table_exists("test_table")
        assert exists is True
        mock_inspector.has_table.assert_called_with("test_table")
        
        # 2. Test delete_table
        await store.delete_table("test_table")
        # Verify execute called with DROP TABLE string
        assert mock_conn.execute.called
        call_args = mock_conn.execute.call_args[0][0]
        assert "DROP TABLE IF EXISTS test_table" in str(call_args)

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    async def test_edge_cases(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """Functional: Test edge cases (empty inputs, unsupported ops)"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        store = PGVectorStore(config=vector_store_config, pg_uri="uri")
        # Use real table to pass SQLAlchemy coercion
        mock_table = Table("t", MetaData(), Column("id", String))
        store._table = mock_table
        
        # 1. Add empty list
        await store.add([])
        assert not mock_session_factory.execute.called
        
        # 2. Delete with unsupported filter_expr
        res = await store.delete(filter_expr="some_expr")
        assert res is False
        
        # 3. Search on non-existent table (mock _get_or_create returning None)
        with patch.object(store, '_get_or_create_table', new_callable=AsyncMock) as mget:
            mget.return_value = None
            res = await store.search([0.1])
            assert res == []

    @pytest.mark.asyncio
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    async def test_exception_handling(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        """Functional: Test exception handling during DB operations"""
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)
        store = PGVectorStore(config=vector_store_config, pg_uri="uri")
        
        # Mock table with required columns to pass internal logic before execution
        store._table = Table(
            "t", 
            MetaData(), 
            Column("id", String, primary_key=True),
            Column("content", Text), # Required for upsert logic
            Column("metadata", JSONB), # Required for upsert logic
            Column("embedding", Vector(2))
        )
        
        # 1. Add Failure (DB Error)
        mock_session_factory.execute.side_effect = Exception("DB Connection Lost")
        with pytest.raises(Exception, match="DB Connection Lost"):
            await store.add([{"id": "1", "content": "c", "embedding": [0.1, 0.2]}])
            
        # 2. Search Failure
        mock_session_factory.execute.side_effect = Exception("Query Failed")
        with pytest.raises(Exception, match="Query Failed"):
            await store.search([0.1, 0.2])
            
    def test_init_invalid_config(self, vector_store_config):
        """Functional: Test initialization with invalid configuration"""
        # Invalid vector_field type
        with pytest.raises(Exception): # build_error raises BaseError which inherits Exception
            PGVectorStore(
                config=vector_store_config,
                pg_uri="uri",
                vector_field=123 # Invalid type
            )
