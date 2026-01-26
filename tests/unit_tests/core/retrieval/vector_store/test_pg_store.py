# coding: utf-8
import json
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from openjiuwen.core.retrieval.common.config import VectorStoreConfig
from openjiuwen.core.retrieval.vector_store.pg_store import PgVectorStore
from openjiuwen.core.retrieval.common.retrieval_result import SearchResult


@pytest.fixture
def mock_asyncpg():
    with patch("openjiuwen.core.retrieval.vector_store.pg_store.asyncpg") as mock:
        yield mock


@pytest.fixture
def mock_register_vector():
    with patch("openjiuwen.core.retrieval.vector_store.pg_store.register_vector") as mock:
        yield mock


@pytest.fixture
def store_config():
    return VectorStoreConfig(
        collection_name="test_collection",
        database_name="test_db",
        distance_metric="cosine"
    )


@pytest.fixture
def pg_store(store_config):
    # Mock imports to avoid runtime errors if not installed in environment (though we installed them)
    # But mainly to ensure we control the mocking
    with patch("openjiuwen.core.retrieval.vector_store.pg_store.asyncpg") as mock_pg:
        store = PgVectorStore(
            config=store_config,
            dsn="postgresql://user:pass@localhost:5432/db",
            table_name="test_embeddings",
            embedding_dim=4
        )
        yield store


@pytest.mark.asyncio
async def test_init_pool(pg_store, mock_register_vector):
    # Setup mock pool and connection
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    # Configure create_pool to return our mock pool
    with patch("openjiuwen.core.retrieval.vector_store.pg_store.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)) as mock_create_pool:
        await pg_store._init_pool()
        
        # Verify pool creation
        mock_create_pool.assert_called_once_with(pg_store.dsn)
        assert pg_store.pool == mock_pool
        
        # Verify extension creation and table creation
        mock_conn.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS vector")
        mock_register_vector.assert_called_once_with(mock_conn)
        
        # Check if CREATE TABLE was called
        create_table_call = [call for call in mock_conn.execute.call_args_list if "CREATE TABLE IF NOT EXISTS" in str(call)]
        assert create_table_call


@pytest.mark.asyncio
async def test_add(pg_store):
    # Setup mocks
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    pg_store.pool = mock_pool
    
    data = [
        {
            "id": "1",
            "content": "test content",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"key": "value"}
        }
    ]
    
    await pg_store.add(data)
    
    # Verify executemany called
    assert mock_conn.executemany.called
    args = mock_conn.executemany.call_args
    assert "INSERT INTO test_embeddings" in args[0][0]
    assert len(args[0][1]) == 1 # 1 batch
    assert args[0][1][0][0] == "1" # id
    assert args[0][1][0][1] == "test content" # content
    assert args[0][1][0][2] == [0.1, 0.2, 0.3, 0.4] # embedding
    assert json.loads(args[0][1][0][3]) == {"key": "value"} # metadata


@pytest.mark.asyncio
async def test_search(pg_store):
    # Setup mocks
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    pg_store.pool = mock_pool
    
    # Mock return data
    mock_row = {
        "id": "1",
        "content": "test content",
        "distance": 0.1,
        "metadata": json.dumps({"key": "value"})
    }
    mock_conn.fetch.return_value = [mock_row]
    
    query_vector = [0.1, 0.2, 0.3, 0.4]
    results = await pg_store.search(query_vector, top_k=2)
    
    # Verify results
    assert len(results) == 1
    assert results[0].id == "1"
    assert results[0].text == "test content"
    assert results[0].metadata == {"key": "value"}
    # Distance 0.1, Cosine metric => Score = 1.0 - 0.1 = 0.9
    assert results[0].score == 0.9
    
    # Verify query
    assert mock_conn.fetch.called
    sql = mock_conn.fetch.call_args[0][0]
    assert "SELECT id" in sql
    assert "ORDER BY embedding <=> $1" in sql
    assert "LIMIT 2" in sql


@pytest.mark.asyncio
async def test_search_with_filters(pg_store):
    # Setup mocks
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    pg_store.pool = mock_pool
    mock_conn.fetch.return_value = []
    
    query_vector = [0.1, 0.2, 0.3, 0.4]
    filters = {"category": "news", "published": True}
    
    await pg_store.search(query_vector, filters=filters)
    
    # Verify query structure
    sql = mock_conn.fetch.call_args[0][0]
    assert "metadata->>'category' = $2" in sql or "metadata->>'category' = $3" in sql
    assert "metadata->>'published' = $3" in sql or "metadata->>'published' = $2" in sql


@pytest.mark.asyncio
async def test_sparse_search(pg_store):
    # Setup mocks
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    pg_store.pool = mock_pool
    
    mock_row = {
        "id": "1",
        "content": "test content",
        "score": 0.5,
        "metadata": json.dumps({"key": "value"})
    }
    mock_conn.fetch.return_value = [mock_row]
    
    results = await pg_store.sparse_search("test query", top_k=5)
    
    assert len(results) == 1
    assert results[0].score == 0.5
    
    # Verify query
    assert mock_conn.fetch.called
    sql = mock_conn.fetch.call_args[0][0]
    assert "to_tsvector" in sql
    assert "plainto_tsquery" in sql


@pytest.mark.asyncio
async def test_delete(pg_store):
    # Setup mocks
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    pg_store.pool = mock_pool
    
    # Test delete by ID
    await pg_store.delete(ids=["1", "2"])
    
    assert mock_conn.execute.called
    args = mock_conn.execute.call_args
    assert "DELETE FROM test_embeddings" in args[0][0]
    assert args[0][1] == ["1", "2"]
    
    # Test delete with no args (should return False)
    mock_conn.execute.reset_mock()
    result = await pg_store.delete()
    assert result is False
    assert not mock_conn.execute.called


@pytest.mark.asyncio
async def test_table_exists(pg_store):
    # Setup mocks
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    pg_store.pool = mock_pool
    
    mock_conn.fetchval.return_value = True
    
    exists = await pg_store.table_exists("test_embeddings")
    assert exists is True
    assert mock_conn.fetchval.called
    assert "information_schema.tables" in mock_conn.fetchval.call_args[0][0]


@pytest.mark.asyncio
async def test_delete_table(pg_store):
    # Setup mocks
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    pg_store.pool = mock_pool
    
    await pg_store.delete_table("test_embeddings")
    
    mock_conn.execute.assert_called_with("DROP TABLE IF EXISTS test_embeddings")


@pytest.mark.asyncio
async def test_hybrid_search(pg_store):
    # Setup mocks
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    pg_store.pool = mock_pool
    
    # Setup search results
    vec_row = {"id": "1", "content": "vec match", "distance": 0.2, "metadata": json.dumps({})}
    sparse_row = {"id": "2", "content": "text match", "score": 0.8, "metadata": json.dumps({})}
    
    # Configure mock to return different results based on query
    # fetch is called by both search and sparse_search
    # We can use side_effect to return different values
    
    async def fetch_side_effect(sql, *args):
        if "ORDER BY embedding" in sql: # Vector search
            return [vec_row]
        elif "to_tsvector" in sql: # Sparse search
            return [sparse_row]
        return []
        
    mock_conn.fetch.side_effect = fetch_side_effect
    
    # We need to mock rrf_fusion as well since it might be imported inside the method
    with patch("openjiuwen.core.retrieval.utils.fusion.rrf_fusion") as mock_fusion:
        # Mock fusion result
        mock_fusion.return_value = [
            SearchResult(id="1", text="vec match", score=0.9, metadata={}),
            SearchResult(id="2", text="text match", score=0.8, metadata={})
        ]
        
        results = await pg_store.hybrid_search(
            query_text="query",
            query_vector=[0.1]*4,
            top_k=2
        )
        
        assert len(results) == 2
        assert mock_conn.fetch.call_count >= 2 # Called at least twice
        assert mock_fusion.called

