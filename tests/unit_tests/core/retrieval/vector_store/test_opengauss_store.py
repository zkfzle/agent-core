# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
OpenGaussVectorStore unit tests
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from openjiuwen.core.retrieval.common.config import VectorStoreConfig
from openjiuwen.core.retrieval.indexing.vector_fields.pg_fields import PGVectorField
from openjiuwen.core.retrieval.vector_store.opengauss_store import OpenGaussVectorStore


@pytest.fixture
def vector_store_config():
    return VectorStoreConfig(
        collection_name="test_collection",
        distance_metric="cosine",
    )


@pytest.fixture
def mock_session_factory():
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock()

    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=None)
    mock_session.begin.return_value = mock_transaction

    return mock_session


class TestOpenGaussVectorStore:
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_ensure_extension(
        self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory
    ):
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)

        store = OpenGaussVectorStore(config=vector_store_config, pg_uri="postgresql+asyncpg://user:pass@localhost/db")
        await store._ensure_extension()

        # Verify raw SQL
        args, _ = mock_session_factory.execute.call_args
        sql_obj = args[0]
        assert str(sql_obj) == "CREATE EXTENSION IF NOT EXISTS datavec"

    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_create_table_and_index_hnsw(
        self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory
    ):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.run_sync = AsyncMock(return_value=False)  # table exists = False
        mock_conn.execute = AsyncMock()

        mock_engine.connect.return_value = mock_conn
        mock_engine.begin.return_value = mock_conn

        # Default config uses HNSW implicitly via defaults (if PGVectorField used) or simple vector string
        # OpenGaussVectorStore init creates PGVectorField from string "embedding"
        store = OpenGaussVectorStore(config=vector_store_config, pg_uri="postgresql+asyncpg://user:pass@localhost/db")
        store._async_session = MagicMock(return_value=mock_session_factory)

        await store._get_or_create_table(dim=128)

        create_calls = [str(args[0]).strip() for args, _ in mock_conn.execute.call_args_list]

        # Verify Table
        create_table_sql = next((sql for sql in create_calls if "CREATE TABLE" in sql), None)
        assert create_table_sql is not None
        assert "VECTOR(128)" in create_table_sql

        # Verify Index
        index_sql = next((sql for sql in create_calls if "CREATE INDEX" in sql), None)
        assert index_sql is not None
        assert "USING vectors" in index_sql
        assert "index_type = 'hnsw'" in index_sql

    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_create_table_and_index_ivfflat(
        self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory
    ):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.run_sync = AsyncMock(return_value=False)  # table exists = False
        mock_conn.execute = AsyncMock()

        mock_engine.connect.return_value = mock_conn
        mock_engine.begin.return_value = mock_conn  # Reuse for transaction

        # Use explicit IVFFlat config with lists=50
        field = PGVectorField(vector_field="embedding", index_type="ivfflat", lists=50)
        store = OpenGaussVectorStore(
            config=vector_store_config, pg_uri="postgresql+asyncpg://user:pass@localhost/db", vector_field=field
        )

        store._async_session = MagicMock(return_value=mock_session_factory)

        await store._get_or_create_table(dim=128)

        create_calls = [str(args[0]).strip() for args, _ in mock_conn.execute.call_args_list]

        # Verify Index (IVFFlat)
        index_sql = next((sql for sql in create_calls if "CREATE INDEX" in sql), None)
        assert index_sql is not None
        assert "USING vectors" in index_sql
        assert "index_type = 'ivfflat'" in index_sql
        assert "nlists = 50" in index_sql

    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_search_sql_generation(
        self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory
    ):
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)

        store = OpenGaussVectorStore(config=vector_store_config, pg_uri="postgresql+asyncpg://user:pass@localhost/db")

        from sqlalchemy import Table, Column, MetaData, String
        from pgvector.sqlalchemy import Vector

        store._table = Table(
            "test_collection", MetaData(), Column("id", String, primary_key=True), Column("embedding", Vector(3))
        )

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session_factory.execute = AsyncMock(return_value=mock_result)

        await store.search([1.0, 2.0, 3.0])

        args, _ = mock_session_factory.execute.call_args
        stmt = args[0]
        compiled = str(stmt.compile(dialect=postgresql.dialect()))

        assert "<=>" in compiled
        assert "ORDER BY" in compiled

    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_dimension_discovery(
        self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory
    ):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_conn.run_sync = AsyncMock(return_value=True)  # table exists

        mock_result = MagicMock()
        mock_result.fetchone.return_value = ["vector(128)"]
        mock_conn.execute = AsyncMock(return_value=mock_result)

        mock_engine.connect.return_value = mock_conn

        store = OpenGaussVectorStore(config=vector_store_config, pg_uri="postgresql+asyncpg://user:pass@localhost/db")

        table = await store._get_or_create_table(dim=0)
        assert table.c.embedding.type.dim == 128

    @patch("openjiuwen.core.retrieval.vector_store.pg_store.create_async_engine")
    @patch("openjiuwen.core.retrieval.vector_store.pg_store.async_sessionmaker")
    @pytest.mark.asyncio
    async def test_delete(self, mock_sessionmaker, mock_create_engine, vector_store_config, mock_session_factory):
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session_factory)

        store = OpenGaussVectorStore(config=vector_store_config, pg_uri="postgresql+asyncpg://user:pass@localhost/db")

        from sqlalchemy import Table, Column, MetaData, String

        store._table = Table("test_collection", MetaData(), Column("id", String, primary_key=True))

        await store.delete(ids=["1", "2"])

        assert mock_session_factory.execute.called
