# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
openGauss Vector Store Implementation

Supports native openGauss vector search using datavec extension.
"""

import asyncio
import json
import logging
import re
from typing import Any, List, Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    text,
    inspect,
    Table,
    MetaData,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncEngine
from pgvector.sqlalchemy import Vector

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.common.logging import logger
from openjiuwen.core.retrieval.common.config import VectorStoreConfig
from openjiuwen.core.retrieval.indexing.vector_fields.pg_fields import PGVectorField
from openjiuwen.core.retrieval.vector_store.pg_store import PGVectorStore


class OpenGaussVectorStore(PGVectorStore):
    """openGauss vector store implementation using native datavec extension"""

    def __init__(
        self,
        config: VectorStoreConfig,
        pg_uri: str,
        text_field: str = "content",
        vector_field: str | PGVectorField = "embedding",
        sparse_vector_field: str = "sparse_vector",
        metadata_field: str = "metadata",
        doc_id_field: str = "document_id",
        **kwargs: Any,
    ):
        """
        Initialize openGauss vector store

        Args:
            config: Vector store configuration
            pg_uri: PostgreSQL connection URI (postgresql+asyncpg://...)
            text_field: Text field name
            vector_field: Vector field name (str) or definition (PGVectorField)
            sparse_vector_field: Sparse vector field name
            metadata_field: Metadata field name
            doc_id_field: Document ID field name
        """
        # Enforce constraints for inheritance compatibility
        if text_field != "content" or metadata_field != "metadata":
            raise ValueError(
                "OpenGaussVectorStore currently only supports text_field='content' and metadata_field='metadata'"
            )

        super().__init__(
            config, pg_uri, text_field, vector_field, sparse_vector_field, metadata_field, doc_id_field, **kwargs
        )

    @staticmethod
    def _validate_identifier(name: str) -> None:
        """Validate identifier to prevent SQL injection"""
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise build_error(StatusCode.RETRIEVAL_KB_DATABASE_CONFIG_INVALID, error_msg=f"Invalid identifier: {name}")

    async def _ensure_extension(self):
        """Ensure datavec extension exists"""
        async with self._async_session() as session:
            async with session.begin():
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS datavec"))

    async def _fetch_dim_from_db(self, table_name: str, col_name: str) -> int:
        """Fetch vector dimension from existing table"""
        # datavec stores type info in pg_attribute.atttypmod
        # For vector(128), atttypmod usually stores dimension info or we can use format_type
        sql = "SELECT format_type(atttypid, atttypmod) as type_str FROM pg_attribute WHERE attrelid = :table_name::regclass AND attname = :col_name"

        async with self._engine.connect() as conn:
            result = await conn.execute(text(sql), {"table_name": table_name, "col_name": col_name})
            row = result.fetchone()
            if not row:
                raise build_error(
                    StatusCode.RETRIEVAL_KB_DATABASE_CONFIG_INVALID,
                    error_msg=f"Column {col_name} not found in table {table_name}",
                )

            type_str = row[0]
            # Parse "vector(128)" or similar
            match = re.search(r"(?i)vector\((\d+)\)", type_str)
            if match:
                return int(match.group(1))

            # Fallback or error if type format is unexpected
            raise build_error(
                StatusCode.RETRIEVAL_KB_DATABASE_CONFIG_INVALID,
                error_msg=f"Could not determine dimension for column {col_name}. Type: {type_str}",
            )

    async def _get_or_create_table(self, dim: int = 0) -> Table:
        """Get existing table or create new one if dim is provided"""
        if self._table is not None:
            return self._table

        if self.table_ref is not None:
            self._table = self.table_ref
            return self._table

        async with self._engine.connect() as conn:
            table_exists = await conn.run_sync(lambda sync_conn: inspect(sync_conn).has_table(self.collection_name))

        if table_exists:
            # Discover dimension
            db_dim = await self._fetch_dim_from_db(self.collection_name, self.vector_col_name)

            if dim > 0 and db_dim != dim:
                raise build_error(
                    StatusCode.RETRIEVAL_KB_DATABASE_CONFIG_INVALID,
                    error_msg=f"Vector dimension mismatch. DB: {db_dim}, Config: {dim}",
                )
            dim = db_dim

            # Manually define table to ensure SQLAlchemy compatibility
            self._table = Table(
                self.collection_name,
                self._metadata,
                Column("id", String, primary_key=True),
                Column(self.text_field, Text),
                Column(self.metadata_field, JSONB),
                Column(self.vector_col_name, Vector(dim)),
            )
            return self._table

        # Create new table
        if dim == 0:
            return None  # Cannot create without dimension

        if dim > 2000:
            raise build_error(
                StatusCode.RETRIEVAL_KB_DATABASE_CONFIG_INVALID,
                error_msg=f"openGauss vector only supports up to 2000 dimensions (safe limit). Got {dim}.",
            )

        # Validate identifiers
        self._validate_identifier(self.collection_name)
        self._validate_identifier(self.text_field)
        self._validate_identifier(self.metadata_field)
        self._validate_identifier(self.vector_col_name)

        await self._ensure_extension()

        # Raw SQL Table Creation
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "{self.collection_name}" (
            id VARCHAR PRIMARY KEY,
            "{self.text_field}" TEXT,
            "{self.metadata_field}" JSONB,
            "{self.vector_col_name}" VECTOR({dim})
        )
        """

        async with self._engine.begin() as conn:
            await conn.execute(text(create_table_sql))

        # Manually define table object for runtime use
        self._table = Table(
            self.collection_name,
            self._metadata,
            Column("id", String, primary_key=True),
            Column(self.text_field, Text),
            Column(self.metadata_field, JSONB),
            Column(self.vector_col_name, Vector(dim)),
        )

        # Create Index
        index_name = f"idx_{self.collection_name}_{self.vector_col_name}"
        self._validate_identifier(index_name)

        index_method = self.vector_field_config.index_type

        if index_method == "hnsw":
            m = self.vector_field_config.m
            ef = self.vector_field_config.ef_construction
            # Using vectors ... WITH (index_type='hnsw', ...)
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS "{index_name}" ON "{self.collection_name}" 
            USING vectors ("{self.vector_col_name}") 
            WITH (index_type = 'hnsw', m = {m}, ef_construction = {ef})
            """
            async with self._engine.begin() as conn:
                await conn.execute(text(index_sql))

        elif index_method == "ivfflat":
            lists = getattr(self.vector_field_config, "lists", 100)  # Fallback if missing
            # Using vectors ... WITH (index_type='ivfflat', ...)
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS "{index_name}" ON "{self.collection_name}" 
            USING vectors ("{self.vector_col_name}") 
            WITH (index_type = 'ivfflat', nlists = {lists})
            """
            async with self._engine.begin() as conn:
                await conn.execute(text(index_sql))

        return self._table
