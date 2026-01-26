# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
PgVector Store Implementation

Supports vector search, sparse search (text matching), and hybrid search using PostgreSQL with pgvector extension.
"""

import asyncio
import json
from typing import Any, List, Optional

try:
    import asyncpg
    from pgvector.asyncpg import register_vector
except ImportError:
    asyncpg = None
    register_vector = None

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.common.logging import logger
from openjiuwen.core.retrieval.common.config import VectorStoreConfig
from openjiuwen.core.retrieval.common.retrieval_result import SearchResult
from openjiuwen.core.retrieval.vector_store.base import VectorStore


class PgVectorStore(VectorStore):
    """PostgreSQL vector store implementation using pgvector"""

    def __init__(
        self,
        config: VectorStoreConfig,
        dsn: str,
        table_name: str = "embeddings",
        text_field: str = "content",
        vector_field: str = "embedding",
        sparse_vector_field: str = "sparse_vector",
        metadata_field: str = "metadata",
        doc_id_field: str = "document_id",
        embedding_dim: int = 1536,
        **kwargs: Any,
    ):
        """
        Initialize PgVector store

        Args:
            config: Vector store configuration
            dsn: PostgreSQL connection string (postgresql://user:pass@host:port/db)
            table_name: Table name
            text_field: Text field name
            vector_field: Vector field name
            sparse_vector_field: Sparse vector field name (for hybrid search)
            metadata_field: Metadata field name
            doc_id_field: Document ID field name
            embedding_dim: Dimension of the embedding vector
        """
        if asyncpg is None:
            raise ImportError("asyncpg and pgvector are required for PgVectorStore." \
                              "Please install with `pip install asyncpg pgvector.")

        self.config = config
        self.dsn = dsn
        self.table_name = table_name
        self.text_field = text_field
        self.vector_field = vector_field
        self.sparse_vector_field = sparse_vector_field
        self.metadata_field = metadata_field
        self.doc_id_field = doc_id_field
        self.embedding_dim = embedding_dim
        self.collection_name = config.collection_name
        
        # Map distance metrics to PostgreSQL operators
        # <-> : Euclidean (L2) distance
        # <=> : Cosine distance
        # <#> : Inner product
        metric_map = {
            "cosine": "<=>",
            "euclidean": "<->",
            "dot": "<#>"
        }
        self.operator = metric_map.get(config.distance_metric, "<=>")
        
        self.pool: Optional[asyncpg.Pool] = None
        self._lock = asyncio.Lock()

    async def _init_pool(self) -> None:
        """Initialize connection pool and ensure table exists"""
        if self.pool:
            return

        async with self._lock:
            if self.pool:
                return
            
            try:
                self.pool = await asyncpg.create_pool(self.dsn)
                async with self.pool.acquire() as conn:
                    # Enable vector extension
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await register_vector(conn)
                    
                    # Create table
                    # We use a JSONB column for metadata to be flexible
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id TEXT PRIMARY KEY,
                            {self.text_field} TEXT,
                            {self.vector_field} vector({self.embedding_dim}),
                            {self.metadata_field} JSONB
                        )
                    """)
                    
                    # Create index for vector search (HNSW is recommended for performance)
                    # Note: index creation might be slow for large tables, 
                    # consider doing it manually or checking existence
                    # Here we assume basic setup. For production, index tuning is needed.
                    index_name = f"{self.table_name}_vec_idx"
                    # We skip automatic index creation to avoid blocking initialization on large tables
                    # await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON 
                    # {self.table_name} USING hnsw ({self.vector_field} vector_cosine_ops)")
                    
                    # Enable full text search extension if needed (pg_trgm etc) or just use to_tsvector
            except Exception as e:
                logger.error(f"Failed to initialize PgVectorStore: {e}")
                raise

    @staticmethod
    def create_client(database_name: str, path_or_uri: str, token: str = "", **kwargs) -> Any:
        """Create a client for the PostgreSQL vector store.

        This method serves as a placeholder in the `pg_store` implementation,
        as the actual database connection pool is managed asynchronously by
        the `_init_pool` method using `asyncpg`. It simply returns the
        connection URI.

        Args:
            database_name (str): The name of the database (unused).
            path_or_uri (str): The connection URI for the PostgreSQL database.
            token (str, optional): Authentication token (unused). Defaults to "".
            **kwargs: Additional keyword arguments (unused).

        Returns:
            str: The PostgreSQL connection URI provided in `path_or_uri`.
        """
        return path_or_uri

    def check_vector_field(self) -> None:
        """Check if vector field configuration is consistent with actual database"""
        # Since this method is synchronous and asyncpg is async, 
        # we cannot easily check the DB schema here without blocking.
        # We assume consistency or rely on _init_pool to handle setup.
        pass

    async def add(
        self,
        data: dict | List[dict],
        batch_size: int | None = 128,
        **kwargs: Any,
    ) -> None:
        """Add vectors"""
        await self._init_pool()
        
        if isinstance(data, dict):
            data = [data]
            
        if batch_size is None or batch_size <= 0:
            batch_size = 128

        async with self.pool.acquire() as conn:
            # Prepare data
            records = []
            for item in data:
                # Extract ID
                doc_id = str(item.get("id", item.get("pk", "")))
                if not doc_id:
                    import uuid
                    doc_id = str(uuid.uuid4())
                
                content = item.get(self.text_field, "")
                embedding = item.get(self.vector_field)
                
                # Construct metadata
                metadata = item.get(self.metadata_field, {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                
                # Ensure fields are preserved in metadata
                if self.doc_id_field in item and self.doc_id_field not in metadata:
                    metadata[self.doc_id_field] = item[self.doc_id_field]
                if "chunk_id" in item and "chunk_id" not in metadata:
                    metadata["chunk_id"] = item["chunk_id"]
                if self.sparse_vector_field in item:
                    # Store sparse vector in metadata as it's not a native column in this schema
                    metadata[self.sparse_vector_field] = item[self.sparse_vector_field]
                    
                records.append((doc_id, content, embedding, json.dumps(metadata)))

            # Batch insert using executemany
            # Note: We use ON CONFLICT to support upsert
            query = f"""
                INSERT INTO {self.table_name} (id, {self.text_field}, {self.vector_field}, {self.metadata_field})
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO UPDATE 
                SET {self.text_field} = EXCLUDED.{self.text_field}, 
                    {self.vector_field} = EXCLUDED.{self.vector_field},
                    {self.metadata_field} = EXCLUDED.{self.metadata_field}
            """
            
            # Process in batches
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                try:
                    await conn.executemany(query, batch)
                except Exception as e:
                    logger.error(f"Failed to insert batch into {self.table_name}: {e}")
                    raise

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Vector search"""
        await self._init_pool()
        
        # Build filter clause
        where_parts = ["1=1"]
        params = [query_vector]
        param_idx = 2
        
        if filters:
            for k, v in filters.items():
                # JSONB filter: metadata->>'key' = 'value'
                if isinstance(v, (int, float, bool)):
                    where_parts.append(f"{self.metadata_field}->>'{k}' = ${param_idx}::text")
                else:
                    where_parts.append(f"{self.metadata_field}->>'{k}' = ${param_idx}")
                params.append(str(v))
                param_idx += 1

        where_clause = " AND ".join(where_parts)
        
        # SQL query
        # We calculate score as 1 - distance (assuming cosine/L2 normalized context)
        # Note: pgvector distance operators return distance, not similarity
        # <=> cosine distance: 0 (same) to 2 (opposite)
        # We transform to similarity: 1 - (distance / 2) for cosine? 
        # Or just 1 - distance for simple cosine if normalized? 
        # Standard convention: Cosine Similarity = 1 - Cosine Distance
        
        sql = f"""
            SELECT id, {self.text_field}, {self.metadata_field}, 
                   ({self.vector_field} {self.operator} $1) as distance
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {self.vector_field} {self.operator} $1
            LIMIT {top_k}
        """
        
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(sql, *params)
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

        results = []
        for row in rows:
            dist = row['distance']
            
            # Convert distance to score based on metric
            if self.config.distance_metric == "cosine":
                score = 1.0 - dist # Cosine distance is 1 - Cosine Similarity
            elif self.config.distance_metric == "euclidean":
                score = 1.0 / (1.0 + dist) # Simple conversion
            else:
                score = -dist # Inner product: higher is better, but pgvector <#> is negative inner product
                
            metadata = json.loads(row[self.metadata_field])
            
            results.append(SearchResult(
                id=row['id'],
                text=row[self.text_field],
                score=score,
                metadata=metadata
            ))
            
        return results

    async def sparse_search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Sparse search (using PostgreSQL Full Text Search)"""
        await self._init_pool()
        
        # Build filter clause
        where_parts = ["1=1"]
        params = [query_text]
        param_idx = 2
        
        if filters:
            for k, v in filters.items():
                if isinstance(v, (int, float, bool)):
                    where_parts.append(f"{self.metadata_field}->>'{k}' = ${param_idx}::text")
                else:
                    where_parts.append(f"{self.metadata_field}->>'{k}' = ${param_idx}")
                params.append(str(v))
                param_idx += 1

        where_clause = " AND ".join(where_parts)
        
        # Use simple english configuration for tsvector
        # to_tsquery allows boolean search, plainto_tsquery takes plain text
        sql = f"""
            SELECT id, {self.text_field}, {self.metadata_field}, 
                   ts_rank_cd(to_tsvector('english', {self.text_field}), plainto_tsquery('english', $1)) as score
            FROM {self.table_name}
            WHERE {where_clause} AND to_tsvector('english', {self.text_field}) @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT {top_k}
        """
        
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(sql, *params)
            except Exception as e:
                logger.warning(f"Sparse search failed: {e}")
                return []

        results = []
        for row in rows:
            metadata = json.loads(row[self.metadata_field])
            results.append(SearchResult(
                id=row['id'],
                text=row[self.text_field],
                score=row['score'],
                metadata=metadata
            ))
            
        return results

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Hybrid search (sparse retrieval + vector retrieval)"""
        # Execute both searches concurrently
        task_vector = (
            asyncio.create_task(self.search(query_vector, top_k * 2, filters))
            if query_vector
            else asyncio.create_task(asyncio.sleep(0, result=[]))
        )
        task_text = asyncio.create_task(self.sparse_search(query_text, top_k * 2, filters))
        
        results_vector = await task_vector
        results_text = await task_text
        
        # Use RRF fusion from common utils if available, or simple weighted implementation
        from openjiuwen.core.retrieval.utils.fusion import rrf_fusion
        
        # Convert SearchResults to format expected by rrf_fusion if needed, 
        # or just pass lists if rrf_fusion handles SearchResult objects (it usually handles retrieval results)
        # Based on Milvus implementation, we might need to be careful with types.
        # rrf_fusion in openjiuwen usually expects RetrievalResult or SearchResult lists.
        # Checking imports in Milvus implementation: it uses rrf_fusion.
        
        fused_results = rrf_fusion([results_vector, results_text], k=60)
        return fused_results[:top_k]

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Delete vectors"""
        await self._init_pool()
        
        if not ids and not filter_expr:
            return False
            
        async with self.pool.acquire() as conn:
            try:
                if ids:
                    await conn.execute(f"DELETE FROM {self.table_name} WHERE id = ANY($1)", ids)
                    return True
                
                # Note: filter_expr support is limited here, would need SQL parsing
                # For now we log warning if only filter_expr is provided
                if filter_expr:
                    logger.warning("PgVectorStore delete currently supports 'ids' only, not generic 'filter_expr'")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to delete from {self.table_name}: {e}")
                return False
        return False

    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        await self._init_pool()
        async with self.pool.acquire() as conn:
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
            """, table_name)
            return exists

    async def delete_table(self, table_name: str) -> None:
        """Delete a table"""
        await self._init_pool()
        async with self.pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    def close(self) -> None:
        """Close connection pool"""
        if self.pool:
            asyncio.create_task(self.pool.close())