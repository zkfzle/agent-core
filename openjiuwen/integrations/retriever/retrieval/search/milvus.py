# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Milvus-based retrieval wrapper.

Supports BM25 (sparse), vector search (dense), and hybrid search.
"""

import asyncio
from collections.abc import Callable
from typing import Any, List, Optional

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.retrieval.embed_models import EmbedModel
from openjiuwen.integrations.retriever.retrieval.search.retrieval_models import BaseRetriever as _BaseRetriever
from openjiuwen.integrations.retriever.retrieval.search.retrieval_models import (
    Dataset,
    Document,
    RetrievalResult,
    TextChunk,
)
from openjiuwen.integrations.retriever.retrieval.search.rrf import reciprocal_rank_fusion
from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import milvus_manager


def rrf_nodes(rankings: list[list[TextNode]], k: int = 60) -> list[TextNode]:
    """Merge ranked lists of nodes; 为混合检索写入融合后的 score."""
    from collections import defaultdict

    id2node = {}
    id_rankings = []
    for ranking in rankings:
        ids = []
        for node in ranking:
            if node.node_id not in id2node:
                id2node[node.node_id] = node
            ids.append(node.node_id)
        id_rankings.append(ids)

    # 计算 RRF 分数并写入 metadata['score']
    scores = defaultdict(float)
    for rlist in id_rankings:
        for rank, node_id in enumerate(rlist):
            scores[node_id] += 1.0 / (rank + k)

    ranked_ids = reciprocal_rank_fusion(id_rankings)

    fused: list[TextNode] = []
    for nid in ranked_ids:
        if nid not in id2node:
            raise RuntimeError(f"After RRF ranking, {nid=} not in {id2node=}")
        n = id2node[nid]
        n.metadata = n.metadata or {}
        n.metadata["score"] = scores.get(nid, 0.0)
        fused.append(n)
    return fused


def _scale_cosine(score: float | None) -> float | None:
    """Linearly map cosine similarity from [-1, 1] to [0, 1]."""
    if score is None:
        return None
    return (float(score) + 1.0) / 2.0


def _milvus_result_to_nodes(
    results: list[dict], text_field: str = "content", metadata_field: str = "metadata"
) -> list[TextNode]:
    """Convert Milvus search results to TextNode list with scores."""
    nodes: list[TextNode] = []
    for item in results:
        # Extract fields
        node_id = str(item.get("id", item.get("pk", "")))
        text = item.get(text_field, "")
        metadata = item.get(metadata_field, {})
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        # Get score (distance for vector search, score for BM25)
        score = item.get("score", item.get("distance", 0.0))

        # Ensure raw_score记录原始检索分数
        raw_score = float(score) if score is not None else None
        metadata.setdefault("raw_score", raw_score)
        metadata.setdefault("raw_score_scaled", _scale_cosine(raw_score))

        node = TextNode(
            id_=node_id,
            text=text,
            metadata=metadata,
        )
        node.metadata["score"] = float(score) if score else 0.0
        nodes.append(node)

    return nodes


class BaseMilvusWrapper:
    """Base wrapper for Milvus collections."""

    def __init__(
        self,
        collection_name: str,
        milvus_uri: str,
        milvus_token: str | None = None,
        text_field: str = "content",
        vector_field: str = "embedding",
        sparse_vector_field: str = "sparse_vector",
        metadata_field: str = "metadata",
    ) -> None:
        self.collection_name = collection_name
        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self.text_field = text_field
        self.vector_field = vector_field
        self.sparse_vector_field = sparse_vector_field
        self.metadata_field = metadata_field

        self._client = milvus_manager.get_client(
            uri=self.milvus_uri,
            token=self.milvus_token,
        )

    @staticmethod
    def __del__() -> None:
        try:
            milvus_manager.release()
        except Exception:
            """Should ignore the error"""
            pass

    @property
    def client(self) -> MilvusClient:
        return self._client


class BaseRetriever(BaseMilvusWrapper, _BaseRetriever):
    """Base class for retrieval using Milvus.

    Supports BM25 (text search via sparse vectors), vector search, and hybrid search.
    """

    def __init__(
        self,
        collection_name: str,
        milvus_uri: str,
        milvus_token: str | None = None,
        embed_model: EmbedModel | None = None,
        text_field: str = "content",
        vector_field: str = "embedding",
        sparse_vector_field: str = "sparse_vector",
        metadata_field: str = "metadata",
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            text_field=text_field,
            vector_field=vector_field,
            sparse_vector_field=sparse_vector_field,
            metadata_field=metadata_field,
        )

        self.embed_model = embed_model
        self.dataset = Dataset(title=collection_name, uri=f"{milvus_uri}/{collection_name}")

    def make_query(
        self,
        query: str | VectorStoreQuery,
        topk: int,
        mode: str,
        embed_model: EmbedModel | None = None,
        query_config: dict | None = None,
    ) -> VectorStoreQuery:
        """Construct a query."""
        if isinstance(query, str):
            query = VectorStoreQuery(
                query_str=query,
                similarity_top_k=topk,
                mode=mode,
                **(query_config or {}),
            )

        if query.query_embedding is None and query.mode != VectorStoreQueryMode.TEXT_SEARCH:
            embed_model = embed_model or self.embed_model
            if embed_model is None:
                raise RuntimeError("require embedding model for vector search")

            emb = embed_model.embed_query(query.query_str)
            query.query_embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)

        return query

    def search(
        self,
        query: str | VectorStoreQuery,
        topk: int = 5,
        mode: str | VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        custom_query: Callable[[dict[str, Any], str | None], dict[str, Any]] = None,
        *,
        query_config: dict | None = None,
    ) -> list[TextNode]:
        """Synchronous search wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.async_search(
                query=query,
                topk=topk,
                mode=mode,
                custom_query=custom_query,
                query_config=query_config,
            )
        )

    async def async_search(
        self,
        query: str | VectorStoreQuery,
        topk: int = 5,
        mode: str | VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        query_config: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[TextNode]:
        """Asynchronous search using Milvus."""
        query = self.make_query(query, topk=topk, mode=mode, query_config=query_config)
        logger.debug(
            "[BaseRetriever.async_search] collection=%r mode=%r topk=%r has_embed=%r",
            self.collection_name,
            query.mode,
            query.similarity_top_k,
            self.embed_model is not None,
        )

        if query.mode == VectorStoreQueryMode.DEFAULT:
            # Vector search
            nodes = await self._vector_search(query)
            if score_threshold is not None:
                nodes = [n for n in nodes if float(n.metadata.get("score", 0.0) or 0.0) >= score_threshold]
            logger.debug("[BaseRetriever.async_search] vector hits=%d", len(nodes))

            # Fallback to BM25 if vector search returns empty
            if not nodes:
                nodes = await self._text_search(query)
                logger.debug("[BaseRetriever.async_search] vector empty, bm25 fallback hits=%d", len(nodes))
            return nodes

        if query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            nodes = await self._text_search(query)
            logger.debug("[BaseRetriever.async_search] bm25 hits=%d", len(nodes))
            return nodes

        if query.mode == VectorStoreQueryMode.HYBRID:
            return await self._hybrid_search(query)

        raise NotImplementedError(f"unsupported {query.mode=}")

    async def _vector_search(self, query: VectorStoreQuery) -> list[TextNode]:
        """Perform vector similarity search."""
        if query.query_embedding is None:
            return []

        output_fields = [self.text_field, self.metadata_field, "document_id"]

        # Run search in thread pool since MilvusClient is sync
        results = await asyncio.to_thread(
            self._client.search,
            collection_name=self.collection_name,
            data=[query.query_embedding],
            anns_field=self.vector_field,
            limit=query.similarity_top_k,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE", "params": {}},
        )

        # Flatten results (search returns list of lists)
        if results and len(results) > 0:
            return _milvus_result_to_nodes(
                results[0],
                text_field=self.text_field,
                metadata_field=self.metadata_field,
            )
        return []

    async def _text_search(self, query: VectorStoreQuery) -> list[TextNode]:
        """Perform BM25 full-text search using Milvus sparse vector search.

        Uses the native BM25 function in Milvus for full-text search.
        The collection must have a sparse vector field with BM25 function enabled.
        """
        output_fields = [self.text_field, self.metadata_field, "document_id"]

        try:
            # Use native full-text search with BM25 metric
            results = await asyncio.to_thread(
                self._client.search,
                collection_name=self.collection_name,
                data=[query.query_str],  # Pass text directly, BM25 function handles tokenization
                anns_field=self.sparse_vector_field,
                limit=query.similarity_top_k,
                output_fields=output_fields,
                search_params={"metric_type": "BM25"},
            )

            # Flatten results (search returns list of lists)
            if results and len(results) > 0:
                return _milvus_result_to_nodes(
                    results[0],
                    text_field=self.text_field,
                    metadata_field=self.metadata_field,
                )
            return []
        except Exception as e:
            logger.warning(f"BM25 text search failed: {e}")
            return []

    async def _hybrid_search(self, query: VectorStoreQuery) -> list[TextNode]:
        """Perform hybrid search combining vector and BM25 search with native RRF ranking.

        Uses Milvus's native hybrid_search with RRFRanker for optimal performance.
        """
        output_fields = [self.text_field, self.metadata_field, "document_id"]

        try:
            # Build search requests for hybrid search
            search_requests = []

            # Dense vector search request
            if query.query_embedding is not None:
                dense_req = AnnSearchRequest(
                    data=[query.query_embedding],
                    anns_field=self.vector_field,
                    param={"metric_type": "COSINE", "params": {}},
                    limit=query.similarity_top_k,
                )
                search_requests.append(dense_req)

            # Sparse BM25 search request
            sparse_req = AnnSearchRequest(
                data=[query.query_str],  # Text directly, BM25 function tokenizes
                anns_field=self.sparse_vector_field,
                param={"metric_type": "BM25"},
                limit=query.similarity_top_k,
            )
            search_requests.append(sparse_req)

            if not search_requests:
                return []

            # Use native hybrid search with RRF ranking
            results = await asyncio.to_thread(
                self._client.hybrid_search,
                collection_name=self.collection_name,
                reqs=search_requests,
                ranker=RRFRanker(k=60),  # RRF with k=60
                limit=query.similarity_top_k,
                output_fields=output_fields,
            )

            if results and len(results) > 0:
                return _milvus_result_to_nodes(
                    results[0] if isinstance(results[0], list) else results,
                    text_field=self.text_field,
                    metadata_field=self.metadata_field,
                )
            return []

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to separate searches: {e}")
            # Fallback to manual RRF if native hybrid search fails
            return await self._hybrid_search_fallback(query)

    async def _hybrid_search_fallback(self, query: VectorStoreQuery) -> list[TextNode]:
        """Fallback hybrid search using separate searches and manual RRF fusion."""
        # Run both searches concurrently
        task_vector = asyncio.create_task(self._vector_search(query))
        task_text = asyncio.create_task(self._text_search(query))

        nodes_vector = await task_vector
        nodes_text = await task_text

        logger.debug(
            "[BaseRetriever._hybrid_search_fallback] vector_hits=%d text_hits=%d",
            len(nodes_vector),
            len(nodes_text),
        )

        # Merge with RRF
        if not nodes_vector and not nodes_text:
            return []
        if not nodes_vector:
            return nodes_text[: query.similarity_top_k]
        if not nodes_text:
            return nodes_vector[: query.similarity_top_k]

        return rrf_nodes([nodes_vector, nodes_text])[: query.similarity_top_k]

    def list_datasets(
        self,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> list[Dataset]:
        return [self.dataset] if not name or name == self.dataset.title else []

    def list_documents(self, document_id: str) -> list[Document]:
        """List documents by ID."""
        try:
            results = self._client.query(
                collection_name=self.collection_name,
                filter=f'document_id == "{document_id}"',
                output_fields=[self.text_field, self.metadata_field, "document_id"],
                limit=1,
            )

            if not results:
                return []

            doc = results[0]
            metadata = doc.get(self.metadata_field, {})
            if isinstance(metadata, str):
                import json

                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}

            return [
                Document(
                    document_id=document_id,
                    title=metadata.get("title", ""),
                    uri=None if CONFIG.get("hide_local_urls", True) else f"{self.dataset.uri}/{document_id}",
                    chunks=[TextChunk(content=doc.get(self.text_field, ""), similarity_score=1.0)],
                    metadata=metadata,
                )
            ]

        except Exception as e:
            logger.warning(f"Failed to get document {document_id}: {e}")
            return []

    def search_relevant_documents(
        self, question: str, datasets: Optional[List[Dataset]] = None, top_k: int = 5
    ) -> RetrievalResult:
        if datasets is None:
            datasets = []
        dataset_set = {(dataset.title, dataset.uri) for dataset in datasets}
        if dataset_set and (self.dataset.title, self.dataset.uri) not in dataset_set:
            return RetrievalResult(query=question, datasets=[], documents=[])

        results = self.search(query=question, topk=top_k)

        return RetrievalResult(
            query=question,
            datasets=[self.dataset],
            documents=[
                Document(
                    document_id=doc.id_,
                    title=doc.metadata.get("title", ""),
                    url=None if CONFIG.get("hide_local_urls", True) else f"{self.dataset.uri}/{doc.id_}",
                    chunks=[TextChunk(content=doc.text, similarity_score=doc.metadata.get("score", 1.0))],
                )
                for doc in results
            ],
        )


class BaseChunkRetriever(BaseRetriever):
    """Retriever that inherits BM25 text search from BaseRetriever.

    Note:
        BM25 search operates on the text content field. Title matching is handled
        via the BM25 scoring on content. For additional title-based filtering,
        consider using metadata filters in custom_query.

        Assumes "title" is in "metadata":
            {"metadata": {"title": ...}, "embedding": ..., "content": ...}
    """

    # Inherits _text_search from BaseRetriever which uses native BM25
    pass
