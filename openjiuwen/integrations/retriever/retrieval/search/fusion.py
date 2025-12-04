# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import copy
import itertools
from typing import Any, List, Literal, Optional

from elasticsearch import AsyncElasticsearch
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.retrieval.search.es import BaseRetriever, rrf_nodes
from openjiuwen.integrations.retriever.retrieval.search.retrieval_models import BaseRetriever as _BaseRetriever
from openjiuwen.integrations.retriever.retrieval.search.retrieval_models import (
    Dataset,
    Document,
    RetrievalResult,
    TextChunk,
)
from openjiuwen.integrations.retriever.retrieval.search.triple import TripleBeamSearch
from openjiuwen.integrations.retriever.retrieval.utils import deduplicate


class GraphRetriever(_BaseRetriever):
    def __init__(
        self,
        chunk_retriever: BaseRetriever,
        triple_retriever: BaseRetriever,
    ) -> None:
        """Graph retriever.

        Args:
            chunk_retriever (BaseRetriever): Retriever that returns chunks.
                Expected attributes of a chunk:
                    `TextNode.node_id` -> chunk_id
                    `TextNode.text` -> indexed content
                    `TextNode.metadata["title"]` -> document title

            triple_retriever (BaseRetriever): Retriever that returns triples.
                Expected attributes of a triple:
                    `TextNode.text` -> indexed content; e.g., "subject predicate object"
                    `TextNode.metadata["chunk_id"]` -> chunk_id pointing to the source chunk
                    `TextNode.metadata["triple"]` -> raw triple; e.g., ["subject", "predicate", "object"]

        """
        self.chunk_retriever = chunk_retriever
        self.triple_retriever = triple_retriever

    def search(
        self,
        query: str | VectorStoreQuery,
        topk: int = 5,
        mode: str | VectorStoreQueryMode = "default",
        source: Literal["hybrid", "chunks", "triples"] = "hybrid",
        topk_triples: int | None = None,
        *,
        query_config: dict | None = None,
        graph_expansion: bool = True,
        graph_expansion_config: dict | None = None,
    ) -> list[TextNode]:

        return asyncio.get_event_loop().run_until_complete(
            self.async_search(
                query=query,
                topk=topk,
                mode=mode,
                source=source,
                topk_triples=topk_triples,
                query_config=query_config,
                graph_expansion=graph_expansion,
                graph_expansion_config=graph_expansion_config,
            )
        )

    async def async_search(
        self,
        query: str | VectorStoreQuery,
        topk: int = 5,
        mode: str | VectorStoreQueryMode = "default",
        source: Literal["hybrid", "chunks", "triples"] = "hybrid",
        topk_triples: int | None = None,
        *,
        query_config: dict | None = None,
        graph_expansion: bool = False,
        graph_expansion_config: dict | None = None,
    ) -> list[TextNode]:
        """Search passages.

        Args:
            query (str | VectorStoreQuery): Query.
                If a `VectorStoreQuery` instance is given, `topk` and `mode` will be ignored.
            topk (int, optional): Number of passages to return. Defaults to 5.
            mode (str | VectorStoreQueryMode, optional): Retrieval mode. Defaults to "default".
                "default" -> Dense Retrieval;
                "text_search" -> BM25;
                "hybrid" -> Dense Retrieval + BM25;

            source (Literal["hybrid", "chunks", "triples"], optional): Data source to retrieve. Defaults to "hybrid".
                "chunks" -> Search chunks directly;
                "triples" -> Search chunks by matching triples;
                "hybrid" -> Search both chunks and triples;

            topk_triples (int | None, optional): Number of triples to match. Defaults to `5 * topk`.
            graph_expansion (bool, optional): Whether to do graph expansion. Defaults to False.
            query_config (dict | None, optional): Extra args for creating a `VectorStoreQuery`. Defaults to None.
                See: `llama_index.core.vector_stores.VectorStoreQuery`
            graph_expansion_config (dict | None, optional): Args for graph expansion. Defaults to None.
                See: `grag.search.triple.TripleBeamSearch`

        Raises:
            ValueError: If `source` is invalid.

        Returns:
            list[TextNode]: `topk` chunks.
        """
        query = self.chunk_retriever.make_query(query, topk=topk, mode=mode, query_config=query_config)
        if source == "chunks":
            nodes = await self.chunk_retriever.async_search(query)

        elif source == "hybrid":
            nodes, _ = await self._search_hybrid_source(query, topk_triples)
            nodes = nodes[:topk]

        elif source == "triples":
            nodes, _ = await self._search_by_triples(query, topk_triples)
            nodes = nodes[:topk]

        else:
            raise ValueError(f"unknown {source=}")

        if graph_expansion:
            nodes = await self.graph_expansion(
                query=query.query_str,
                chunks=nodes,
                **(graph_expansion_config or {}),
            )

        return nodes

    async def graph_expansion(
        self,
        query: str,
        chunks: list[TextNode],
        triples: list[TextNode] | None = None,
        topk: int | None = None,
        **kwargs: Any,
    ) -> list[TextNode]:
        if not triples:
            # initial triples
            chunk_id2triples = await self._fetch_triples(chunks)
            triples = list(itertools.chain.from_iterable(chunk_id2triples.values()))

        beams = TripleBeamSearch(retriever=self.triple_retriever, **kwargs)(query, triples)

        if not beams:
            return chunks[:topk] if topk else chunks

        max_length = max(len(x) for x in beams)
        triples = []
        for col in range(max_length):
            for beam in beams:
                if col >= len(beam):
                    continue
                triples.append(beam[col])

        new_chunks = await self._fetch_chunks(triples)

        nodes = rrf_nodes([new_chunks, chunks]) if new_chunks else chunks

        return nodes[:topk] if topk else nodes

    async def _search_hybrid_source(
        self,
        query: VectorStoreQuery,
        topk_triples: int | None = None,
    ) -> tuple[list[TextNode], list[TextNode]]:
        """Search via hybrid data source and return (chunks, triples)."""
        chunks, (_chunks, triples) = await asyncio.gather(
            self.chunk_retriever.async_search(query),
            self._search_by_triples(copy.copy(query), topk_triples),  # shallow copy is enough
        )
        chunks = rrf_nodes([chunks, _chunks])
        return chunks, triples

    async def _search_by_triples(
        self,
        query: VectorStoreQuery,
        topk_triples: int | None = None,
    ) -> tuple[list[TextNode], list[TextNode]]:
        """Search chunks by finding top-K triples and return (chunks, triples)."""
        _topk = query.similarity_top_k

        if topk_triples is None:
            topk_triples = _topk * 5

        query.similarity_top_k = topk_triples
        triples = await self.triple_retriever.async_search(query)
        # Note: len(chunks) <= len(triples) after deduplication
        chunks = await self._fetch_chunks(triples)
        query.similarity_top_k = _topk  # restore

        return chunks, triples

    async def _fetch_triples(self, chunks: list[TextNode]) -> dict[str, list[TextNode]]:
        """Return a dict mapping from each chunk's id to their triples."""
        chunk_id2triples: dict[str, list[TextNode]] = {x.node_id: [] for x in chunks}

        es: AsyncElasticsearch = self.triple_retriever.es.client
        responses = await asyncio.gather(
            *[
                es.search(
                    index=self.triple_retriever.es.index_name,
                    query={"term": {"metadata.chunk_id": id_}},
                    source_excludes=[self.triple_retriever.es.vector_field],
                )
                for id_ in chunk_id2triples
            ]
        )

        for resp in responses:
            hits = resp["hits"]["hits"]
            for hit in hits:
                node = TextNode.from_json(hit["_source"]["metadata"]["_node_content"])
                node.text = hit["_source"][self.triple_retriever.es.text_field]
                chunk_id2triples[node.metadata["chunk_id"]].append(node)

        return chunk_id2triples

    async def _fetch_chunks(self, triples: list[TextNode]) -> list[TextNode]:
        """Return a list of associated chunks."""
        chunk_ids = deduplicate(node.metadata["chunk_id"] for node in triples)
        es: AsyncElasticsearch = self.chunk_retriever.es.client
        responses = await asyncio.gather(
            *[
                es.get(
                    index=self.chunk_retriever.es.index_name,
                    id=id_,
                    source_excludes=[self.chunk_retriever.es.vector_field],
                )
                for id_ in chunk_ids
            ]
        )

        chunks = []
        for resp in responses:
            node = TextNode.from_json(resp["_source"]["metadata"]["_node_content"])
            node.text = resp["_source"][self.chunk_retriever.es.text_field]
            chunks.append(node)

        return chunks

    def list_datasets(
        self,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ):
        return self.chunk_retriever.list_datasets(name=name) + self.triple_retriever.list_datasets(name=name)

    def list_documents(self, dataset_id: str, document_id: str):
        if dataset_id == self.chunk_retriever.es_index:
            return self.chunk_retriever.list_documents(document_id)
        elif dataset_id == self.triple_retriever.es_index:
            return self.triple_retriever.list_documents(document_id)
        return []

    def search_relevant_documents(
        self, question: str, datasets: Optional[List[Dataset]] = None, top_k: int = 5, graph_expansion: bool = False
    ) -> RetrievalResult:
        if datasets is None:
            datasets = []
        dataset_set = {(dataset.title, dataset.uri) for dataset in datasets}
        self_dataset_set = {(dataset.title, dataset.uri) for dataset in self.list_datasets()}
        if dataset_set and dataset_set != self_dataset_set:
            return []

        results = self.search(query=question, topk=top_k, graph_expansion=graph_expansion)
        result = RetrievalResult(
            query=question,
            datasets=self.list_datasets(),
            documents=[
                Document(
                    document_id=doc.id_,
                    title=doc.metadata["title"],
                    url=(
                        None
                        if CONFIG.get("hide_local_urls", True)
                        else f"{self.chunk_retriever.dataset.uri}/_doc/{doc.id_}"
                    ),
                    chunks=[TextChunk(content=doc.text, similarity_score=1.0)],
                )
                for doc in results
            ],
        )
        return result
