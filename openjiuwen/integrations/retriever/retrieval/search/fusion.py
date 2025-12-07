# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import copy
import itertools
from typing import Any, List, Literal, Optional

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.retrieval.search.milvus import (
    BaseRetriever,
    rrf_nodes,
)
from openjiuwen.integrations.retriever.retrieval.search.retrieval_models import (
    BaseRetriever as _BaseRetriever,
)
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
        self._closed = False

    def search(
        self,
        query: str | VectorStoreQuery,
        topk: int = 5,
        mode: str | VectorStoreQueryMode = "default",
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
                topk_triples=topk_triples,
                query_config=query_config,
                graph_expansion=graph_expansion,
                graph_expansion_config=graph_expansion_config,
            )
        )

    async def close(self) -> None:
        """关闭内部 retriever 的 ES 客户端，避免连接未释放。"""
        if self._closed:
            return
        for r in (self.chunk_retriever, self.triple_retriever):
            if r and hasattr(r, "close"):
                try:
                    await r.close()
                except Exception:
                    logger.exception("关闭 GraphRetriever 内部连接失败")
        self._closed = True

    async def async_search(
        self,
        query: str | VectorStoreQuery,
        topk: int = 5,
        mode: str | VectorStoreQueryMode = "default",
        *,
        query_config: dict | None = None,
        graph_expansion: bool = False,
        graph_expansion_config: dict | None = None,
        score_threshold_vector: float | None = None,
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

            topk_triples (int | None, optional): Number of triples to match. Defaults to `5 * topk`.
            graph_expansion (bool, optional): Whether to do graph expansion. Defaults to False.
            query_config (dict | None, optional): Extra args for creating a `VectorStoreQuery`. Defaults to None.
                See: `llama_index.core.vector_stores.VectorStoreQuery`
            graph_expansion_config (dict | None, optional): Args for graph expansion. Defaults to None.
                See: `grag.search.triple.TripleBeamSearch`

        Raises:
        Returns:
            list[TextNode]: `topk` chunks.
        """
        query = self.chunk_retriever.make_query(
            query, topk=topk, mode=mode, query_config=query_config
        )
        nodes = await self.chunk_retriever.async_search(
            query, score_threshold=score_threshold_vector
        )

        logger.warning(
            "[graph] 图检索入口：graph_expansion=%s chunk命中=%d topk=%d mode=%s",
            graph_expansion,
            len(nodes),
            topk,
            mode,
        )
        if graph_expansion:
            logger.info(
                "[graph] 图扩展开启：query=%s 初始chunk数=%d",
                query.query_str,
                len(nodes),
            )
            nodes = await self.graph_expansion(
                query=query.query_str,
                chunks=nodes,
                query_embedding=query.query_embedding,
                **(graph_expansion_config or {}),
            )
            logger.info("[graph] 图扩展完成：扩展后chunk数=%d", len(nodes))

        # 如果是纯向量检索且传入了阈值，使用原始相似度(raw_score)过滤
        if mode == "default" and score_threshold_vector is not None:
            filtered = []
            for n in nodes:
                raw = n.metadata.get("raw_score", None)
                if "raw_score_scaled" not in n.metadata:
                    # 线性映射到 0-1，方便阈值
                    n.metadata["raw_score_scaled"] = (
                        (float(raw) + 1.0) / 2.0 if raw is not None else None
                    )
                scaled = n.metadata.get("raw_score_scaled", None)
                if scaled is not None and scaled >= score_threshold_vector:
                    filtered.append(n)
            nodes = filtered

        return nodes

    async def graph_expansion(
        self,
        query: str,
        chunks: list[TextNode],
        triples: list[TextNode] | None = None,
        topk: int | None = None,
        query_embedding: list[float] | None = None,
        **kwargs: Any,
    ) -> list[TextNode]:
        if not triples:
            # initial triples
            chunk_id2triples = await self._fetch_triples(chunks)
            triples = list(itertools.chain.from_iterable(chunk_id2triples.values()))
            logger.info(
                "[graph] 从 chunk 索引取三元组：chunks=%d triples=%d (无三元组=%d)",
                len(chunk_id2triples),
                len(triples),
                sum(1 for v in chunk_id2triples.values() if not v),
            )

            logger.info(
                "[graph] 从 chunk 索引取三元组：chunks=%d triples=%d (无三元组=%d)",
                len(chunk_id2triples),
                len(triples),
                sum(1 for v in chunk_id2triples.values() if not v),
            )

        triple_beam_search = TripleBeamSearch(retriever=self.triple_retriever, **kwargs)
        beams = await triple_beam_search.beam_search(query, triples)

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
        logger.info(
            "[graph] 图扩展 beam 结果：三元组=%d 补充chunk=%d",
            len(triples),
            len(new_chunks),
        )

        # 为补充的 chunk 计算原始向量分数，便于 score_threshold 生效
        if query_embedding and new_chunks:
            # chunk_id 为 uuid，使用 metadata["chunk_id"] 过滤
            chunk_ids = [c.node_id for c in new_chunks if c.node_id]
            if not chunk_ids:
                return nodes[:topk] if topk else nodes

            ids_expr = ",".join([f'"{cid}"' for cid in chunk_ids])
            expr = f'metadata["chunk_id"] in [{ids_expr}]'
            try:
                res = await asyncio.to_thread(
                    self.chunk_retriever.client.search,
                    collection_name=self.chunk_retriever.collection_name,
                    data=[query_embedding],
                    anns_field=self.chunk_retriever.vector_field,
                    limit=len(chunk_ids),
                    output_fields=[self.chunk_retriever.metadata_field],
                    filter=expr,
                    search_params={"metric_type": "COSINE", "params": {}},
                )
                if res and len(res) > 0:
                    scored = {
                        str(item.get("id", item.get("pk", ""))): float(
                            item.get("score", 0.0)
                        )
                        for item in res[0]
                    }
                    for c in new_chunks:
                        if c.node_id in scored:
                            c.metadata = c.metadata or {}
                            c.metadata["raw_score"] = scored[c.node_id]
                            c.metadata["raw_score_scaled"] = (
                                scored[c.node_id] + 1.0
                            ) / 2.0
            except Exception as e:
                logger.warning("[graph] 为扩展 chunk 计算向量分数失败: %s", e)

        nodes = rrf_nodes([new_chunks, chunks]) if new_chunks else chunks

        return nodes[:topk] if topk else nodes

    async def _search_hybrid_source(
        self,
        query: VectorStoreQuery,
        topk_triples: int | None = None,
        score_threshold_vector: float | None = None,
    ) -> tuple[list[TextNode], list[TextNode]]:
        """Search via hybrid data source and return (chunks, triples)."""
        chunks, (_chunks, triples) = await asyncio.gather(
            self.chunk_retriever.async_search(
                query, score_threshold=score_threshold_vector
            ),
            self._search_by_triples(
                copy.copy(query), topk_triples
            ),  # shallow copy is enough
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
        # 为基于三元组反查的 chunks 计算原始向量分数
        if query.query_embedding is not None and chunks:
            # chunk_id 为 uuid，使用 metadata.chunk_id 过滤
            chunk_ids = [c.node_id for c in chunks if c.node_id]
            if chunk_ids:
                ids_expr = ",".join([f'"{cid}"' for cid in chunk_ids])
                expr = f'metadata["chunk_id"] in [{ids_expr}]'
                try:
                    res = await asyncio.to_thread(
                        self.chunk_retriever.client.search,
                        collection_name=self.chunk_retriever.collection_name,
                        data=[query.query_embedding],
                        anns_field=self.chunk_retriever.vector_field,
                        limit=len(chunk_ids),
                        output_fields=[self.chunk_retriever.metadata_field],
                        filter=expr,
                        search_params={"metric_type": "COSINE", "params": {}},
                    )
                    if res and len(res) > 0:
                        scored = {
                            str(item.get("id", item.get("pk", ""))): float(
                                item.get("score", 0.0)
                            )
                            for item in res[0]
                        }
                        for c in chunks:
                            if c.node_id in scored:
                                c.metadata = c.metadata or {}
                                c.metadata["raw_score"] = scored[c.node_id]
                                c.metadata["raw_score_scaled"] = (
                                    scored[c.node_id] + 1.0
                                ) / 2.0
                except Exception as e:
                    logger.warning(
                        "[hybrid] 为三元组补充的 chunk 计算向量分数失败: %s", e
                    )
        query.similarity_top_k = _topk  # restore

        return chunks, triples

    async def _fetch_triples(self, chunks: list[TextNode]) -> dict[str, list[TextNode]]:
        """Return a dict mapping from each chunk's id to their triples."""
        chunk_id2triples: dict[str, list[TextNode]] = {x.node_id: [] for x in chunks}

        if not chunk_id2triples:
            return chunk_id2triples

        client = self.triple_retriever.client
        collection_name = self.triple_retriever.collection_name
        text_field = self.triple_retriever.text_field
        metadata_field = self.triple_retriever.metadata_field

        # Query triples for each chunk_id
        async def fetch_for_chunk(chunk_id: str) -> list[TextNode]:
            # Use filter to find triples associated with this chunk
            filter_expr = f'metadata["chunk_id"] == "{chunk_id}"'
            try:
                results = await asyncio.to_thread(
                    client.query,
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=[text_field, metadata_field, "document_id"],
                    limit=1000,  # Reasonable limit for triples per chunk
                )

                nodes = []
                for item in results:
                    metadata = item.get(metadata_field, {})

                    node = TextNode(
                        id_=str(item.get("pk", "")),
                        text=item.get(text_field, ""),
                        metadata=metadata,
                    )
                    nodes.append(node)
                return nodes
            except Exception as e:
                logger.warning(f"Failed to fetch triples for chunk {chunk_id}: {e}")
                return []

        # Fetch triples for all chunks concurrently
        tasks = [fetch_for_chunk(chunk_id) for chunk_id in chunk_id2triples]
        results = await asyncio.gather(*tasks)

        for chunk_id, triples in zip(chunk_id2triples.keys(), results):
            chunk_id2triples[chunk_id] = triples

        return chunk_id2triples

    async def _fetch_chunks(self, triples: list[TextNode]) -> list[TextNode]:
        """Return a list of associated chunks."""
        chunk_ids = deduplicate(
            node.metadata["chunk_id"] for node in triples if "chunk_id" in node.metadata
        )

        if not chunk_ids:
            return []

        client = self.chunk_retriever.client
        collection_name = self.chunk_retriever.collection_name
        text_field = self.chunk_retriever.text_field
        metadata_field = self.chunk_retriever.metadata_field

        async def fetch_chunk(chunk_id: str) -> TextNode | None:
            # Query by node_id or pk
            # Try filtering by document_id or using the chunk_id as pk
            try:
                pk_id = int(chunk_id)
                results = await asyncio.to_thread(
                    client.query,
                    collection_name=collection_name,
                    filter=f"pk == {pk_id}",
                    output_fields=[text_field, metadata_field, "document_id"],
                    limit=1,
                )
            except ValueError:
                pass

            if not results:
                return None

            item = results[0]
            metadata = item.get(metadata_field, {}) or {}
            if isinstance(metadata, str):
                import json

                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            # 保留原始分数，避免后续覆盖
            # 对于通过三元组反查的 chunk，这里没有原始向量分数，不做回退
            metadata.setdefault("raw_score", None)

            node = TextNode(
                id_=chunk_id,
                text=item.get(text_field, ""),
                metadata=metadata,
            )
            return node

        # Fetch all chunks concurrently
        tasks = [fetch_chunk(chunk_id) for chunk_id in chunk_ids]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        chunks = [node for node in results if node is not None]
        return chunks

    def list_datasets(
        self,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ):
        return self.chunk_retriever.list_datasets(
            name=name
        ) + self.triple_retriever.list_datasets(name=name)

    def list_documents(self, dataset_id: str, document_id: str):
        if dataset_id == self.chunk_retriever.collection_name:
            return self.chunk_retriever.list_documents(document_id)
        elif dataset_id == self.triple_retriever.collection_name:
            return self.triple_retriever.list_documents(document_id)
        return []

    def search_relevant_documents(
        self,
        question: str,
        datasets: Optional[List[Dataset]] = None,
        top_k: int = 5,
        graph_expansion: bool = False,
    ) -> RetrievalResult:
        if datasets is None:
            datasets = []
        dataset_set = {(dataset.title, dataset.uri) for dataset in datasets}
        self_dataset_set = {
            (dataset.title, dataset.uri) for dataset in self.list_datasets()
        }
        if dataset_set and dataset_set != self_dataset_set:
            return []

        results = self.search(
            query=question, topk=top_k, graph_expansion=graph_expansion
        )
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
