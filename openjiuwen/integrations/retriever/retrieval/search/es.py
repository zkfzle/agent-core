# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import copy
from collections.abc import Callable
from typing import Any, List, Optional

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers.vectorstore import AsyncBM25Strategy
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

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

# 避免测试收集时的循环 import，必要时定义一个轻量 BaseESWrapper 兜底
try:
    from openjiuwen.integrations.retriever.doc_process.components.indexing.base_indexer import (
        BaseESWrapper,  # type: ignore
    )
except Exception:  # pragma: no cover

    class BaseESWrapper:
        def __init__(
            self,
            es_index: str,
            es_url: str,
            es_client: AsyncElasticsearch | None = None,
        ) -> None:
            self.es_index = es_index
            self.es_url = es_url
            self.es_client = es_client or AsyncElasticsearch(self.es_url, timeout=600)
            self._es = ElasticsearchStore(index_name=self.es_index, es_client=self.es_client)

        @property
        def es(self) -> ElasticsearchStore:
            return self._es


def rrf_nodes(rankings: list[list[TextNode]], k: int = 60) -> list[TextNode]:
    """Merge ranked lists of nodes;为混合检索写入融合后的 score."""
    from collections import defaultdict

    id2node = {}
    id_rankings = []
    for ranking in rankings:
        ids = []
        for node in ranking:
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


def _extract_nodes(nodes_raw: list[Any]) -> list[TextNode]:
    """将 NodeWithScore/ TextNode 统一为 TextNode，并保留真实 score（若返回中无分数则保持 None/原值）。"""
    nodes: list[TextNode] = []
    for item in nodes_raw:
        if isinstance(item, TextNode):
            node = item
            score = getattr(item, "score", None)
        else:
            node = getattr(item, "node", None)
            score = getattr(item, "score", None)
        if node is None:
            continue
        if score is not None:
            try:
                score_val = float(score)
            except Exception:
                score_val = score
            node.metadata = node.metadata or {}
            node.metadata["score"] = score_val
        nodes.append(node)
    return nodes


def _attach_scores(nodes: list[TextNode], scores: list[Any] | None) -> list[TextNode]:
    """将检索结果中的相似度/相关性分数写入 metadata['score']。"""
    if not nodes or not scores:
        return nodes
    for n, s in zip(nodes, scores):
        try:
            score_val = float(s)
        except Exception:
            score_val = s
        n.metadata = n.metadata or {}
        n.metadata["score"] = score_val
    return nodes


class BaseRetriever(BaseESWrapper, _BaseRetriever):
    """Base class for retrieval.

    Support BM25, vector search, and hybrid search.

    """

    def __init__(
        self,
        es_index: str,
        es_url: str,
        embed_model: EmbedModel | None = None,
        es_client: AsyncElasticsearch | None = None,
    ) -> None:
        super().__init__(
            es_index=es_index,
            es_url=es_url,
            es_client=es_client,
        )

        self.embed_model = embed_model
        self._es_bm25: ElasticsearchStore | None = None

        self.dataset = Dataset(title=es_index, uri=es_url + "/" + es_index)

    @property
    def es_dense(self) -> ElasticsearchStore:
        return self.es

    @property
    def es_bm25(self) -> ElasticsearchStore:
        if self._es_bm25 is None:
            self._es_bm25 = ElasticsearchStore(
                index_name=self.es_index,
                es_client=self.es_client,
                retrieval_strategy=AsyncBM25Strategy(),
            )

        return self._es_bm25

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
        """Search.

        Args:
            query (str | VectorStoreQuery): Query.
            topk (int, optional): Top K to return. Defaults to 5.
                If `VectorStoreQuery` is given, `VectorStoreQuery.similarity_top_k` will be used instead.
            mode (str | VectorStoreQueryMode, optional): Query mode. Defaults to VectorStoreQueryMode.DEFAULT.
                "default" -> vector search
                "text_search" -> BM25
                "hybrid" -> hybrid search by merging results of vector search and BM25
            custome_query (Callable, optional): Function to customize the Elasticsearch query body. Defaults to None.
            query_config (dict, optional): Extra args to `VectorStoreQuery`. Defaults to None.

        Raises:
            NotImplementedError: Unsupported query mode.

        Returns:
            list[TextNode]: Top K retrieval results.

        """
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
        custom_query: Callable[[dict[str, Any], str | None], dict[str, Any]] = None,
        query_config: dict | None = None,
    ) -> list[TextNode]:
        """Asynchronous search."""
        query = self.make_query(query, topk=topk, mode=mode, query_config=query_config)
        logger.debug(
            "[BaseRetriever.async_search] index=%r mode=%r topk=%r has_embed=%r",
            self.es_index,
            query.mode,
            query.similarity_top_k,
            self.embed_model is not None,
        )

        if query.mode == VectorStoreQueryMode.DEFAULT:
            res = await self.es_dense.aquery(query, custom_query=custom_query)
            nodes = _attach_scores(_extract_nodes(res.nodes), getattr(res, "similarities", None))
            logger.debug("[BaseRetriever.async_search] vector hits=%d", len(nodes))
            # 若向量检索为空，自动降级 BM25 兜底，避免调用侧拿到空列表
            if not nodes:
                res_bm25 = await self.es_bm25.aquery(
                    VectorStoreQuery(
                        query_str=query.query_str,
                        similarity_top_k=query.similarity_top_k,
                        mode=VectorStoreQueryMode.TEXT_SEARCH,
                    ),
                    custom_query=custom_query,
                )
                nodes = _attach_scores(_extract_nodes(res_bm25.nodes), getattr(res_bm25, "similarities", None))
                logger.debug("[BaseRetriever.async_search] vector empty, bm25 fallback hits=%d", len(nodes))
            return nodes

        if query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            res = await self.es_bm25.aquery(query, custom_query=custom_query)
            logger.debug("[BaseRetriever.async_search] bm25 hits=%d", len(res.nodes))
            return _attach_scores(_extract_nodes(res.nodes), getattr(res, "similarities", None))

        if query.mode == VectorStoreQueryMode.HYBRID:
            return await self._hybrid_search(query, custom_query=custom_query)

        raise NotImplementedError(f"unsupported {query.mode=}")

    async def _hybrid_search(
        self,
        query: VectorStoreQuery,
        custom_query: Callable[[dict[str, Any], str | None], dict[str, Any]] = None,
    ) -> list[TextNode]:
        _query_mode = query.mode  # backup

        # Run Dense
        query.mode = VectorStoreQueryMode.DEFAULT
        task_dense = asyncio.create_task(self.es_dense.aquery(query, custom_query=custom_query))

        # Run BM25
        _query = copy.deepcopy(query)
        _query.mode = VectorStoreQueryMode.TEXT_SEARCH
        _query.query_embedding = None
        task_bm25 = asyncio.create_task(self.es_bm25.aquery(_query, custom_query=custom_query))

        # Synchronize
        nodes_dense = _extract_nodes((await task_dense).nodes)
        nodes_bm25 = _extract_nodes((await task_bm25).nodes)
        # 写入各自分数，便于后续阈值过滤
        nodes_dense = _attach_scores(nodes_dense, getattr((await task_dense), "similarities", None))
        nodes_bm25 = _attach_scores(nodes_bm25, getattr((await task_bm25), "similarities", None))
        logger.debug("[BaseRetriever._hybrid_search] dense_hits=%d bm25_hits=%d", len(nodes_dense), len(nodes_bm25))

        query.mode = _query_mode  # restore

        # RRF is not available with free license of Elasticsearch
        return rrf_nodes([nodes_dense, nodes_bm25])[: query.similarity_top_k]

    def list_datasets(
        self,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> list[Dataset]:
        return [self.dataset] if not name or name == self.dataset.title else []

    def list_documents(self, document_id: str) -> list[Document]:
        es = self.es.client
        try:
            doc = asyncio.run(
                es.get(
                    index=self.es.index_name,
                    id=document_id,
                    source_excludes=[self.es.vector_field, "metadata._node_content"],
                )
            )
            doc = doc["_source"]
            doc = Document(
                document_id=document_id,
                title=doc["metadata"]["title"],
                uri=None if CONFIG.get("hide_local_urls", True) else f"{self.dataset.uri}/_doc/{document_id}",
                chunks=[TextChunk(content=doc["content"], similarity_score=1.0)],
                metadata=doc["metadata"],
            )
            return [doc]

        except NotFoundError:
            return []

    def search_relevant_documents(
        self, question: str, datasets: Optional[List[Dataset]] = None, top_k: int = 5
    ) -> RetrievalResult:
        if datasets is None:
            datasets = []
        dataset_set = {(dataset.title, dataset.uri) for dataset in datasets}
        if dataset_set and (self.dataset.title, self.dataset.uri) not in dataset_set:
            return []

        results = self.search(
            query=question,
            topk=top_k,
        )
        result = RetrievalResult(
            query=question,
            datasets=[self.dataset],
            documents=[
                Document(
                    document_id=doc.id_,
                    title=doc.metadata["title"],
                    url=None if CONFIG.get("hide_local_urls", True) else f"{self.dataset.uri}/_doc/{doc.id_}",
                    chunks=[TextChunk(content=doc.text, similarity_score=1.0)],
                )
                for doc in results
            ],
        )
        return result


class BaseChunkRetriever(BaseRetriever):
    """Retriever that matches both title and content when performing BM25 search.

    Note:
        Assume "title" is in "metadata":
            {"metadata": {"title": ...}, "embedding": ..., "content": ...}

    """

    @staticmethod
    def should_match_title(body: dict, query: str) -> dict:
        try:
            bool_query = body["query"]["bool"]
            if not isinstance(bool_query, dict):
                return body

            must_clause = bool_query.pop("must")
            if not isinstance(must_clause, list):
                return body

        except KeyError:
            return body

        must_clause.append({"match": {"metadata.title": query}})
        bool_query["should"] = must_clause
        return body

    async def async_search(
        self,
        query: str | VectorStoreQuery,
        topk: int = 5,
        mode: str | VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        custom_query: Callable[[dict[str, Any], str | None], dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[TextNode]:
        if custom_query is None:
            if isinstance(query, VectorStoreQuery):
                mode = query.mode

            if mode in [VectorStoreQueryMode.TEXT_SEARCH, VectorStoreQueryMode.HYBRID]:
                custom_query = self.should_match_title

        return await super().async_search(
            query=query,
            topk=topk,
            mode=mode,
            custom_query=custom_query,
            **kwargs,
        )
