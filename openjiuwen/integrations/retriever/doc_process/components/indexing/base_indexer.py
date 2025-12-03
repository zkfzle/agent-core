# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import itertools
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import BulkIndexError
from llama_index.core.schema import TextNode
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.doc_process.components.chunking.text_splitter import TextSplitter
from openjiuwen.integrations.retriever.retrieval.embed_models import EmbedModel



class BaseESWrapper:
    """Base class that wraps Elasticsearch and Llamaindex."""

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

    def __del__(self) -> None:
        """Destructor to ensure async cleanup of Elasticsearch client."""
        if hasattr(self, "es_client") and self.es_client is not None:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._async_close())
                except RuntimeError:
                    asyncio.run(self._async_close())
            except Exception:
                """Should ignore the error"""
                pass

    async def _async_close(self):
        try:
            if self.es_client is not None:
                await self.es_client.close()
        except Exception:
            """Should ignore the error"""
            pass

    async def close(self):
        await self._async_close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    def es(self) -> ElasticsearchStore:
        return self._es


class BaseIndexer(BaseESWrapper, metaclass=ABCMeta):
    """Abstract base class for indexing.

    Notes:
        Need to implement data-specific preprocessing and define mappings for metadata.

    """

    def __init__(
        self,
        es_index: str,
        es_url: str,
        embed_model: EmbedModel | None = None,
        splitter: TextSplitter | None = None,
        es_client: AsyncElasticsearch | None = None,
    ) -> None:
        super().__init__(
            es_index=es_index,
            es_url=es_url,
            es_client=es_client,
        )

        if embed_model and not isinstance(embed_model, EmbedModel):
            raise TypeError(f"{type(embed_model)=}")

        self.embed_model = embed_model
        self.splitter = splitter

    @abstractmethod
    def preprocess(self, doc: dict, splitter: TextSplitter) -> list[TextNode]:
        """Preprocess a document and return a list of chunks."""
        pass

    @abstractmethod
    def get_metadata_mappings(self, **kwargs: Any) -> dict:
        """Return mappings for metadata.

        Examples:
            {"properties": {"title": {"type": "text"}}}

        """
        pass

    async def create_es_index(self, distance_strategy: str = "cosine", analyzer: str | None = None) -> None:
        """Create Elasticsearch index.

        如果 embed_model 为空，则只建文本/metadata 映射；否则同时建向量字段。
        """
        client: AsyncElasticsearch = self.es.client

        metadata_mappings = self.get_metadata_mappings(analyzer=analyzer)["properties"]
        if "doc_id" in metadata_mappings or "ref_doc_id" in metadata_mappings or "document_id" in metadata_mappings:
            raise ValueError(
                f"`doc_id`, `ref_doc_id`, `document_id` are occupied by LlamaIndex. "
                "We should use other field names to avoid potential conficts and/or unexpected behaviour."
            )

        props = {
            "metadata": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "ref_doc_id": {"type": "keyword"},
                    **metadata_mappings,
                }
            },
        }
        # 文本字段：vector-only 时不建立倒排（index: false），否则正常 text
        if self.embed_model is None:
            props[self.es.text_field] = {"type": "text"}  # bm25/hybrid 场景
        else:
            props[self.es.text_field] = (
                {"type": "text", "index": False} if getattr(self, "vector_only", False) else {"type": "text"}
            )

        # 仅在存在嵌入模型时添加向量字段映射
        if self.embed_model is not None:
            props[self.es.vector_field] = {
                "type": "dense_vector",
                "dims": self.embed_model.get_embedding_dimension(),
                "index": True,
                "similarity": distance_strategy,
            }

        await client.indices.create(
            index=self.es.index_name,
            mappings={"properties": props},
        )

    def embed_nodes(self, nodes: list[TextNode], batch_size: int = 32) -> list[TextNode]:
        if self.embed_model is None:
            return nodes

        texts = [node.text for node in nodes]
        eff_bsz = getattr(self.embed_model, "max_batch_size", None) or batch_size or 1
        embeddings = self.embed_model.embed_docs(texts, batch_size=eff_bsz)
        # 支持返回 list[list[float]] 或 tensor/ndarray
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes

    def build_index(
        self,
        dataset: Iterable[dict],
        batch_size: int = 128,
        distance_strategy: Literal["cosine", "dot_product", "l2_norm"] = "cosine",
        es_analyzer: str | None = None,
        *,
        debug: bool = False,
    ) -> None:
        """Build an Elasticsearch index for the input `dataset`.

        Note: If the index already exists, data will be added to it.

        Args:
            dataset (Iterable[dict]): Dataset of documents.
            batch_size (int, optional): Batch size for embedding passages. Defaults to 128.
            distance_strategy (str): Similarity metric supported by Elasticsearch. Defaults to cosine.
            es_analyzer (str, optional): Elasticsearch tokenizer for text field. Defaults to None.
            debug (bool, optional): Debug mode. Defaults to False.
                If True, index the first 100 documents only.
        """

        asyncio.run(
            self.build_index(
                dataset,
                batch_size=batch_size,
                distance_strategy=distance_strategy,
                es_analyzer=es_analyzer,
                debug=debug,
            )
        )

    async def build_index(
        self,
        dataset: Iterable[dict],
        batch_size: int = 128,
        distance_strategy: str = "cosine",
        es_analyzer: str | None = None,
        *,
        debug: bool = False,
    ) -> None:
        # 如果没有嵌入模型，只建文本索引
        client: AsyncElasticsearch = self.es.client
        index_exists = await client.indices.exists(index=self.es.index_name)

        if not index_exists:
            await self.create_es_index(distance_strategy=distance_strategy, analyzer=es_analyzer)
        await self._wait_for_index_ready()

        datastream = dataset
        if debug:
            datastream = itertools.islice(dataset, 100)

        cache = []
        logger.debug("Indexing documents...")
        for doc in datastream:
            cache.extend(self.preprocess(doc, self.splitter))

            if len(cache) > batch_size:
                logger.debug("Adding %d nodes to index", len(cache))
                nodes = cache[:batch_size]
                cache = cache[batch_size:]
                if self.embed_model is None:
                    await self._bulk_text_nodes(nodes)
                else:
                    await self._add_with_retry(self.embed_nodes(nodes, batch_size))

        if cache:
            logger.debug("Adding %d nodes to index", len(cache))
            if self.embed_model is None:
                await self._bulk_text_nodes(cache)
            else:
                await self._add_with_retry(self.embed_nodes(cache, batch_size))

        # Final refresh once after all batches
        try:
            await client.indices.refresh(index=self.es.index_name, request_timeout=120)
        except Exception as e:
            logger.warning("Refresh after indexing failed: %s", e)

    async def _bulk_text_nodes(self, nodes: list[TextNode]) -> None:
        """仅文本索引时的批量写入，避免向量字段缺失导致报错。"""
        client: AsyncElasticsearch = self.es.client
        actions = []
        for node in nodes:
            actions.append({"index": {"_index": self.es.index_name}})
            actions.append(
                {
                    "content": node.text,
                    "metadata": node.metadata,
                }
            )
        if actions:
            await self._wait_for_index_ready()
            # 不每批强制 refresh，写完统一 refresh
            await client.bulk(body=actions, refresh=False, request_timeout=120)

    async def async_delete_nodes(self, doc_id: str, refresh: bool = True) -> int:
        client: AsyncElasticsearch = self.es.client
        index_exists = await client.indices.exists(index=self.es.index_name)
        if not index_exists:
            return 0
        # Use term query against the keyword metadata field
        body = {"query": {"term": {"metadata.file_id": {"value": doc_id}}}}
        resp = await client.delete_by_query(index=self.es.index_name, body=body, refresh=refresh)
        return int(resp.get("deleted", 0))

    def delete_nodes(self, doc_id: str, refresh: bool = True) -> int:
        """
        Delete documents from Elasticsearch where `metadata.file_id == doc_id`.

        Args:
            doc_id (str): The document ID to delete.
            refresh (bool): Whether to refresh the index after deletion. Defaults to True.

        Returns:
            int: number of deleted documents.
        """
        return asyncio.run(self.async_delete_nodes(doc_id, refresh=refresh))

    async def _wait_for_index_ready(self, timeout: str = "60s") -> None:
        """确保索引 primary shard active 后再写入，减少 bulk 超时风险。"""
        client: AsyncElasticsearch = self.es.client
        try:
            await client.cluster.health(
                index=self.es.index_name,
                wait_for_status="yellow",
                wait_for_active_shards="1",
                timeout=timeout,
            )
        except Exception as e:
            logger.warning("Wait for index health failed (index=%s): %s", self.es.index_name, e)

    async def _add_with_retry(self, nodes: list[TextNode], *, request_timeout: int = 120) -> None:
        """调用 llama-index async_add，捕获 BulkIndexError 后自动降批重试。"""
        if not nodes:
            return
        try:
            await self._wait_for_index_ready()
            await self.es.async_add(
                nodes=nodes,
                create_index_if_not_exists=False,
                request_timeout=request_timeout,
            )
        except BulkIndexError as e:
            if len(nodes) <= 1:
                raise
            mid = len(nodes) // 2
            logger.warning(
                "BulkIndexError, retrying with smaller batches (%s -> %s + %s): %s",
                len(nodes),
                mid,
                len(nodes) - mid,
                e,
            )
            await self._add_with_retry(nodes[:mid], request_timeout=request_timeout)
            await self._add_with_retry(nodes[mid:], request_timeout=request_timeout)
