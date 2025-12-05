# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# GraphRAG component code is licensed under Mulan PSL v2.

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal

from llama_index.core.schema import TextNode
from pymilvus import DataType, Function, FunctionType, MilvusClient, MilvusException

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.doc_process.components.chunking.text_splitter import TextSplitter
from openjiuwen.integrations.retriever.retrieval.embed_models.base import EmbedModel
from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import milvus_manager


class BaseMilvusWrapper:
    """Base class that wraps Milvus collections using MilvusClient."""

    def __init__(
        self,
        collection_name: str,
        uri: str,
        token: str | None = None,
        text_field: str = "content",
        vector_field: str = "embedding",
        sparse_vector_field: str = "sparse_vector",
        metadata_field: str = "metadata",
    ) -> None:
        self.collection_name = collection_name
        self.uri = uri
        self.token = token
        self.text_field = text_field
        self.vector_field = vector_field
        self.sparse_vector_field = sparse_vector_field
        self.metadata_field = metadata_field

        self._client = milvus_manager.get_client(uri=self.uri, token=self.token)

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

    def collection_exists(self) -> bool:
        """Check if collection exists."""
        return self._client.has_collection(self.collection_name)

    def drop_collection(self) -> None:
        """Drop the collection if it exists."""
        if self.collection_exists():
            logger.info(f"Dropping collection: {self.collection_name}")
            self._client.drop_collection(self.collection_name)


class BaseMilvusIndexer(BaseMilvusWrapper, metaclass=ABCMeta):
    """Abstract base indexer for Milvus using MilvusClient API."""

    def __init__(
        self,
        collection_name: str,
        uri: str,
        token: str | None = None,
        embed_model: EmbedModel | None = None,
        splitter: TextSplitter | None = None,
        text_field: str = "content",
        vector_field: str = "embedding",
        sparse_vector_field: str = "sparse_vector",
        metadata_field: str = "metadata",
        enable_bm25: bool = True,
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            uri=uri,
            token=token,
            text_field=text_field,
            vector_field=vector_field,
            sparse_vector_field=sparse_vector_field,
            metadata_field=metadata_field,
        )
        if embed_model and not isinstance(embed_model, EmbedModel):
            raise TypeError(f"{type(embed_model)=}")
        self.embed_model = embed_model
        self.splitter = splitter
        self.enable_bm25 = enable_bm25

    @abstractmethod
    def preprocess(self, doc: dict, splitter: TextSplitter) -> list[TextNode]:
        """Preprocess a document and return chunks."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata_mappings(self) -> dict[str, Any]:
        """Return metadata schema information."""
        raise NotImplementedError

    def _ensure_collection(
        self,
        distance_strategy: Literal["L2", "IP", "COSINE"] = "COSINE",
        text_max_length: int = 4096,
    ) -> None:
        """Ensure collection exists, create if it doesn't."""
        if self._client.has_collection(self.collection_name):
            return

        # Build schema using MilvusClient approach
        schema = self._client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )

        # Primary key
        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )

        # Document ID field
        schema.add_field(
            field_name="document_id",
            datatype=DataType.VARCHAR,
            max_length=256,
        )

        # Text content field - enable tokenizer for BM25
        schema.add_field(
            field_name=self.text_field,
            datatype=DataType.VARCHAR,
            max_length=text_max_length,
            enable_analyzer=self.enable_bm25,  # Enable text analysis for BM25
            analyzer_params={"type": "standard"},  # Use standard analyzer
        )

        # Metadata JSON field
        schema.add_field(
            field_name=self.metadata_field,
            datatype=DataType.JSON,
        )

        # Prepare index params
        index_params = self._client.prepare_index_params()

        # Add scalar index on document_id for fast deletion/filtering
        index_params.add_index(
            field_name="document_id",
            index_type="INVERTED",  # Inverted index for VARCHAR
        )

        # Vector field (if embedding model is provided)
        if self.embed_model is not None:
            schema.add_field(
                field_name=self.vector_field,
                datatype=DataType.FLOAT_VECTOR,
                dim=self.embed_model.get_embedding_dimension(),
            )
            # Add dense vector index
            index_params.add_index(
                field_name=self.vector_field,
                index_type="IVF_FLAT",
                metric_type=distance_strategy,
                params={"nlist": 1024},
            )

        # BM25 sparse vector field for full-text search
        if self.enable_bm25:
            schema.add_field(
                field_name=self.sparse_vector_field,
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )

            # Add BM25 function to convert text to sparse vector
            bm25_function = Function(
                name="text_bm25_emb",
                input_field_names=[self.text_field],
                output_field_names=[self.sparse_vector_field],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)

            # Add sparse vector index for BM25
            index_params.add_index(
                field_name=self.sparse_vector_field,
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )

        # Create the collection
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Created collection: {self.collection_name}")

    def embed_nodes(self, nodes: list[TextNode], batch_size: int = 32) -> list[TextNode]:
        if self.embed_model is None:
            return nodes
        texts = [node.text for node in nodes]
        embeddings = self.embed_model.embed_docs(texts, batch_size=batch_size)
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return nodes

    def build_index(
        self,
        dataset: Iterable[dict],
        batch_size: int = 128,
        distance_strategy: Literal["L2", "IP", "COSINE"] = "COSINE",
        *,
        debug: bool = False,
    ) -> None:
        self._ensure_collection(distance_strategy=distance_strategy)
        datastream = dataset if not debug else list(dataset)[:100]

        cache: list[TextNode] = []
        for doc in datastream:
            cache.extend(self.preprocess(doc, self.splitter))
            if len(cache) >= batch_size:
                nodes = cache[:batch_size]
                cache = cache[batch_size:]
                self._insert_nodes(nodes, batch_size=batch_size)
        if cache:
            self._insert_nodes(cache, batch_size=batch_size)

        # Flush using client API
        self._client.flush(self.collection_name)

    def _insert_nodes(self, nodes: list[TextNode], batch_size: int = 32) -> None:
        nodes = self.embed_nodes(nodes, batch_size=batch_size)
        data = []
        for node in nodes:
            item = {
                "document_id": node.metadata.get("file_id"),
                self.text_field: node.text,
                self.metadata_field: node.metadata,
            }
            if self.embed_model is not None:
                item[self.vector_field] = node.embedding
            # Note: sparse_vector is auto-generated by BM25 function, don't include it
            data.append(item)

        self._client.insert(collection_name=self.collection_name, data=data)

    def delete_nodes(self, doc_id: str) -> int:
        """Delete all chunks belonging to doc_id."""
        filter_expr = f'document_id == "{doc_id}"'
        try:
            result = self._client.delete(
                collection_name=self.collection_name,
                filter=filter_expr,
            )
            # MilvusClient.delete returns dict with delete count or int
            if isinstance(result, dict):
                return result.get("delete_count", 0)
            return int(result) if result else 0
        except MilvusException as exc:
            logger.error("Failed to delete document %s from Milvus: %s", doc_id, exc)
            return 0
