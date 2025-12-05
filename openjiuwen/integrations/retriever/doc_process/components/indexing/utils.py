# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import json
from typing import Iterator

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.retrieval.utils.milvus import iter_index
from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import milvus_manager

BATCH_SIZE = 256

empty_triples_count = 0


def _process_triples_for_chunk(
    chunk_id: str,
    chunk: str,
    file_id: str,
    chunk2triples: dict[str, list[list[str]]],
) -> Iterator[dict]:
    """Process triples for a single chunk and yield document dictionaries.

    Args:
        chunk_id: The chunk identifier
        chunk: The chunk text content
        file_id: The file identifier
        chunk2triples: Dictionary mapping chunks to extracted triples

    Yields:
        Document dictionaries containing text and metadata
    """
    global empty_triples_count

    triples = chunk2triples.get(chunk, None)
    if triples is None:
        logger.warning(f"chunk not found in chunk2triples: {chunk[:100]}...")
        return

    if not triples:
        empty_triples_count += 1
        return

    for triple in triples:
        if not isinstance(triple, list):
            raise TypeError(f"Expected list, got {type(triple)=}")

        if len(triple) == 0:
            continue

        triple = [str(x) for x in triple]

        yield {
            "text": " ".join(triple),
            "metadata": {
                "chunk_id": chunk_id,
                "triple": triple,
                "file_id": file_id,
            },
        }


def prepare_triples(
    chunk2triples: dict[str, list[list[str]]],
    config_obj,
    file_id: str = None,
) -> Iterator[dict]:
    """Generate document dictionaries from index and extracted triples.

    This dictionary is used to then build the triplets index.

    Args:
        backend: Backend type ("elasticsearch" or "milvus")
        chunk2triples: Dictionary mapping chunks to extracted triples
        config: Configuration object containing index settings
        file_id: Optional file ID to filter results

    Yields:
        Iterator[dict]: Document dictionaries containing text and metadata
    """
    global empty_triples_count
    empty_triples_count = 0  # Reset counter

    cfg = config_obj or CONFIG

    index_name = cfg.chunk_es_index

    yield from _prepare_triples_milvus(
        index_name=index_name,
        milvus_uri=cfg.milvus_uri,
        milvus_token=cfg.milvus_token,
        chunk2triples=chunk2triples,
        file_id=file_id,
    )

    logger.debug("Total number of empty triples: %d", empty_triples_count)


def _prepare_triples_milvus(
    index_name: str,
    milvus_uri: str,
    milvus_token: str = None,
    chunk2triples: dict[str, list[list[str]]] = None,
    file_id: str = None,
) -> Iterator[dict]:
    """Prepare triples from Milvus backend using MilvusClient API."""
    # Get MilvusClient from singleton manager
    client = milvus_manager.get_client(
        uri=milvus_uri,
        token=milvus_token,
    )

    try:
        # Build filter expression for file_id
        filter_expr = f'document_id == "{file_id}"' if file_id else None

        # Use iter_index for efficient batched iteration
        for batch in iter_index(
            client=client,
            collection_name=index_name,
            batch_size=BATCH_SIZE,
            output_fields=["pk", "content", "metadata", "document_id"],
            filter_expr=filter_expr,
            exclude_fields=["embedding"],
        ):
            for item in batch:
                chunk_id = str(item.get("pk", item.get("id", "unknown")))
                chunk = item.get("content", "")

                # Get file_id from metadata or document_id field
                metadata = item.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}

                item_file_id = metadata.get("file_id", "unknown")

                yield from _process_triples_for_chunk(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    file_id=item_file_id,
                    chunk2triples=chunk2triples,
                )
    finally:
        # Release client reference
        try:
            milvus_manager.release()
        except Exception:
            """Should ignore the error"""
            pass
