# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""索引删除接口（独立于索引构建）。"""

from __future__ import annotations

import asyncio
from typing import Optional, Tuple

import openjiuwen.integrations.retriever.config.configuration as grag_config
from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.doc_process.indexing import IndexConfig
from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import milvus_manager


def _collection_names(kb_id: str, chunk_collection: Optional[str], triple_collection: Optional[str]) -> Tuple[str, str]:
    """生成 chunk/triple collection 名；优先显式指定，否则按 kb_id 派生。"""
    chunk_col = chunk_collection or f"kb_{kb_id}_chunks"
    triple_col = triple_collection or f"kb_{kb_id}_triples"
    return chunk_col, triple_col


async def delete_indexes(
    kb_id: str,
    use_graph: bool = True,
    chunk_collection: Optional[str] = None,
    triple_collection: Optional[str] = None,
    config_obj: Optional[grag_config.GraphRAGConfig] = None,
) -> None:
    """
    删除指定 kb/collection 名的 chunk/triple collection。

    Args:
        kb_id: 索引标识（必填，用于派生默认 collection 名）。
        use_graph: 是否同时删除 triple collection（可选，默认 True）。
        chunk_collection: 显式指定 chunk collection 名（可选）。
        triple_collection: 显式指定 triple collection 名（可选）。
        config_obj: 配置对象（可选，默认使用全局 CONFIG）。
    """
    cfg = config_obj or grag_config.CONFIG
    if cfg is None:
        raise ValueError("config_obj (GraphRAGConfig) is required")

    chunk_col, triple_col = _collection_names(kb_id, chunk_collection, triple_collection)

    try:
        client = milvus_manager.get_client(
            uri=cfg.milvus_uri,
            token=getattr(cfg, "milvus_token", None),
        )

        # Delete chunk collection
        if client.has_collection(collection_name=chunk_col):
            await asyncio.to_thread(client.drop_collection, collection_name=chunk_col)
            logger.info(f"Deleted chunk collection: {chunk_col}")
        else:
            logger.debug(f"Chunk collection does not exist: {chunk_col}")

        # Delete triple collection if use_graph
        if use_graph:
            if client.has_collection(collection_name=triple_col):
                await asyncio.to_thread(client.drop_collection, collection_name=triple_col)
                logger.info(f"Deleted triple collection: {triple_col}")
            else:
                logger.debug(f"Triple collection does not exist: {triple_col}")

    except Exception as e:
        logger.error(f"Error deleting collections: {e}")
        raise


async def delete_document(doc_id: str, collection_info: IndexConfig) -> None:
    """
    按 doc_id 从指定 collection 中删除文档（chunk/triple）。

    Args:
        doc_id: 文档 ID（必填）。
        collection_info: 关于collection的信息
    """
    cfg = collection_info.external_config or grag_config.CONFIG
    if cfg is None:
        raise ValueError("index_config.external_config (GraphRAGConfig) is required")

    chunk_col, triple_col = _collection_names(
        collection_info.kb_id, collection_info.chunk_index, collection_info.triple_index
    )

    try:
        client = milvus_manager.get_client(
            uri=cfg.milvus_uri,
            token=getattr(cfg, "milvus_token", None),
        )

        # Delete from chunk collection by document_id filter
        if client.has_collection(collection_name=chunk_col):
            # Use delete with filter expression
            filter_expr = f'document_id == "{doc_id}"'
            await asyncio.to_thread(
                client.delete,
                collection_name=chunk_col,
                filter=filter_expr,
            )
            logger.info(f"Deleted document {doc_id} from chunk collection: {chunk_col}")
        else:
            logger.debug(f"Chunk collection does not exist: {chunk_col}")

        # Delete from triple collection if use_graph
        if collection_info.use_graph:
            if client.has_collection(collection_name=triple_col):
                filter_expr = f'document_id == "{doc_id}"'
                await asyncio.to_thread(
                    client.delete,
                    collection_name=triple_col,
                    filter=filter_expr,
                )
                logger.info(f"Deleted document {doc_id} from triple collection: {triple_col}")
            else:
                logger.debug(f"Triple collection does not exist: {triple_col}")

    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise


__all__ = ["delete_indexes", "delete_document"]
