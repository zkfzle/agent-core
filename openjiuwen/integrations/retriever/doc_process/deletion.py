# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""索引删除接口（独立于索引构建）。"""

from __future__ import annotations

from typing import Optional, Tuple

from elasticsearch import AsyncElasticsearch

import openjiuwen.integrations.retriever.config.configuration as grag_config


def _index_names(kb_id: str, chunk_index: Optional[str], triple_index: Optional[str]) -> Tuple[str, str]:
    """生成 chunk/triple 索引名；优先显式指定，否则按 kb_id 派生。"""
    chunk_idx = chunk_index or f"kb_{kb_id}_chunks"
    triple_idx = triple_index or f"kb_{kb_id}_triples"
    return chunk_idx, triple_idx


async def delete_indexes(
    kb_id: str, use_graph: bool = True, chunk_index: Optional[str] = None, triple_index: Optional[str] = None
) -> None:
    """
    删除指定 kb/索引名的 chunk/triple 索引。

    Args:
        kb_id: 索引标识（必填，用于派生默认索引名）。
        use_graph: 是否同时删除 triple 索引（可选，默认 True）。
        chunk_index: 显式指定 chunk 索引名（可选）。
        triple_index: 显式指定 triple 索引名（可选）。
    """
    chunk_idx, triple_idx = _index_names(kb_id, chunk_index, triple_index)
    aes = AsyncElasticsearch(grag_config.CONFIG.es_url)
    try:
        await aes.indices.delete(index=chunk_idx, ignore=[400, 404])
        if use_graph:
            await aes.indices.delete(index=triple_idx, ignore=[400, 404])
    finally:
        await aes.close()


async def delete_document(
    kb_id: str,
    doc_id: str,
    use_graph: bool = True,
    chunk_index: Optional[str] = None,
    triple_index: Optional[str] = None,
) -> None:
    """
    按 doc_id 从指定索引中删除文档（chunk/triple）。

    Args:
        kb_id: 索引标识（必填，用于派生默认索引名）。
        doc_id: 文档 ID（必填）。
        use_graph: 是否同时删除 triple 索引中的文档（可选，默认 True）。
        chunk_index: 显式指定 chunk 索引名（可选）。
        triple_index: 显式指定 triple 索引名（可选）。
    """
    chunk_idx, triple_idx = _index_names(kb_id, chunk_index, triple_index)
    aes = AsyncElasticsearch(grag_config.CONFIG.es_url)
    try:
        await aes.delete_by_query(index=chunk_idx, body={"query": {"term": {"file_id": doc_id}}}, ignore=[404])
        if use_graph:
            await aes.delete_by_query(index=triple_idx, body={"query": {"term": {"file_id": doc_id}}}, ignore=[404])
    finally:
        await aes.close()


__all__ = ["delete_indexes", "delete_document"]
