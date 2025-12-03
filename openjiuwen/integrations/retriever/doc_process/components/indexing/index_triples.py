# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import argparse
import asyncio
import json
from typing import Any

from elasticsearch import Elasticsearch
from llama_index.core.schema import TextNode

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.doc_process.components.chunking.text_splitter import TextSplitter
from openjiuwen.integrations.retriever.doc_process.components.indexing.base_indexer import BaseIndexer
from openjiuwen.integrations.retriever.doc_process.components.indexing.utils import prepare_triples



class TripleIndexer(BaseIndexer):
    def preprocess(self, doc: dict, splitter: TextSplitter) -> list[TextNode]:
        return [TextNode(text=doc["text"], metadata=doc["metadata"])]

    def get_metadata_mappings(self, **kwargs: Any) -> dict:
        return {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "triple": {"type": "text", "index": False},
                "file_id": {"type": "keyword"},  # Add file_id mapping for deletion queries
            }
        }


async def delete_triple_entries(doc_id: str) -> dict:
    """
    Delete triple entries from the triple index for a given document ID
    This function handles the deletion process for triple indices from the ES index

    Args:
        doc_id: The document ID to delete

    Returns:
        dict: Combined deletion result with success status and counts
    """
    try:
        # Delete from triple index
        logger.info(f"🔍 Deleting from triple index...")
        triple_indexer = TripleIndexer(
            es_index=CONFIG.triple_es_index,
            es_url=CONFIG.es_url,
        )
        triple_deleted_count = await triple_indexer.async_delete_nodes(doc_id)

        # Log results
        logger.info(f"Triple index deletion completed successfully:")
        logger.info(f"   Triple index: {triple_deleted_count} triples deleted")

    except Exception:
        logger.error(f"Error during triple index deletion")
        raise


async def index_triples(
    chunk2triples: dict = None,
    data_path: str = None,
    file_id: str = None,
    config_obj=None,
    embed_model=None,
):
    cfg = config_obj or CONFIG
    # 根据 index_type 决定是否计算向量
    embed_model = embed_model or getattr(cfg, "embed_model_instance", None)
    idx_type = (getattr(cfg, "index_type", None) or "hybrid").lower()
    if embed_model is None and idx_type in ("vector", "hybrid"):
        raise ValueError("embed_model is required when index_type is vector/hybrid (SDK 不再自动创建 embedding)")

    # 初始化索引器
    es = TripleIndexer(
        es_index=cfg.triple_es_index,
        es_url=cfg.es_url,
        embed_model=embed_model,
    )

    logger.info(f"🔍 正在构建三元组索引...")
    logger.info(f"   数据文件: {data_path}")
    logger.info(f"   三元组ES URL: {cfg.es_url}")
    logger.info(f"   三元组ES 索引: {cfg.triple_es_index}")
    logger.info(f"   文本ES URL: {cfg.es_url}")
    logger.info(f"   文本ES 索引: {cfg.chunk_es_index}")
    logger.info(f"   批处理大小: {cfg.batch_size}")
    logger.info("")

    if chunk2triples is None:
        chunk2triples = {}
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk2triples.update(json.loads(line))

    datastream = prepare_triples(Elasticsearch(cfg.es_url), chunk2triples, cfg.chunk_es_index, file_id=file_id)
    await es.build_index(datastream, batch_size=cfg.batch_size, debug=False)

    logger.info("三元组索引构建完成！")