# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import json
from typing import Any

from llama_index.core.schema import TextNode
from pymilvus import MilvusException

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.doc_process.components.chunking.text_splitter import TextSplitter
from openjiuwen.integrations.retriever.doc_process.components.indexing.milvus_wrapper import BaseMilvusIndexer
from openjiuwen.integrations.retriever.doc_process.components.indexing.utils import prepare_triples
from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import milvus_manager


class MilvusTripleIndexer(BaseMilvusIndexer):
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


async def delete_triple_entries(doc_id: str, config_obj=None) -> dict:
    """
    Delete triple entries from the triple index for a given document ID
    This function handles the deletion process for triple indices

    Args:
        doc_id: The document ID to delete

    Returns:
        dict: Combined deletion result with success status and counts
    """
    cfg = config_obj or CONFIG

    client = milvus_manager.get_client(
        uri=cfg.milvus_uri,
        token=getattr(cfg, "milvus_token", None),
    )

    try:
        logger.info("Deleting from text index...")

        filter_expr = f'document_id == "{doc_id}"'
        try:
            result = client.delete(
                collection_name=cfg.triple_es_index,
                filter=filter_expr,
            )
            # MilvusClient.delete returns dict with delete count or int
            if isinstance(result, dict):
                return result.get("delete_count", 0)
            return int(result) if result else 0
        except MilvusException as exc:
            logger.error("Failed to delete document %s from Milvus: %s", doc_id, exc)
            return 0

    except Exception as e:
        logger.error("Error during text chunk deletion: %s", e)
        raise e


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
    triple_indexer = MilvusTripleIndexer(
        collection_name=cfg.triple_es_index,
        uri=cfg.milvus_uri,
        token=cfg.milvus_token,
        embed_model=embed_model,
    )

    # vector-only 时关闭倒排（仅向量）；bm25/hybrid 默认保留文本索引
    index_type = (cfg.index_type or "hybrid").lower()
    if index_type == "vector":
        setattr(triple_indexer, "vector_only", True)
    else:
        setattr(triple_indexer, "vector_only", False)

    logger.info("正在构建三元组索引...")
    logger.info(f"   数据文件: {data_path}")
    logger.info(f"   三元组ES 索引: {cfg.triple_es_index}")
    logger.info(f"   文本ES 索引: {cfg.chunk_es_index}")
    logger.info(f"   批处理大小: {cfg.batch_size}")
    logger.info("")

    if chunk2triples is None:
        chunk2triples = {}
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk2triples.update(json.loads(line))

    datastream = prepare_triples(chunk2triples=chunk2triples, config_obj=cfg, file_id=file_id)

    await asyncio.to_thread(triple_indexer.build_index, datastream, batch_size=cfg.batch_size, debug=False)

    logger.info("三元组索引构建完成！")
