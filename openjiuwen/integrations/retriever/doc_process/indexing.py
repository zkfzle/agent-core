# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
索引构建接口（基于前端已切好的 chunks）。
默认按 kb_id 派生索引名，可显式传入自定义索引名。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from openjiuwen.core.common.logging import logger

import openjiuwen.integrations.retriever.config.configuration as grag_config
import openjiuwen.integrations.retriever.doc_process.components.pipeline.build_grag_index as build_mod
from openjiuwen.integrations.retriever.doc_process.components.indexing.index import delete_text_entries
from openjiuwen.integrations.retriever.doc_process.components.indexing.index_triples import delete_triple_entries


class IndexConfig(BaseModel):
    """建立索引所需配置项"""

    kb_id: str = Field(description="知识库标识，用于派生 chunk/triple 索引名（必填）")
    index_type: Literal["hybrid", "bm25", "vector"] = Field(
        default="hybrid", description="索引类型 hybrid/bm25/vector（可选，默认 hybrid）"
    )
    use_graph: bool = Field(default=True, description="是否抽取并写入三元组（可选，默认 True）")
    chunk_index: Optional[str] = Field(default=None, description="显式指定 chunk 索引名（可选，默认按 kb_id 派生）")
    triple_index: Optional[str] = Field(default=None, description="显式指定 triple 索引名（可选，默认按 kb_id 派生）")
    external_config: Optional[Any] = Field(default=None, description="可选外部传入的配置对象（SDK 模式）")


def _index_names(
    kb_id: str,
    chunk_index: Optional[str],
    triple_index: Optional[str],
) -> Tuple[str, str]:
    """生成 chunk/triple 索引名；优先显式指定，否则按 kb_id 派生。"""
    chunk_idx = chunk_index or f"kb_{kb_id}_chunks"
    triple_idx = triple_index or f"kb_{kb_id}_triples"
    return chunk_idx, triple_idx


def _set_env_for_index(index_config: IndexConfig):
    """设置索引名：优先直接写入 cfg（SDK 注入）。"""
    chunk_idx, triple_idx = _index_names(index_config.kb_id, index_config.chunk_index, index_config.triple_index)
    cfg = index_config.external_config
    if cfg is not None:
        cfg.chunk_es_index = chunk_idx
        cfg.triple_es_index = triple_idx
        cfg.index_type = (index_config.index_type or "hybrid").lower()
        cfg.use_graph_index = index_config.use_graph
    return chunk_idx, triple_idx


async def build_doc_index_from_chunks(
    doc_id: str,
    chunks: List[Dict[str, Any]],
    index_config: IndexConfig,
    embed_model: Optional[Any] = None,
    llm_client: Optional[Any] = None,
) -> bool:
    """
    基于前端已切好的 chunks 写入索引（chunk/triple）。

    Args:
        doc_id: 文档 ID（必填）。
        chunks: 已切好的片段列表，每项至少包含 chunk_text（必填）。
        index_config: 建立索引所需配置项。
        embed_model: 可选外部传入的 embedding 模型实例。
        llm_client: 可选外部传入的 LLM 客户端实例。

    Returns:
        True if success。
    """
    if not doc_id:
        raise ValueError("doc_id is required for build_doc_index_from_chunks")
    if not chunks:
        raise ValueError("chunks cannot be empty")

    # 若调用方传入 config/模型，则写入配置模块供下游使用（减少全局耦合）
    cfg = index_config.external_config or grag_config.CONFIG
    if cfg is None:
        raise ValueError("config is required (GraphRAGConfig)")
    if index_config.external_config is not None:
        import openjiuwen.integrations.retriever.config.configuration as cfg_mod  # local import

        cfg_mod.CONFIG = index_config.external_config
        grag_config.CONFIG = index_config.external_config
    if embed_model is not None:
        setattr(cfg, "embed_model_instance", embed_model)
    if llm_client is not None:
        setattr(cfg, "llm_client_instance", llm_client)
    if index_config.use_graph and getattr(cfg, "llm_client_instance", None) is None and llm_client is None:
        raise ValueError("llm_client is required when use_graph=True (SDK 不再自动创建 LLM 客户端)")

    logger.info(
        "Start indexing kb_id=%s doc_id=%s index_type=%s use_graph=%s chunk_count=%s",
        index_config.kb_id,
        doc_id,
        index_config.index_type,
        index_config.use_graph,
        len(chunks),
    )

    # 保留调用方传入的 external_config，避免索引名无法写回配置
    index_config = IndexConfig(
        kb_id=index_config.kb_id,
        index_type=index_config.index_type,
        use_graph=index_config.use_graph,
        chunk_index=index_config.chunk_index,
        triple_index=index_config.triple_index,
        external_config=cfg,
    )

    _set_env_for_index(index_config)
    cfg.precomputed_chunks = True
    cfg.chunk_unit = "char"

    # 清理同一 doc_id 旧数据，避免上次失败遗留的半截索引
    try:
        await delete_text_entries(doc_id)
        if index_config.use_graph:
            await delete_triple_entries(doc_id)
    except Exception as e:
        logger.warning("Failed to clean previous doc entries before re-indexing doc_id=%s: %s", doc_id, e)

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            for c in chunks:
                json.dump({"content": c.get("chunk_text", "")}, tmp)
                tmp.write("\n")
            tmp_path = tmp.name

        file_info = {
            "id": doc_id,
            "filename": f"{doc_id}.jsonl",
            "filepath": tmp_path,
        }
        grag_config_obj = build_mod.GRAGConfig(
            file=file_info,
            skip_triple_extraction=not index_config.use_graph,
            skip_triple_index=not index_config.use_graph,
            config_obj=cfg,
            embed_model=embed_model,
            llm_client=llm_client,
        )
        await build_mod.build_grag_index(config=grag_config_obj)
        logger.info(
            "✅ 索引构建完成 kb_id=%s doc_id=%s (use_graph=%s)", index_config.kb_id, doc_id, index_config.use_graph
        )
        return True
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                """Should ignore the error"""
                pass


__all__ = ["build_doc_index_from_chunks", "IndexConfig"]
