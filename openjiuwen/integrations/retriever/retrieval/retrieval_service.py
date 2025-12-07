# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
KB 侧的检索辅助方法（agentcore 提供，agentstudio 可直接调用）。

功能：
- 基于指定知识库的索引执行检索，支持图索引/Agentic。
"""

import logging
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

import openjiuwen.integrations.retriever.config.configuration as grag_config
from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.integrations.retriever.retrieval.embed_models import EmbedModel
from openjiuwen.integrations.retriever.retrieval.search.agents.base import SearchAgent
from openjiuwen.integrations.retriever.retrieval.search.fusion import GraphRetriever
from openjiuwen.integrations.retriever.retrieval.search.milvus import BaseRetriever
from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import (
    milvus_manager,
)


class KBQuery(BaseModel):
    """知识库查询对象"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(description="查询语句（必填）")
    retrieval_type: Literal["hybrid", "bm25", "vector"] = Field(
        default="hybrid", description="检索模式 text/vector/hybrid（可选，默认 hybrid）"
    )
    use_graph: bool = Field(
        default=False, description="是否启用三元组索引（可选，默认 False）"
    )
    topk: int = Field(default=5, description="返回条数上限（可选，默认 5）")
    score_threshold: float | None = Field(
        default=None, description="分数过滤阈值（可选，默认 None）"
    )
    graph_expansion: bool = Field(
        default=False, description="是否启用图扩展（可选，默认 False）"
    )
    use_agent: bool = Field(
        default=False, description="是否使用 agentic 检索（可选，默认 False）"
    )
    use_sync: bool = Field(
        default=True, description="Agentic 时是否同步调用（可选，默认 True）"
    )
    config_obj: Optional[grag_config.GraphRAGConfig] = Field(
        default=None, description="配置对象"
    )
    embed_model: Optional[EmbedModel] = Field(
        default=None, description="EmbedModel 实例，向量/混合检索必填"
    )
    llm_client: Optional[BaseModelClient] = Field(
        default=None, description="BaseModelClient 实例，图/Agentic 检索必填"
    )


def _collection_names(kb_id: str) -> Tuple[str, str]:
    """生成指定知识库的 chunk/triple collection 名。"""
    return f"kb_{kb_id}_chunks", f"kb_{kb_id}_triples"


def _attach_collections(
    cfg: grag_config.GraphRAGConfig, kb_id: str, index_type: str, use_graph: bool
) -> Tuple[str, str]:
    """写入 cfg 的 collection 名和开关。"""
    chunk_col, triple_col = _collection_names(kb_id)
    cfg.chunk_collection = chunk_col
    cfg.triple_collection = triple_col
    cfg.index_type = (index_type or "hybrid").lower()
    cfg.use_graph_index = use_graph
    return chunk_col, triple_col


def _mode_from_retrieval_type(rt: str) -> str:
    rt = (rt or "hybrid").lower()
    if rt in ("bm25", "text", "text_search"):
        return "text_search"
    if rt in ("vector", "dense"):
        return "default"
    return "hybrid"


async def search_kb(kb_id: str, query: KBQuery) -> List[str]:
    """
    基于指定知识库的索引执行检索，返回文本列表。

    Args:
        kb_id: 索引标识（必填，用于派生索引名）。
        query: 知识库查询对象（必填）。

    Returns:
        文本列表，长度不超过 query.topk。
    """
    cfg = query.config_obj or grag_config.CONFIG
    if cfg is None:
        raise ValueError("config_obj (GraphRAGConfig) is required for search_kb")
    chunk_col, triple_col = _attach_collections(
        cfg, kb_id, index_type="hybrid", use_graph=query.use_graph
    )
    mode = _mode_from_retrieval_type(query.retrieval_type)
    logger.debug(
        "[search_kb] kb_id=%r retrieval_type=%s mode=%s use_graph=%r topk=%d",
        kb_id,
        query.retrieval_type,
        mode,
        query.use_graph,
        query.topk,
    )
    logger.debug("[search_kb] chunk_col=%r triple_col=%r", chunk_col, triple_col)

    embed = query.embed_model or getattr(cfg, "embed_model_instance", None)
    logger.debug("[search_kb] embed_model provided=%r", embed is not None)
    if mode != "text_search" and embed is None:
        raise ValueError("embed_model_instance is required for vector/hybrid search")
    chunk_ret = BaseRetriever(
        collection_name=chunk_col,
        milvus_uri=cfg.milvus_uri,
        milvus_token=getattr(cfg, "milvus_token", None),
        embed_model=None if mode == "text_search" else embed,
    )

    def _filter_by_score(nodes):
        # 仅向量模式使用 score_threshold；bm25/hybrid 不用阈值。无分数时不过滤。
        if query.score_threshold is None or mode in {"text_search", "hybrid"}:
            return nodes
        filtered = []
        for n in nodes:
            val = _score_val(n)
            if val is None:
                raise ValueError("raw_score/raw_score_scaled missing for vector result; cannot apply score_threshold")
            if val >= query.score_threshold:
                filtered.append(n)
        return filtered

    if not query.use_graph:
        nodes = await chunk_ret.async_search(
            query=query.query, topk=query.topk, mode=mode
        )
        logger.debug("[search_kb] chunk-only hits=%d", len(nodes))
        nodes = _filter_by_score(nodes)
        return [n.text for n in nodes]

    triple_ret = BaseRetriever(
        collection_name=triple_col,
        milvus_uri=cfg.milvus_uri,
        milvus_token=getattr(cfg, "milvus_token", None),
        embed_model=None if mode == "text_search" else embed,
    )
    graph_ret = GraphRetriever(chunk_ret, triple_ret)
    score_thr = None if mode in {"text_search", "hybrid"} else query.score_threshold

    if query.use_agent:
        # 根据日志级别控制 Agent 详细输出，默认 DEBUG 时开启
        verbose = logger.isEnabledFor(logging.DEBUG)
        agent = SearchAgent(
            retriever=graph_ret,
            retriever_config={},
            use_sync=query.use_sync,
            use_agent=True,
            mode=mode,
            config_obj=cfg,
            llm_client=query.llm_client or getattr(cfg, "llm_client_instance", None),
            verbose=verbose,
        )
        nodes = await agent.search(query.query)
        logger.debug("[search_kb] agent hits=%d", len(nodes))
        return _filter_by_score(nodes)

    nodes = await graph_ret.async_search(
        query=query.query,
        topk=query.topk,
        mode=mode,
        graph_expansion=query.graph_expansion,
        score_threshold_vector=score_thr,
    )
    logger.debug(
        "[search_kb] graph search done: use_graph=%s graph_expansion=%s hits=%d",
        query.use_graph,
        query.graph_expansion,
        len(nodes),
    )
    nodes = _filter_by_score(nodes)
    return [n.text for n in nodes[: query.topk]]


def _score_val(node) -> float:
    """取向量分数：优先 raw_score_scaled，其次用 raw_score 线性映射；两者都无则返回 None。"""
    meta = getattr(node, "metadata", {}) or {}
    if meta.get("raw_score_scaled") is not None:
        try:
            return float(meta.get("raw_score_scaled"))
        except Exception:
            return None
    if meta.get("raw_score") is not None:
        try:
            raw = float(meta.get("raw_score"))
            return (raw + 1.0) / 2.0
        except Exception:
            return None
    return None


async def _collection_exists(milvus_uri: str, collection_name: str) -> bool:
    """轻量检查 Milvus collection 是否存在。"""
    try:
        client = milvus_manager.get_client(uri=milvus_uri)
        return client.has_collection(collection_name=collection_name)
    except Exception:
        return False


async def _search_kb_nodes(kb_id: str, query: KBQuery):
    """内部共用的检索，返回节点列表（不截断文本）。"""
    cfg = query.config_obj or grag_config.CONFIG
    if cfg is None:
        raise ValueError("config_obj (GraphRAGConfig) is required for search")
    chunk_col, triple_col = _attach_collections(
        cfg, kb_id, index_type="hybrid", use_graph=query.use_graph
    )
    mode = _mode_from_retrieval_type(query.retrieval_type)
    logger.debug(
        "[_search_kb_nodes] kb_id=%r retrieval_type=%s mode=%s use_graph=%r topk=%d",
        kb_id,
        query.retrieval_type,
        mode,
        query.use_graph,
        query.topk,
    )
    logger.debug("[_search_kb_nodes] chunk_col=%r triple_col=%r", chunk_col, triple_col)
    logger.debug(
        "[_search_kb_nodes] graph_expansion=%s use_agent=%s",
        query.graph_expansion,
        query.use_agent,
    )

    embed = query.embed_model or getattr(cfg, "embed_model_instance", None)
    logger.debug("[_search_kb_nodes] embed_model provided=%r", embed is not None)
    if mode != "text_search" and embed is None:
        raise ValueError("embed_model_instance is required for vector/hybrid search")
    chunk_ret = BaseRetriever(
        collection_name=chunk_col,
        milvus_uri=cfg.milvus_uri,
        milvus_token=getattr(cfg, "milvus_token", None),
        embed_model=None if mode == "text_search" else embed,
    )

    def _filter_by_score(nodes):
        # 仅纯向量模式应用 score_threshold；bm25/hybrid 不用阈值。无分数时不过滤。
        if query.score_threshold is None or mode in {"text_search", "hybrid"}:
            return nodes
        filtered = []
        for n in nodes:
            val = _score_val(n)
            if val is None:
                raise ValueError("raw_score/raw_score_scaled missing for vector result; cannot apply score_threshold")
            if val >= query.score_threshold:
                filtered.append(n)
        return filtered

    try:
        # 若 chunk collection 不存在，直接返回空或降级逻辑
        if not await _collection_exists(cfg.milvus_uri, chunk_col):
            if query.use_graph:
                return []
            return []

        # bm25 / hybrid 均不传阈值
        score_thr = None if mode in {"text_search", "hybrid"} else query.score_threshold

        if not query.use_graph:
            nodes = await chunk_ret.async_search(
                query=query.query, topk=query.topk, mode=mode, score_threshold=score_thr
            )
            logger.debug("[_search_kb_nodes] chunk-only hits=%d", len(nodes))
            return _filter_by_score(nodes)

        # 图检索前也检查 triple 索引
        if not await _collection_exists(cfg.milvus_uri, triple_col):
            # triple 不存在，降级为 chunk
            nodes = await chunk_ret.async_search(
                query=query.query, topk=query.topk, mode=mode, score_threshold=score_thr
            )
            logger.debug(
                "[_search_kb_nodes] triple missing, chunk fallback hits=%d", len(nodes)
            )
            return _filter_by_score(nodes)

        triple_ret = BaseRetriever(
            collection_name=triple_col,
            milvus_uri=cfg.milvus_uri,
            milvus_token=getattr(cfg, "milvus_token", None),
            embed_model=None if mode == "text_search" else embed,
        )
        graph_ret = GraphRetriever(chunk_ret, triple_ret)

        if query.use_agent:
            agent = SearchAgent(
                retriever=graph_ret,
                retriever_config={},
                use_sync=query.use_sync,
                use_agent=True,
                mode=mode,
                config_obj=cfg,
                llm_client=query.llm_client
                or getattr(cfg, "llm_client_instance", None),
            )
            nodes = await agent.search(query.query)
            logger.debug("[_search_kb_nodes] agent hits=%d", len(nodes))
            return _filter_by_score(nodes)

        logger.warning(
            "[graph] 进入图检索：kb_id=%s graph_expansion=%s use_graph=%s mode=%s",
            kb_id,
            query.graph_expansion,
            query.use_graph,
            mode,
        )
        nodes = await graph_ret.async_search(
            query=query.query,
            topk=query.topk,
            mode=mode,
            graph_expansion=query.graph_expansion,
            score_threshold_vector=score_thr,
        )
        # 向量模式下确保分数存在；图扩展回查失败时兜底
        if mode == "default":
            for n in nodes:
                meta = getattr(n, "metadata", {}) or {}
                if isinstance(meta, str):
                    import json

                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                if meta.get("raw_score_scaled") is None:
                    raw_score_val = meta.get("raw_score")
                    if raw_score_val is not None:
                        try:
                            meta["raw_score_scaled"] = (float(raw_score_val) + 1.0) / 2.0
                        except Exception:
                            meta["raw_score_scaled"] = None
                    elif getattr(n, "score", None) is not None:
                        try:
                            meta["raw_score_scaled"] = (float(n.score) + 1.0) / 2.0
                        except Exception:
                            meta["raw_score_scaled"] = None
                    else:
                        # 强制填 0，避免后续比较出现 None
                        meta["raw_score_scaled"] = 0.0
                n.metadata = meta
        logger.debug("[_search_kb_nodes] 图检索命中=%d", len(nodes))
        return _filter_by_score(nodes)

    except Exception as e:
        # 当索引不存在时，若请求 use_graph=True，降级为 chunk 检索；否则返回空
        if query.use_graph:
            try:
                nodes = await chunk_ret.async_search(
                    query=query.query, topk=query.topk, mode=mode
                )
                logger.debug(
                    "[_search_kb_nodes] NotFoundError chunk fallback hits=%d",
                    len(nodes),
                )
                return _filter_by_score(nodes)
            except Exception:
                return []
        return []


async def search_kb_multi(kb_ids: List[str], query: KBQuery) -> List[str]:
    """
    跨多个知识库检索：分别检索后用 RRF 融合（基于 rank），返回全局 topk。

    Note:
        - 当前使用简单 RRF（1/(rank+k)），忽略分值，适合不同量纲的向量/BM25。
        - 返回文本列表；如需来源，可在上层改返回结构携带 kb_id。

    Args:
        kb_ids: 知识库标识列表（必填）。
        query: 知识库查询对象（必填）。

    Returns:
        文本列表，长度不超过 query.topk。
    """
    if (query.config_obj or grag_config.CONFIG) is None:
        raise ValueError("config_obj (GraphRAGConfig) is required for search")
    logger.debug(
        "[search_kb_multi] kb_ids=%r retrieval_type=%s use_graph=%r topk=%d",
        kb_ids,
        query.retrieval_type,
        query.use_graph,
        query.topk,
    )
    # 多 KB 融合：对文本去重并按 score 全局排序
    merged: dict[str, float] = {}
    query_no_graph: Optional[KBQuery] = None
    for kid in kb_ids:
        try:
            nodes = await _search_kb_nodes(kb_id=kid, query=query)
        except Exception as e:
            # 若检索失败，若指定 use_graph=True，尝试降级为 chunk 检索
            logger.warning("[search_kb_multi] kid=%r failed: %s", kid, e)
            if query.use_graph:
                if query_no_graph is None:
                    query_no_graph = query.model_copy()
                    query_no_graph.use_graph = query_no_graph.graph_expansion = (
                        query_no_graph.use_agent
                    ) = False
                try:
                    nodes = await _search_kb_nodes(kb_id=kid, query=query_no_graph)
                except Exception:
                    """Should ignore the error"""
                    continue
            else:
                continue
        except Exception:
            """Should ignore the error"""
            continue

        logger.debug("[search_kb_multi] kid=%r hits=%d", kid, len(nodes))
        for n in nodes:
            txt = n.text
            sc = _score_val(n)
            merged[txt] = max(merged.get(txt, 0.0), sc)

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return [txt for txt, _ in ranked[: query.topk]]


async def search_kb_multi_with_source(kb_ids: List[str], query: KBQuery) -> List[dict]:
    """
    跨多个知识库检索，返回包含来源的结果列表：
    [{"kb_id": ..., "text": ...}, ...] （RRF 融合后截取 topk）。

    Args:
        kb_ids: 知识库标识列表（必填）。
        query: 知识库查询对象（必填）。

    Returns:
        列表，元素包含 text/score/kb_ids。
    """
    if (query.config_obj or grag_config.CONFIG) is None:
        raise ValueError("config_obj (GraphRAGConfig) is required for search")
    logger.debug(
        "[search_kb_multi_with_source] kb_ids=%r retrieval_type=%s use_graph=%r topk=%d",
        kb_ids,
        query.retrieval_type,
        query.use_graph,
        query.topk,
    )
    # 使用 score 全局排序并去重文本，来源汇总
    merged: dict[str, dict] = {}
    query_no_graph: Optional[KBQuery] = None
    for kid in kb_ids:
        try:
            nodes = await _search_kb_nodes(kb_id=kid, query=query)
        except Exception as e:
            logger.warning("[search_kb_multi_with_source] kid=%r failed: %s", kid, e)
            if query.use_graph:
                if query_no_graph is None:
                    query_no_graph = query.model_copy()
                    query_no_graph.use_graph = query_no_graph.graph_expansion = (
                        query_no_graph.use_agent
                    ) = False
                try:
                    nodes = await _search_kb_nodes(kb_id=kid, query=query_no_graph)
                except Exception:
                    nodes = []
            else:
                nodes = []

        logger.debug("[search_kb_multi_with_source] kid=%r hits=%d", kid, len(nodes))
        for n in nodes:
            txt = n.text
            sc_raw = _score_val(n)
            sc = 0.0 if sc_raw is None else float(sc_raw)  # 避免 None 参与比较
            meta = getattr(n, "metadata", {}) or {}
            if isinstance(meta, str):
                import json

                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            # 原始相似度：仅使用底层检索返回的 raw_score，不回退到融合分数
            raw_sc = meta.get("raw_score", None)
            raw_sc_scaled = meta.get("raw_score_scaled", None)
            if txt not in merged:
                merged[txt] = {
                    "text": txt,
                    "score": sc,
                    "raw_score": raw_sc,
                    "raw_score_scaled": raw_sc_scaled,
                    "kb_ids": set(),
                }
            merged[txt]["score"] = max(merged[txt]["score"], sc)
            if raw_sc is not None:
                merged[txt]["raw_score"] = max(
                    merged[txt].get("raw_score", raw_sc) or raw_sc, raw_sc
                )
            if raw_sc_scaled is not None:
                merged[txt]["raw_score_scaled"] = max(
                    merged[txt].get("raw_score_scaled", raw_sc_scaled) or raw_sc_scaled,
                    raw_sc_scaled,
                )
            merged[txt]["kb_ids"].add(kid)

    if not merged:
        return []

    ranked = sorted(
        (
            {
                "text": v["text"],
                "score": v["score"],
                "raw_score": v.get("raw_score", None),
                "raw_score_scaled": v.get("raw_score_scaled", None),
                "kb_ids": sorted(list(v["kb_ids"])),
            }
            for v in merged.values()
        ),
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[: query.topk]


def _rrf_ids(rankings: List[List[str]], k: int = 60) -> List[str]:
    """跨列表的简易 RRF，返回融合后的 id 顺序。"""
    from collections import defaultdict

    scores = defaultdict(float)
    for rlist in rankings:
        for rank, id_ in enumerate(rlist):
            scores[id_] += 1.0 / (rank + k)
    return [id_ for id_, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
