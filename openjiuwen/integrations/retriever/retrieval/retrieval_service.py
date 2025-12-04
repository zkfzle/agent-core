# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
KB 侧的检索辅助方法（agentcore 提供，agentstudio 可直接调用）。

功能：
- 基于指定知识库的索引执行检索，支持图索引/Agentic。
"""

from typing import List, Literal, Optional, Tuple

from elasticsearch import AsyncElasticsearch, NotFoundError
from pydantic import BaseModel, ConfigDict, Field

import openjiuwen.integrations.retriever.config.configuration as grag_config
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.integrations.retriever.retrieval.embed_models import EmbedModel
from openjiuwen.integrations.retriever.retrieval.search.agents.base import SearchAgent
from openjiuwen.integrations.retriever.retrieval.search.es import BaseRetriever
from openjiuwen.integrations.retriever.retrieval.search.fusion import GraphRetriever
from openjiuwen.core.common.logging import logger


class KBQuery(BaseModel):
    """知识库查询对象"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(description="查询语句（必填）")
    retrieval_type: Literal["hybrid", "bm25", "vector"] = Field(
        default="hybrid", description="检索模式 text/vector/hybrid（可选，默认 hybrid）"
    )
    use_graph: bool = Field(default=False, description="是否启用三元组索引（可选，默认 False）")
    source: Literal["hybrid", "chunk", "triple"] = Field(
        default="hybrid", description="GraphRetriever的检索策略，hybrid/chunk/triple（可选，默认 hybrid）"
    )
    topk: int = Field(default=5, description="返回条数上限（可选，默认 5）")
    score_threshold: float | None = Field(default=None, description="分数过滤阈值（可选，默认 None）")
    graph_expansion: bool = Field(default=False, description="是否启用图扩展（可选，默认 False）")
    use_agent: bool = Field(default=False, description="是否使用 agentic 检索（可选，默认 False）")
    use_sync: bool = Field(default=True, description="Agentic 时是否同步调用（可选，默认 True）")
    config_obj: Optional[grag_config.GraphRAGConfig] = Field(default=None, description="配置对象")
    embed_model: Optional[EmbedModel] = Field(default=None, description="EmbedModel 实例，向量/混合检索必填")
    llm_client: Optional[BaseModelClient] = Field(default=None, description="BaseModelClient 实例，图/Agentic 检索必填")


def _index_names(kb_id: str) -> Tuple[str, str]:
    """生成指定知识库的 chunk/triple 索引名。"""
    return f"kb_{kb_id}_chunks", f"kb_{kb_id}_triples"


def _attach_indices(cfg: grag_config.GraphRAGConfig, kb_id: str, index_type: str, use_graph: bool) -> Tuple[str, str]:
    """写入 cfg 的索引名和开关，不再依赖 env。"""
    chunk_idx, triple_idx = _index_names(kb_id)
    cfg.chunk_es_index = chunk_idx
    cfg.triple_es_index = triple_idx
    cfg.index_type = (index_type or "hybrid").lower()
    cfg.use_graph_index = use_graph
    return chunk_idx, triple_idx


def _mode_from_retrieval_type(rt: str) -> str:
    rt = (rt or "hybrid").lower()
    if rt in ("bm25", "text", "text_search"):
        return "text_search"
    if rt in ("vector", "dense"):
        return "default"
    return "hybrid"


def _normalize_source(use_graph: bool, source: str) -> str:
    """If graph is requested but source不是 hybrid/triple，则强制为 hybrid，确保 chunk+triple 都参与。"""
    if not use_graph:
        return source or "hybrid"
    src = (source or "hybrid").lower()
    if src not in {"hybrid", "triple"}:
        return "hybrid"
    return src


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
    chunk_idx, triple_idx = _attach_indices(cfg, kb_id, index_type="hybrid", use_graph=query.use_graph)
    mode = _mode_from_retrieval_type(query.retrieval_type)
    source = _normalize_source(query.use_graph, query.source)
    logger.debug(
        "[search_kb] kb_id=%r retrieval_type=%s mode=%s use_graph=%r source=%s topk=%d",
        kb_id,
        query.retrieval_type,
        mode,
        query.use_graph,
        source,
        query.topk,
    )
    logger.debug("[search_kb] chunk_idx=%r triple_idx=%r", chunk_idx, triple_idx)

    embed = query.embed_model or getattr(cfg, "embed_model_instance", None)
    logger.debug("[search_kb] embed_model provided=%r", embed is not None)
    if mode != "text_search" and embed is None:
        raise ValueError("embed_model_instance is required for vector/hybrid search")
    chunk_ret = BaseRetriever(
        es_index=chunk_idx, es_url=cfg.es_url, embed_model=None if mode == "text_search" else embed
    )

    def _filter_by_score(nodes):
        if query.score_threshold is None:
            return nodes
        return [n for n in nodes if _score_val(n) >= query.score_threshold]

    if not query.use_graph:
        nodes = await chunk_ret.async_search(query=query.query, topk=query.topk, mode=mode)
        logger.debug("[search_kb] chunk-only hits=%d", len(nodes))
        nodes = _filter_by_score(nodes)
        return [n.text for n in nodes]

    triple_ret = BaseRetriever(
        es_index=triple_idx, es_url=cfg.es_url, embed_model=None if mode == "text_search" else embed
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
            llm_client=query.llm_client or getattr(cfg, "llm_client_instance", None),
        )
        nodes = await agent.search(query.query)
        logger.debug("[search_kb] agent hits=%d", len(nodes))
        nodes = _filter_by_score(nodes)
        return [getattr(n, "text", "") for n in nodes]

    nodes = await graph_ret.async_search(
        query=query.query,
        topk=query.topk,
        source=source,
        mode=mode,
        graph_expansion=query.graph_expansion,
    )
    logger.debug("[search_kb] graph hits=%d", len(nodes))
    nodes = _filter_by_score(nodes)
    return [n.text for n in nodes[: query.topk]]


def _score_val(node) -> float:
    """统一取 score，优先 metadata['score']，否则尝试属性；缺省返回 0。"""
    val = None
    try:
        val = (getattr(node, "metadata", {}) or {}).get("score")
    except Exception:
        val = None
    if val is None:
        try:
            val = getattr(node, "score", None)
        except Exception:
            val = None
    try:
        return 0.0 if val is None else float(val)
    except Exception:
        return 0.0


async def _index_exists(es_url: str, index: str) -> bool:
    """轻量检查索引是否存在，避免向不存在的索引发起查询。"""
    aes = AsyncElasticsearch(es_url)
    try:
        return await aes.indices.exists(index=index)
    except Exception:
        return False
    finally:
        await aes.close()


async def _search_kb_nodes(kb_id: str, query: KBQuery):
    """内部共用的检索，返回节点列表（不截断文本）。"""
    cfg = query.config_obj or grag_config.CONFIG
    if cfg is None:
        raise ValueError("config_obj (GraphRAGConfig) is required for search")
    chunk_idx, triple_idx = _attach_indices(cfg, kb_id, index_type="hybrid", use_graph=query.use_graph)
    mode = _mode_from_retrieval_type(query.retrieval_type)
    source = _normalize_source(query.use_graph, query.source)
    logger.debug(
        "[_search_kb_nodes] kb_id=%r retrieval_type=%s mode=%s use_graph=%r source=%s topk=%d",
        kb_id,
        query.retrieval_type,
        mode,
        query.use_graph,
        source,
        query.topk,
    )
    logger.debug("[_search_kb_nodes] chunk_idx=%r triple_idx=%r", chunk_idx, triple_idx)

    embed = query.embed_model or getattr(cfg, "embed_model_instance", None)
    logger.debug("[_search_kb_nodes] embed_model provided=%r", embed is not None)
    if mode != "text_search" and embed is None:
        raise ValueError("embed_model_instance is required for vector/hybrid search")
    chunk_ret = BaseRetriever(
        es_index=chunk_idx, es_url=cfg.es_url, embed_model=None if mode == "text_search" else embed
    )

    def _filter_by_score(nodes):
        if query.score_threshold is None:
            return nodes
        return [n for n in nodes if _score_val(n) >= query.score_threshold]

    try:
        # 若 chunk 索引不存在，直接返回空或降级逻辑
        if not await _index_exists(cfg.es_url, chunk_idx):
            if query.use_graph:
                return []
            return []

        if not query.use_graph:
            nodes = await chunk_ret.async_search(query=query.query, topk=query.topk, mode=mode)
            logger.debug("[_search_kb_nodes] chunk-only hits=%d", len(nodes))
            return _filter_by_score(nodes)

        # 图检索前也检查 triple 索引
        if not await _index_exists(cfg.es_url, triple_idx):
            # triple 不存在，降级为 chunk
            nodes = await chunk_ret.async_search(query=query.query, topk=query.topk, mode=mode)
            logger.debug("[_search_kb_nodes] triple missing, chunk fallback hits=%d", len(nodes))
            return _filter_by_score(nodes)

        triple_ret = BaseRetriever(
            es_index=triple_idx, es_url=cfg.es_url, embed_model=None if mode == "text_search" else embed
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
                llm_client=query.llm_client or getattr(cfg, "llm_client_instance", None),
            )
            nodes = await agent.search(query.query)
            logger.debug("[_search_kb_nodes] agent hits=%d", len(nodes))
            return _filter_by_score(nodes)

        nodes = await graph_ret.async_search(
            query=query.query,
            topk=query.topk,
            source=source,
            mode=mode,
            graph_expansion=query.graph_expansion,
        )
        logger.debug("[_search_kb_nodes] graph hits=%d", len(nodes))
        return _filter_by_score(nodes)

    except NotFoundError:
        # 当索引不存在时，若请求 use_graph=True，降级为 chunk 检索；否则返回空
        if query.use_graph:
            try:
                nodes = await chunk_ret.async_search(query=query.query, topk=query.topk, mode=mode)
                logger.debug("[_search_kb_nodes] NotFoundError chunk fallback hits=%d", len(nodes))
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
        "[search_kb_multi] kb_ids=%r retrieval_type=%s use_graph=%r source=%s topk=%d",
        kb_ids,
        query.retrieval_type,
        query.use_graph,
        query.source,
        query.topk,
    )
    # 多 KB 融合：对文本去重并按 score 全局排序
    merged: dict[str, float] = {}
    query_no_graph: Optional[KBQuery] = None
    for kid in kb_ids:
        try:
            nodes = await _search_kb_nodes(kb_id=kid, query=query)
        except NotFoundError:
            # 若指定 use_graph=True 但无三元组索引，自动降级为 chunk 检索
            if query.use_graph:
                if query_no_graph is None:
                    query_no_graph = query.model_copy()
                    query_no_graph.use_graph = query_no_graph.graph_expansion = query_no_graph.use_agent = False
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
        "[search_kb_multi_with_source] kb_ids=%r retrieval_type=%s use_graph=%r source=%s topk=%d",
        kb_ids,
        query.retrieval_type,
        query.use_graph,
        query.source,
        query.topk,
    )
    # 使用 score 全局排序并去重文本，来源汇总
    merged: dict[str, dict] = {}
    query_no_graph: Optional[KBQuery] = None
    for kid in kb_ids:
        try:
            nodes = await _search_kb_nodes(kb_id=kid, query=query)
        except NotFoundError:
            if query.use_graph:
                if query_no_graph is None:
                    query_no_graph = query.model_copy()
                    query_no_graph.use_graph = query_no_graph.graph_expansion = query_no_graph.use_agent = False
                try:
                    nodes = await _search_kb_nodes(kb_id=kid, query=query_no_graph)
                except Exception:
                    nodes = []
            else:
                nodes = []
        except Exception:
            nodes = []

        logger.debug("[search_kb_multi_with_source] kid=%r hits=%d", kid, len(nodes))
        for n in nodes:
            txt = n.text
            sc = _score_val(n)
            if txt not in merged:
                merged[txt] = {"text": txt, "score": sc, "kb_ids": set()}
            merged[txt]["score"] = max(merged[txt]["score"], sc)
            merged[txt]["kb_ids"].add(kid)

    if not merged:
        return []

    ranked = sorted(
        (
            {
                "text": v["text"],
                "score": v["score"],
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
