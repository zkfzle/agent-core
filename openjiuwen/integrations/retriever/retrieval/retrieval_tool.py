# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
知识库检索包装类（统一异步入口，统一输入输出）。

约定：
- 输入：kb_ids 列表，单 KB 也传 ["kb-id"]。
- 输出：List[dict]，每条包含 text / score / kb_ids，便于前端统一解析。

特点：
- 参考 openjiuwen.core.utils.tool.base.Tool，提供异步 invoke。
- 内部复用 retrieval_service 的 search_kb_multi_with_source（多 KB 去重聚合；单 KB 也同样返回带来源结构）。
"""

from __future__ import annotations

from typing import Iterable, List

from openjiuwen.core.common.logging import logger

from . import retrieval_service
from .retrieval_service import KBQuery


class KnowledgeBaseRetriever:
    """知识库检索包装类，同步/异步皆可调用。

    约定（统一输入输出，单 KB 也遵循列表形式）：
    - 输入：kb_ids 列表，单 KB 也传 ["kb-id"]。
    - 输出：List[dict]，每条包含 text / score / kb_ids。

    Args:
        kb_ids: 知识库标识列表（必填）。
        retrieval_type: 检索模式 text/vector/hybrid（可选，默认 hybrid）。
        use_graph: 是否启用三元组索引（可选，默认 False）。
        source: GraphRetriever 源（可选，默认 hybrid）。
        topk: 返回条数上限（可选，默认 5）。
        score_threshold: 分数过滤阈值（可选，默认 None）。
        graph_expansion: 是否启用图扩展（可选，默认 False）。
        use_agent: 是否使用 agentic 检索（可选，默认 False）。
        use_sync: Agentic 时是否同步模式（可选，默认 True）。
    """

    def __init__(
        self,
        kb_ids: Iterable[str],
        retrieval_type: str = "hybrid",
        use_graph: bool = False,
        source: str = "hybrid",
        topk: int = 5,
        score_threshold: float | None = None,
        graph_expansion: bool = False,
        use_agent: bool = False,
        use_sync: bool = True,
        *,
        config_obj=None,
        llm_client=None,
        embed_model=None,
    ) -> None:
        """
        Args:
            kb_ids: 统一使用列表，即使单 KB 也传 ["kb-id"]。
            retrieval_type: vector / bm25 / hybrid。
            use_graph: 是否走图索引（需提前建 triple）。
            source: GraphRetriever 源，默认 hybrid。
            topk: 最大返回数量。
            score_threshold: 分数阈值过滤。
            graph_expansion: 是否图扩展。
            use_agent: 是否走 Agentic（LLM 费用）。
            use_sync: Agentic 时是否启用 SyncGE。
        """
        self.kb_ids = list(kb_ids)
        self.retrieval_type = retrieval_type
        self.use_graph = use_graph
        self.source = source
        self.topk = topk
        self.score_threshold = score_threshold
        self.graph_expansion = graph_expansion
        self.use_agent = use_agent
        self.use_sync = use_sync
        self.config_obj = config_obj
        self.llm_client = llm_client
        self.embed_model = embed_model

    async def invoke(self, query: str, **override) -> List[dict]:
        """
        异步调用入口。

        override 支持覆盖初始化时的参数，例如 kb_ids/topk/retrieval_type/use_graph 等。
        """
        kb_ids = list(override.get("kb_ids", self.kb_ids))
        retrieval_type = override.get("retrieval_type", self.retrieval_type)
        use_graph = override.get("use_graph", self.use_graph)
        source = override.get("source", self.source)
        topk = override.get("topk", self.topk)
        score_threshold = override.get("score_threshold", self.score_threshold)
        graph_expansion = override.get("graph_expansion", self.graph_expansion)
        use_agent = override.get("use_agent", self.use_agent)
        use_sync = override.get("use_sync", self.use_sync)
        config_obj = override.get("config_obj", self.config_obj)
        llm_client = override.get("llm_client", self.llm_client)
        embed_model = override.get("embed_model", self.embed_model)

        if not kb_ids:
            raise ValueError("kb_ids 不能为空；单 KB 也请传 ['kb-id']。")

        logger.debug(
            "[KnowledgeBaseRetriever.invoke] kb_ids=%r retrieval_type=%s use_graph=%r source=%s topk=%d "
            "embed_model_provided=%r llm_client_provided=%r",
            kb_ids,
            retrieval_type,
            use_graph,
            source,
            topk,
            embed_model is not None,
            llm_client is not None,
        )
        query_obj = KBQuery(
            query=query,
            retrieval_type=retrieval_type,
            use_graph=use_graph,
            source=source,
            topk=topk,
            score_threshold=score_threshold,
            graph_expansion=graph_expansion,
            use_agent=use_agent,
            use_sync=use_sync,
            config_obj=config_obj,
            llm_client=llm_client,
            embed_model=embed_model,
        )
        return await retrieval_service.search_kb_multi_with_source(kb_ids=kb_ids, query=query_obj)


__all__ = ["KnowledgeBaseRetriever", "KBQuery"]
