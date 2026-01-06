# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

__all__ = [
    "GraphConfig",
    "LLMConfig",
    "RetrievalStrategy",
    "DEFAULT_STRATEGY",
    "EpisodeRetrievalStrategy",
    "AddMemStrategy",
    "BaseRankConfig",
    "WeightedRankConfig",
    "RRFRankConfig",
    "register_database_ranking_cls",
    "register_database_query_language",
    "QueryLanguageDefinition",
    "GraphBackendFactory",
    "GraphBackend",
    "EpisodeType",
    "EmbeddingConfig",
    "SearchConfig",
]

from threading import Lock
from typing import Optional, Type

from openjiuwen.core.memory.config.graph.config import EmbeddingConfig, EpisodeType, GraphConfig, LLMConfig
from openjiuwen.core.memory.store.graph_store.backends import GraphBackend, GraphBackendFactory

from .extraction_strategies import (
    DEFAULT_STRATEGY,
    AddMemStrategy,
    EpisodeRetrievalStrategy,
    RetrievalStrategy,
    SearchConfig,
)
from .query_expr import QueryLanguageDefinition, register_database_query_language
from .result_ranking.rank_configs import RANKER_CLS, BaseRankConfig, RRFRankConfig, WeightedRankConfig

__ranker_register_lock = Lock()


def register_database_ranking_cls(
    name: str,
    base: Optional[Type],
    weighted: Optional[Type],
    rrf: Optional[Type],
    force: bool = False,
    raise_error: bool = True,
    **kwargs,
) -> bool:
    """Register result ranker classes for specific database"""
    with __ranker_register_lock:
        if name in RANKER_CLS and not force:
            if raise_error:
                raise ValueError(f'Database named "{name}" is already registered!')
            return False
        RANKER_CLS[name] = dict(base=base, weighted=weighted, rrf=rrf, **kwargs)
    return True
