# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

__all__ = ["GraphBackend", "GraphBackendFactory"]

from openjiuwen.core.common.logging import logger

from .base import GraphBackendFactory
from .graph_backend import GraphBackend

try:
    # 将默认向量数据库后端注册入GraphBackendFactory
    from .milvus_support import MilvusBackend

    GraphBackendFactory.register_backend("milvus", MilvusBackend, force=False)
except ImportError:
    logger.warning("[图谱记忆]默认向量数据库后端（milvus）注册失败")
    raise
