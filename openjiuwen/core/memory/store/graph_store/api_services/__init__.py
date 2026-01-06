# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

__all__ = [
    "BaseReranker",
    "Qwen3Reranker",
    "GPT4oReranker",
    "GraphLLMClient",
    "StandardRerankService",
    "AliyunRerankService",
]

import os

from openjiuwen.core.memory.generation.graph.prompts import TemplateManager

from .llm_client import GraphLLMClient
from .llm_reranker import AliyunRerankService, BaseReranker, GPT4oReranker, Qwen3Reranker, StandardRerankService

TemplateManager().register_in_bulk(os.path.join(os.path.dirname(__file__), "prompts"), name="reranker")
