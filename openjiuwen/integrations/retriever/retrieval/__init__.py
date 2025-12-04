# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Retrieval interfaces exposed for RAG entrypoints.

This package hosts the KnowledgeBase interfaces expected by the frontend.
"""

from .retrieval_tool import KBQuery, KnowledgeBaseRetriever  # noqa: F401

__all__ = ["KnowledgeBaseRetriever", "KBQuery"]
