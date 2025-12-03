# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""文档处理阶段入口。"""
from .chunking import chunk_doc  # noqa: F401
from .deletion import delete_document, delete_indexes  # noqa: F401
from .indexing import IndexConfig, build_doc_index_from_chunks  # noqa: F401
from .parse import parse_doc  # noqa: F401

__all__ = ["chunk_doc", "parse_doc", "build_doc_index_from_chunks", "delete_document", "delete_indexes", "IndexConfig"]
