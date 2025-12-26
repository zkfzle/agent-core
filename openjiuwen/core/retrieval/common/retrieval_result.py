# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
检索结果数据模型

包含 SearchResult 和 RetrievalResult 数据模型。
"""
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """搜索结果数据模型"""
    id: str = Field(..., description="结果 ID")
    text: str = Field(..., description="文本内容")
    score: float = Field(..., description="相关性分数")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="元数据"
    )


class RetrievalResult(BaseModel):
    """检索结果数据模型"""
    text: str = Field(..., description="文本内容")
    score: float = Field(..., description="相关性分数")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="元数据"
    )
    doc_id: Optional[str] = Field(None, description="文档 ID")
    chunk_id: Optional[str] = Field(None, description="块 ID")
