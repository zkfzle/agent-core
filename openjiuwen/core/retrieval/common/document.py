# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
文档数据模型

包含 Document 和 TextChunk 数据模型。
"""
import uuid
from typing import Dict, Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """文档数据模型"""
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()), description="文档 ID")
    text: str = Field(..., description="文档文本内容")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="文档元数据"
    )


class TextChunk(BaseModel):
    """文档块数据模型"""
    id_: str = Field(..., description="块 ID")
    text: str = Field(..., description="块文本内容")
    doc_id: str = Field(..., description="所属文档 ID")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="块元数据"
    )
    embedding: list[float] | None = Field(
        default=None, description="块的嵌入向量"
    )
    
    @classmethod
    def from_document(cls, doc: Document, chunk_text: str, id_: str = "") -> "TextChunk":
        """从 Document 创建 TextChunk"""
        return cls(
            id_=id_ if id_ else str(uuid.uuid4()),
            text=chunk_text,
            doc_id=doc.id_,
            metadata=doc.metadata,
        )
