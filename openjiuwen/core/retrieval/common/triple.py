# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
三元组数据模型

包含 Triple 数据模型。
"""
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field


class Triple(BaseModel):
    """三元组数据模型"""
    subject: str = Field(..., description="主体")
    predicate: str = Field(..., description="谓词")
    object: str = Field(..., description="客体")
    confidence: Optional[float] = Field(None, description="置信度")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="元数据"
    )
