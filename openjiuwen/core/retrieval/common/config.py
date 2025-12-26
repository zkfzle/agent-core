# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
配置类

所有配置类统一放在此文件中。
"""
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal, Dict, Any


class KnowledgeBaseConfig(BaseModel):
    """知识库配置"""
    kb_id: str = Field(..., description="知识库标识符")
    index_type: Literal["hybrid", "bm25", "vector"] = Field(
        default="hybrid", description="索引类型"
    )
    use_graph: bool = Field(default=False, description="是否使用图索引")
    chunk_size: int = Field(default=512, description="分块大小")
    chunk_overlap: int = Field(default=50, description="分块重叠")


class RetrievalConfig(BaseModel):
    """检索配置"""
    top_k: int = Field(default=5, description="返回数量")
    score_threshold: Optional[float] = Field(
        default=None, description="分数阈值"
    )
    use_graph: Optional[bool] = Field(
        default=None, description="是否使用图检索（None 使用默认配置）"
    )
    agentic: bool = Field(default=False, description="是否使用 Agentic 检索")
    graph_expansion: bool = Field(
        default=False, description="是否启用图扩展"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="元数据过滤条件"
    )


class IndexConfig(BaseModel):
    """索引配置"""
    index_name: str = Field(..., description="索引名称")
    index_type: Literal["hybrid", "bm25", "vector"] = Field(
        default="hybrid", description="索引类型"
    )


class VectorStoreConfig(BaseModel):
    """向量存储配置"""
    collection_name: str = Field(..., description="集合名称")
    distance_metric: Literal["cosine", "euclidean", "dot"] = Field(
        default="cosine", description="距离度量"
    )


class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""
    model_name: str = Field(..., description="模型名称")
    api_key: Optional[str] = Field(None, description="API Key")
    base_url: Optional[str] = Field(None, description="API Base URL")
