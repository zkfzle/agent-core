# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
索引管理器抽象基类

提供索引管理的统一接口。
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict

from openjiuwen.core.retrieval.common.config import IndexConfig
from openjiuwen.core.retrieval.common.document import TextChunk
from openjiuwen.core.retrieval.embedding.base import Embedding


class Indexer(ABC):
    """索引管理器抽象基类"""
    
    @abstractmethod
    async def build_index(
        self,
        chunks: List[TextChunk],
        config: IndexConfig,
        embed_model: Optional[Embedding] = None,
        **kwargs: Any,
    ) -> bool:
        """构建索引"""
        pass
    
    @abstractmethod
    async def update_index(
        self,
        chunks: List[TextChunk],
        doc_id: str,
        config: IndexConfig,
        embed_model: Optional[Embedding] = None,
        **kwargs: Any,
    ) -> bool:
        """更新索引"""
        pass
    
    @abstractmethod
    async def delete_index(
        self,
        doc_id: str,
        index_name: str,
        **kwargs: Any,
    ) -> bool:
        """删除索引"""
        pass
    
    @abstractmethod
    async def index_exists(
        self,
        index_name: str,
    ) -> bool:
        """检查索引是否存在"""
        pass
    
    @abstractmethod
    async def get_index_info(
        self,
        index_name: str,
    ) -> Dict[str, Any]:
        """获取索引信息"""
        pass
