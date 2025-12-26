# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
文档解析器抽象基类

继承 Processor，提供文档解析接口。
"""
import os
from abc import abstractmethod
from typing import List, AsyncIterator, Any, Optional

from openjiuwen.core.retrieval.indexing.processor.base import Processor
from openjiuwen.core.retrieval.common.document import Document


class Parser(Processor):
    """文档解析器抽象基类（继承 Processor）"""
    
    async def parse(self, doc: str, doc_id: str = "", **kwargs: Any) -> List[Document]:
        """
        解析文档
        
        Args:
            doc: 文档源（文件路径、URL 等）
            doc_id: 文档ID
            **kwargs: 额外参数
            
        Returns:
            文档列表
        """
        content = await self._parse(doc)
        if content:
            return [Document(
                id_=doc_id,
                text=content,
                metadata={}
            )]
        return []
    
    async def _parse(self, file_path: str) -> Optional[str]:
        pass

    async def lazy_parse(self, doc: str, doc_id: str = "", **kwargs: Any) -> AsyncIterator[Document]:
        """默认基于 parse 的懒加载实现。"""
        docs = await self.parse(doc, doc_id=doc_id, **kwargs)
        for d in docs:
            yield d

    async def process(self, *args: Any, **kwargs: Any) -> Any:
        """兼容 Processor 抽象方法，默认调用 parse。"""
        return await self.parse(*args, **kwargs)
    
    def supports(self, doc: str) -> bool:
        """
        检查是否支持该文档源
        
        Args:
            source: 文档源
            
        Returns:
            是否支持
        """
        return False
