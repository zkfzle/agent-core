# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
文本分块器抽象基类

继承 Processor，提供文本分块接口。
"""
import uuid
from abc import abstractmethod
from typing import List, Optional, Any, Callable

from openjiuwen.core.retrieval.indexing.processor.base import Processor
from openjiuwen.core.retrieval.common.document import Document, TextChunk


class Chunker(Processor):
    """文本分块器抽象基类（继承 Processor）"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function: Optional[Callable[[str], int]] = None,
        **kwargs: Any,
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            length_function: 长度计算函数（默认使用字符数）
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """
        分块文本
        
        Args:
            text: 待分块的文本
            
        Returns:
            分块后的文本列表
        """
        pass
    
    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        分块文档列表
        
        Args:
            documents: 文档列表
            
        Returns:
            文档块列表
        """
        chunks = []
        for doc in documents:
            texts = self.chunk_text(doc.text)
            for i, text in enumerate(texts):
                chunk = TextChunk(
                    id_=str(uuid.uuid4()),
                    text=text,
                    doc_id=doc.id_,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(texts),
                    },
                )
                chunks.append(chunk)
        return chunks

    async def process(self, documents: List[Document], **kwargs: Any) -> List[TextChunk]:
        """
        处理文档（实现 Processor 的 process 方法）
        
        Args:
            documents: 文档列表
            **kwargs: 额外参数
            
        Returns:
            文档块列表
        """
        return self.chunk_documents(documents)
