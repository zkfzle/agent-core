# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
知识库抽象基类

提供知识库的统一接口，作为顶层入口。
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict

from openjiuwen.core.retrieval.common.config import KnowledgeBaseConfig, RetrievalConfig
from openjiuwen.core.retrieval.common.document import Document
from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult
from openjiuwen.core.retrieval.indexing.processor.parser.base import Parser
from openjiuwen.core.retrieval.indexing.processor.chunker.base import Chunker
from openjiuwen.core.retrieval.indexing.processor.extractor.base import Extractor
from openjiuwen.core.retrieval.vector_store.base import VectorStore
from openjiuwen.core.retrieval.embedding.base import Embedding
from openjiuwen.core.retrieval.indexing.indexer.base import Indexer
from openjiuwen.core.common.logging import logger


class KnowledgeBase(ABC):
    """知识库抽象基类"""
    
    def __init__(
        self,
        config: KnowledgeBaseConfig,
        vector_store: Optional[VectorStore] = None,
        embed_model: Optional[Embedding] = None,
        parser: Optional[Parser] = None,
        chunker: Optional[Chunker] = None,
        extractor: Optional[Extractor] = None,
        index_manager: Optional[Indexer] = None,
        llm_client: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.config = config
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.parser = parser
        self.chunker = chunker
        self.extractor = extractor
        self.index_manager = index_manager
        self.llm_client = llm_client
    
    @abstractmethod
    async def parse_files(
        self,
        file_paths: List[str],
        **kwargs: Any,
    ) -> List[Document]:
        """
        从文件路径解析为Document对象列表
        
        Args:
            file_paths: 文件路径列表
            **kwargs: 额外参数
            
        Returns:
            Document对象列表
        """
        pass
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """添加文档到知识库"""
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """检索相关文档"""
        pass
    
    @abstractmethod
    async def delete_documents(
        self,
        doc_ids: List[str],
        **kwargs: Any,
    ) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    async def update_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """更新文档"""
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        pass
    
    async def close(self) -> None:
        """关闭知识库，释放资源"""
        import inspect

        async def _maybe_await(obj):
            if not obj:
                return
            if inspect.iscoroutinefunction(getattr(obj, "close", None)):
                await obj.close()
            elif hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    logger.warning("Failed to close object", exc_info=True)

        await _maybe_await(self.vector_store)
        await _maybe_await(self.index_manager)
