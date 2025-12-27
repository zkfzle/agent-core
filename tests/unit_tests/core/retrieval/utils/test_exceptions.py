# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
异常定义测试用例
"""
import pytest

from openjiuwen.core.retrieval.utils.exceptions import (
    RAGException,
    KnowledgeBaseError,
    KnowledgeBaseIndexError,
    KnowledgeBaseRetrievalError,
    DocumentProcessingError,
    VectorStoreError,
)


class TestRAGException:
    """RAG 异常测试"""

    def test_rag_exception(self):
        """测试基础 RAG 异常"""
        with pytest.raises(RAGException):
            raise RAGException("Test error")

    def test_rag_exception_message(self):
        """测试异常消息"""
        try:
            raise RAGException("Test error message")
        except RAGException as e:
            assert str(e) == "Test error message"


class TestKnowledgeBaseError:
    """知识库错误测试"""

    def test_knowledge_base_error(self):
        """测试知识库错误"""
        with pytest.raises(KnowledgeBaseError):
            raise KnowledgeBaseError("KB error")

    def test_knowledge_base_error_inheritance(self):
        """测试知识库错误继承关系"""
        assert issubclass(KnowledgeBaseError, RAGException)


class TestKnowledgeBaseIndexError:
    """索引错误测试"""

    def test_knowledge_base_index_error(self):
        """测试索引错误"""
        with pytest.raises(KnowledgeBaseIndexError):
            raise KnowledgeBaseIndexError("Index error")

    def test_knowledge_base_index_error_inheritance(self):
        """测试索引错误继承关系"""
        assert issubclass(KnowledgeBaseIndexError, RAGException)


class TestKnowledgeBaseRetrievalError:
    """检索错误测试"""

    def test_knowledge_base_retrieval_error(self):
        """测试检索错误"""
        with pytest.raises(KnowledgeBaseRetrievalError):
            raise KnowledgeBaseRetrievalError("Retrieval error")

    def test_knowledge_base_retrieval_error_inheritance(self):
        """测试检索错误继承关系"""
        assert issubclass(KnowledgeBaseRetrievalError, RAGException)


class TestDocumentProcessingError:
    """文档处理错误测试"""

    def test_document_processing_error(self):
        """测试文档处理错误"""
        with pytest.raises(DocumentProcessingError):
            raise DocumentProcessingError("Processing error")

    def test_document_processing_error_inheritance(self):
        """测试文档处理错误继承关系"""
        assert issubclass(DocumentProcessingError, RAGException)


class TestVectorStoreError:
    """向量存储错误测试"""

    def test_vector_store_error(self):
        """测试向量存储错误"""
        with pytest.raises(VectorStoreError):
            raise VectorStoreError("Vector store error")

    def test_vector_store_error_inheritance(self):
        """测试向量存储错误继承关系"""
        assert issubclass(VectorStoreError, RAGException)

