# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
异常定义

包含所有 RAG 相关的异常类。
"""


class RAGException(Exception):
    """RAG 模块基础异常"""
    pass


class KnowledgeBaseError(RAGException):
    """知识库错误"""
    pass


class IndexError(RAGException):
    """索引错误"""
    pass


class RetrievalError(RAGException):
    """检索错误"""
    pass


class DocumentProcessingError(RAGException):
    """文档处理错误"""
    pass


class VectorStoreError(RAGException):
    """向量存储错误"""
    pass
