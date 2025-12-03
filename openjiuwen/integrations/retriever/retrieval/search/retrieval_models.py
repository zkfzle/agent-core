# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import abc
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openjiuwen.core.common.logging import logger


class TextChunk(BaseModel):
    """
    Represents a semantic chunk of a document with relevance scoring.

    Contains a portion of a document's content about its relevance to a specific query.
    """

    content: str = Field(..., description="Text content of the document chunk")
    similarity_score: float = Field(..., description="Similarity score to query")
    position: Optional[int] = Field(None, description="Position index within the original document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the chunk")

    def __init__(self, **data: Any):
        super().__init__(**data)

    def __str__(self) -> str:
        position_str = f"{self.position}" if self.position is not None else "None"
        return f"TextChunk(position={position_str}, score={self.similarity_score:.4f})"


class Document(BaseModel):
    """
    Represents a complete document in the knowledge base.

    Contains document identifiers, document title, source uri and semantic chunks
    that can be individually retrieved based on relevance.
    """

    document_id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Document title")
    url: Optional[str] = Field(None, description="URL to original source")
    chunks: List[TextChunk] = Field(default_factory=list, description="Semantic chunks of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the document")

    def __init__(self, **data: Any):
        super().__init__(**data)

    @property
    def full_content(self) -> str:
        """Reconstruct full document content from chunks"""
        if hasattr(self.chunks, "position"):
            return "\n\n".join(chunk.content for chunk in sorted(self.chunks, key=lambda x: x.position))
        else:
            return "\n\n".join(chunk.content for chunk in self.chunks)

    def to_dict(self) -> dict:
        """Convert document to serializable dictionary"""
        result = {
            "document_id": self.document_id,
            "content": self.full_content,
            "chunk_count": len(self.chunks),
        }
        if self.title:
            result["title"] = self.title
        if self.url:
            result["url"] = self.url
        return result

    def get_top_chunks(self, k: int = 5) -> List[TextChunk]:
        """Return the top k chunks by similarity score"""
        return sorted(self.chunks, key=lambda x: x.similarity_score, reverse=True)[:k]

    def __str__(self) -> str:
        return f"Document(id={self.document_id!r}, title={self.title!r}, chunks={len(self.chunks)})"


class Dataset(BaseModel):
    """
    Represents a retrievable dataset in the knowledge base.
    """

    description: Optional[str] = Field(None, description="Description of the dataset")
    title: str = Field(..., description="Title of the dataset")
    uri: str = Field(..., description="URI or connection string for accessing the dataset")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the dataset")

    def __init__(self, **data: Any):
        super().__init__(**data)

    def __str__(self) -> str:
        return f"Dataset(id={self.uri!r}, title={self.title!r})"


class RetrievalResult(BaseModel):
    """
    Represents the result of a retrieval operation.
    """

    query: str = Field(..., description="Original query string")
    datasets: List[Dataset] = Field(default_factory=list, description="Datasets used for retrieval")
    documents: List[Document] = Field(default_factory=list, description="Retrieved documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the retrieval")


class BaseRetriever(abc.ABC):
    """
    Abstract base class for Retrieval-Augmented Generation (PAG) providers.

    Defines the interface for interacting with various knowledge sources to retrieve
    relevant documents for a given query.
    """

    @abc.abstractmethod
    def list_datasets(
        self,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> List[Dataset]:
        """
        List available datasets from the RAG Retriever.

        Args:
            name: Optional search query to filter datasets by name/description.
            dataset_id: Optional search id to filter datasets by dataset_id.

        Returns:
            List of matching datasets.
        """
        pass

    @abc.abstractmethod
    def list_documents(
        self,
        dataset_id: str,
        document_id: Optional[str] = None,
    ) -> List[Document]:
        """
        List available documents from the RAG Retriever.

        Args:
            dataset_id: Search id to filter documents by dataset_id.
            document_id: Optional search id to filter documents by document_id.

        Returns:
            List of matching documents.
        """
        pass

    @abc.abstractmethod
    def search_relevant_documents(
        self,
        question: str,
        datasets: Optional[List[Dataset]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Query relevant documents from specified datasets.

        Args:
            question: Search query string.
            datasets: List of datasets to query (empty for all available datasets).
            top_k: Optional maximum number of documents to return.
            similarity_threshold: Optional minimum similarity threshold for documents to return.
        """
        pass
