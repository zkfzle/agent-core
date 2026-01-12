# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import List, Tuple, Any

from openjiuwen.core.common.logging import logger
from openjiuwen.core.retrieval.vector_store.base import VectorStore
from openjiuwen.core.retrieval.embedding.base import Embedding


class SemanticStore:
    """
    Semantic store implementation that uses an embedding model and vector store.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_model: Embedding | None = None):
        """
        Initialize the semantic store with an embedding model and vector store.
        
        Args:
            embedding_model: The embedding model to use for generating embeddings.
            vector_store: The vector store to use for storing and searching embeddings.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def initialize_embedding_model(self, embed_model: Embedding):
        """
        Initialize the embedding model.
        
        Args:
            embed_model: The embedding model to use.
        """
        self.embedding_model = embed_model
    
    async def add_docs(self, docs: List[Tuple[str, str]], table_name: str, scope_id: str | None = None) -> bool:
        """
        Add documents to the vector store after generating their embeddings.
        
        Args:
            docs: A list of (id, text) tuples.
            table_name: The name of the table/collection to add to.
            scope_id: Optional scope ID.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized, please call initialize_embedding_model first.")
            return False
            
        try:
            memory_ids, texts = zip(*docs)
            memory_ids = list(memory_ids)
            texts = list(texts)
            
            # Generate embeddings for the texts
            embeddings = await self.embedding_model.embed_documents(texts=texts)
            
            if len(memory_ids) != len(embeddings):
                raise ValueError(f"memory_ids and embeddings must have same length")
            
            # Prepare data for vector store, content is not stored
            data = []
            for doc_id, embedding in zip(memory_ids, embeddings):
                data.append({
                    "id": doc_id,
                    "embedding": embedding,
                    "scope_id": scope_id
                })
            
            # Add to vector store
            await self.vector_store.add(data, table_name=table_name)
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to semantic store: {e}")
            return False
    
    async def delete_docs(self, ids: List[str], table_name: str) -> bool:
        """
        Delete documents from the vector store by their IDs.
        
        Args:
            ids: A list of document IDs.
            table_name: The name of the table/collection.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            return await self.vector_store.delete(ids=ids, table_name=table_name)
        except Exception as e:
            logger.error(f"Failed to delete documents from semantic store: {e}")
            return False
    
    async def search(self, query: str, table_name: str, scope_id: str | None = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query: The query text.
            table_name: The name of the table/collection to search in.
            scope_id: Optional scope ID to filter by.
            top_k: The number of results to return.
            
        Returns:
            A list of (id, score) tuples.
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized, please call initialize_embedding_model first.")
            return []
            
        try:
            # Generate embedding for the query
            query_embeddings = await self.embedding_model.embed_documents(texts=[query])
            query_embedding = query_embeddings[0]

            # Search in vector store
            results = await self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                scope_id=scope_id,
                table_name=table_name
            )
            
            # Convert to required format
            return [(result.id, result.score) for result in results]
        except Exception as e:
            logger.error(f"Failed to search semantic store: {e}")
            return []
    
    async def delete_table(self, table_name: str) -> bool:
        """
        Delete all documents from a specific table/collection.
        
        Args:
            table_name: The name of the table/collection to delete.
            
        Returns:
            True if successful, False otherwise.

        """
        try:
            return await self.vector_store.delete_table(table_name=table_name)
        except Exception as e:
            logger.error(f"Failed to delete table from semantic store: {e}")
            return False