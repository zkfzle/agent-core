# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from openjiuwen.core.memory.config.graph.config import GraphConfig
from openjiuwen.core.memory.config.graph.query_expr import BaseFilter
from openjiuwen.core.memory.config.graph.result_ranking.rank_configs import BaseRankConfig

from ..api_services.embedding_service import EmbeddingService


@runtime_checkable
class GraphBackend(Protocol):
    """Protocol defining the interface for graph memory backends.

    This protocol defines the standard interface that all graph memory backends
    must implement, providing methods for data storage, retrieval, and search
    operations on graph-structured memory data.

    Additional instance variables that should be implemented:
    config: GraphConfig
    embed_executor: ThreadPoolExecutor
    embedder: Optional[EmbeddingService]
    """

    config: GraphConfig
    embed_executor: ThreadPoolExecutor
    embedder: Optional[EmbeddingService]

    # Factory method
    @classmethod
    def from_config(cls, config: GraphConfig, **kwargs) -> "GraphBackend":
        """Create a backend instance from configuration.

        Args:
            config: Graph configuration object
            **kwargs: Additional configuration parameters

        Returns:
            Configured backend instance
        """

    # Data addition methods
    def refresh(self, *args, **kwargs):
        """Refresh / flush inserted data to database"""

    # Data addition methods
    def add_data(self, collection: str, data: Iterable[Dict], flush: bool = True, is_upsert: bool = False, **kwargs):
        """Add arbitrary data into database.

        Args:
            collection (str): Collection name for data insertion
            data (Iterable[Dict]): Data to insert, must be Iterable type like list or tuple
            flush: Whether to flush changes immediately
            is_upsert: Whether to upsert (update if exists, insert if not) instead of insert
        """

    def add_entities(self, entities: Iterable, flush: bool = True, is_upsert: bool = False, skip_embed: bool = False):
        """Add entity objects to the graph store. Should also create embeddings for entities unless skip_embed=True.

        Args:
            entities: Iterable of Entity objects to add
            flush: Whether to flush changes immediately
            is_upsert: Whether to upsert (update if exists, insert if not) instead of insert
            skip_embed: Whether to skip embedding
        """

    def add_relations(self, relations: Iterable, flush: bool = True, is_upsert: bool = False, skip_embed: bool = False):
        """Add relation objects to the graph store. Should also create embeddings for relations unless skip_embed=True.

        Args:
            relations: Iterable of Relation objects to add
            flush: Whether to flush changes immediately
            is_upsert: Whether to upsert (update if exists, insert if not) instead of insert
            skip_embed: Whether to skip embedding
        """

    def add_episodes(self, episodes: Iterable, flush: bool = True, is_upsert: bool = False, skip_embed: bool = False):
        """Add episode objects to the graph store. Should also create embeddings for episodes unless skip_embed=True.

        Args:
            episodes: Iterable of Episode objects to add
            flush: Whether to flush changes immediately
            is_upsert: Whether to upsert (update if exists, insert if not) instead of insert
            skip_embed: Whether to skip embedding
        """

    # Data query and deletion methods
    def is_empty(self, collection: str) -> bool:
        """Check if a collection is empty

        Args:
            collection (str): name of collection

        Returns:
            bool: whether the collection is empty.
        """

    def query(
        self,
        collection: str,
        ids: Optional[List[Any]] = None,
        expr: Optional[BaseFilter] = None,
        silence_errors: bool = False,
        **kwargs,
    ) -> List[Dict]:
        """Query graph objects from a collection.

        Args:
            collection: Collection name to query
            ids: Optional list of IDs to query
            expr: Optional filter expression
            silence_errors (bool): Supresses Exceptions and return empty list instead. Defaults to False.
            **kwargs: Additional arguments to pass into query, such as "limit".

        Returns:
            List of query results
        """

    def delete(
        self, collection: str, ids: Optional[List[Any]] = None, expr: Optional[BaseFilter] = None, **kwargs
    ) -> Dict:
        """Delete graph objects from a collection.

        Args:
            collection: Collection name to delete from
            ids: Optional list of IDs to delete
            expr: Optional filter expression
            **kwargs: Additional delete parameters

        Returns:
            Result of the delete operation
        """

    # Search methods
    def search(
        self,
        query: str,
        k: int,
        collection: str,
        ranker_config: BaseRankConfig,
        bfs_depth: int = 0,
        bfs_k: int = 0,
        filter_expr: Optional[BaseFilter] = None,
        output_fields: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
        output_dict: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        """Search for graph objects using hybrid search.

        Args:
            query: Search query string
            k: Number of results to return
            collection: Collection to search ("entities", "relations", "episodes", or "all")
            ranker_config: Configuration for search ranking
            bfs_depth: Breadth-first search depth for graph expansion
            bfs_k: Maximum number of nodes to expand in BFS
            filter_expr: Optional filter expression
            output_fields: Fields to include in results
            query_embedding: Pre-computed embedding of query string, optional
            output_dict: Optional dict to store results (for "all" collection)

            **kwargs supports following arguments:

                - language (str, optional): Language to use for reranking, "cn" for Chinese, "en" for English.\
                    Defaults to "en".
                - reranker (Optional[BaseReranker], optional): Cross-encoder re-ranker to use. Defaults to None.
                - min_score (float, optional): Minimum similarity / maximum distance score threshold. Defaults to 0.0.

        Returns:
            Dict[str, List[Dict]]: dict mapping collection names to results
        """

    # Embedding management
    def attach_embedder(self, embedder: EmbeddingService) -> None:
        """Attach an embedding service to the backend.

        Args:
            embedder: Embedding service instance

        Raises:
            ValueError: If embedder is invalid or already attached
        """

    # Utility methods
    def close(self) -> None:
        """Close the backend and clean up resources."""
