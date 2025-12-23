# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from pymilvus import AnnSearchRequest, MilvusClient, MilvusException

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.config.graph import query_expr
from openjiuwen.core.memory.config.graph.config import GraphConfig
from openjiuwen.core.memory.config.graph.result_ranking.rank_configs import BaseRankConfig, WeightedRankConfig
from openjiuwen.core.memory.store.graph_store.graph_objects import BaseGraphObject, Entity, Episode, Relation

from ..api_services.embedding_service import EmbeddingService, VarDimEmbeddingService
from ..api_services.llm_reranker import BaseReranker
from .generate_milvus_schema import generate_schema_and_index
from .utils import batched


class MilvusBackend:
    """
    Reference implementation of GraphBackend, for Milvus Database (lite/standalone/distributed vesions)
    Milvus is an open-source vector database with Apache 2.0 license: https://milvus.io
    """

    # Conform to GraphBackend Protocol, need class to have attributes config, embed_executor, embedder
    config: GraphConfig = None
    embed_executor: ThreadPoolExecutor = None
    embedder: Optional[EmbeddingService] = None

    def __init__(self, config: GraphConfig):
        self.config = config
        self.client = MilvusClient(
            uri=config.uri,
            user=config.user,
            password=config.password,
            token=config.token,
            timeout=config.timeout,
            **config.extras,
        )
        if config.name not in self.client.list_databases(timeout=config.timeout):
            self.client.create_database(config.name, timeout=config.timeout)
        self.client.use_database(config.name, timeout=config.timeout)
        self._build_indices()
        self.full_text_search_params = MappingProxyType({"metric_type": "BM25", "drop_ratio_search": "0"})
        self.dense_search_params = MappingProxyType({"metric_type": self.config.db_embed_config.metric_type})
        if config.embedding_config:
            self.embedder = config.embedding_cls(config.embedding_config)
        else:
            self.embedder = None
        embed_worker_threads = self.config.worker_threads or None
        self.embed_executor = ThreadPoolExecutor(
            max_workers=embed_worker_threads, thread_name_prefix="graph_memory_embed"
        )
        self.field_def = {
            "entities": [key for key in Entity.model_fields if not key.endswith("_bm25")],
            "relations": [key for key in Relation.model_fields if not key.endswith("_bm25")],
            "episodes": [key for key in Episode.model_fields if not key.endswith("_bm25")],
        }

    @staticmethod
    def rerank(query: str, candidates: List[Dict], reranker: BaseReranker, language: str, **kwargs):
        """Perform cross-encoder re-ranking on retrieval results

        Args:
            query (str): Query for retrieval.
            candidates (List[Dict]): Retrieved candidates.
            reranker (Qwen3Reranker): Re-ranker to use, instance of Qwen3Reranker or its subclass.
            language (str): Language for prompt to use in re-ranking.
        """
        llm_scores = reranker.rerank(
            query, [doc.get("entity", doc)["content"] for doc in candidates], language=language, **kwargs
        )
        for doc in candidates:
            doc["distance"] = llm_scores[doc.get("entity", doc)["content"]]
        candidates.sort(key=lambda doc: doc["distance"], reverse=True)

    @classmethod
    def from_config(cls, config: GraphConfig, **kwargs) -> "MilvusBackend":
        """Create a backend instance from configuration.

        Args:
            config: Graph configuration object
            **kwargs: Additional configuration parameters (ignored)

        Returns:
            Configured backend instance
        """
        obj = cls(config=config)
        return obj

    def attach_embedder(self, embedder: EmbeddingService):
        """Attach an embedding service to the database backend.

        Args:
            embedder (EmbeddingService): Embedding service to use.

        Raises:
            ValueError: If there is already an embedder, or provided embedder does not inherit EmbeddingService.
        """
        if isinstance(embedder, EmbeddingService) and self.embedder is None:
            self.embedder = embedder
        elif self.embedder is not None:
            raise ValueError(f"Attempt to re-define {type(self).__name__}.embedder.")
        else:
            raise ValueError(
                f"Embedder must be instance of EmbeddingService or a subclass of it, got {type(embedder)} instead."
            )

    def refresh(self):
        """Flush inserted data to database"""
        for col in self.field_def:
            self.client.flush(col)

    def search(
        self,
        query: str,
        k: int,
        collection: str,
        ranker_config: BaseRankConfig,
        bfs_depth: int = 0,
        bfs_k: int = 0,
        filter_expr: Optional[query_expr.BaseFilter] = None,
        output_fields: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
        output_dict: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        """Perform hybrid search on database, uses semantic embeddings and BM25.

        Args:
            query (str): Input query text.
            k (int): Top-K for results to return.
            collection (str): Collection to search on: "entities", "relations", "episodes", "all". \
                If reranker is supplied, "all" option would use retrieved relations to enhance entity re-ranking.
            ranker_config (BaseRankConfig): Search configuration, support WeightedRankConfig and RRFRankConfig.
            bfs_depth (int, optional): Maximum depth for BFS graph expansion. Defaults to 0.
            bfs_k (int, optional): Top-K for BFS graph expansion. Defaults to 0.
            filter_expr (Optional[query_expr.BaseFilter], optional): Additional filtering expression. Defaults to None.
            output_fields (Optional[List[str]], optional): Fields to return, if None, return all that are applicable.\
                Defaults to None.
            query_embedding (Optional[List[float]], optional): Pre-computed embedding for query text. Defaults to None.
            output_dict (Optional[Dict], optional): Dictionary to store retrieval results in. Defaults to None.
            **kwargs supports following arguments:

                - language (str, optional): Language to use for reranking, "cn" for Chinese, "en" for English.\
                    Defaults to "en".
                - reranker (Optional[BaseReranker], optional): Cross-encoder re-ranker to use. Defaults to None.
                - min_score (float, optional): Minimum similarity / maximum distance score threshold. Defaults to 0.0.

        Returns:
            Dict[str, List[Dict]]: dict mapping collection names to results.
        """
        language: str = kwargs.pop("language", "en")
        reranker: Optional[BaseReranker] = kwargs.pop("reranker", None)
        min_score: float = kwargs.pop("min_score", 0.0)

        if output_dict is None:
            output_dict = dict()
        for col in ["entities", "relations", "episodes"]:
            output_dict[col] = []
        if collection.strip().casefold() == "all":
            tasks = dict()
            for col in ["entities", "relations", "episodes"]:
                future = self.embed_executor.submit(
                    self.search,
                    query=query,
                    k=k,
                    collection=col,
                    ranker_config=ranker_config,
                    bfs_depth=bfs_depth,
                    bfs_k=bfs_k,
                    filter_expr=filter_expr,
                    output_fields=output_fields,
                    min_score=min_score,
                    language=language,
                    reranker=None,
                    query_embedding=query_embedding,
                )
                tasks[future] = col
            for future in as_completed(tasks):
                output_dict[tasks[future]] = future.result()

            # improve entity re-ranking with relations
            self._combined_rerank(query, output_dict, reranker, language, min_score)

            return output_dict

        output_fields = output_fields or self.field_def[collection]
        expr = filter_expr.model_copy() if isinstance(filter_expr, query_expr.BaseFilter) else None
        if bfs_depth > 0 and collection in ("entities", "relations"):
            uuids: Set[str] = set()
            all_results = dict()
            query_embedding = query_embedding or self._query_embedding(query)
            if collection == "entities":
                expansion_fn = self._expand_entities
            else:
                expansion_fn = self._expand_relations

            for graph_expansion in ([True] * bfs_depth) + [False]:
                res = self._raw_hybrid_search(
                    query,
                    k,
                    collection,
                    ranker_config,
                    skip_ranking=True,
                    query_embedding=query_embedding,
                    expr=expr,
                    output_fields=output_fields,
                    language=language,
                    reranker=None,
                )
                new_results = {doc["uuid"]: doc for doc in res}
                new_uuids = set(new_results.keys()).difference(uuids)
                all_results.update(new_results)

                # graph expansion
                if graph_expansion and new_uuids:
                    new_uuids = expansion_fn(filter_expr, new_uuids, lookup=new_results).difference(uuids)
                    if not new_uuids:
                        break
                    uuids.update(new_uuids)
                    if bfs_k > len(new_uuids):
                        is_similarity = (
                            (self.config.db_embed_config.metric_type in {"IP", "COSINE"}) if self.config else True
                        )
                        new_uuids = set(
                            sorted(new_uuids, key=lambda uuid: new_results[uuid]["distance"], reverse=is_similarity)[
                                :bfs_k
                            ]
                        )
                    new_uuids_list = list(new_uuids)
                    if collection == "entities":
                        expr = query_expr.in_list("uuid", new_uuids_list)
                    else:
                        expr = query_expr.in_list("lhs", new_uuids_list) | query_expr.in_list("rhs", new_uuids_list)
                    if filter_expr:
                        expr = filter_expr & expr

            results = self._rank_results(
                query, candidates=list(all_results.values()), reranker=reranker, language=language, min_score=min_score
            )

            if output_dict is not None:
                output_dict[collection] = results

            return output_dict

        results = self._raw_hybrid_search(
            query,
            k,
            collection,
            ranker_config,
            skip_ranking=False,
            expr=expr,
            output_fields=output_fields,
            language=language,
            reranker=reranker,
        )

        if output_dict is not None:
            output_dict[collection] = results

        return output_dict

    def add_data(self, collection: str, data: Iterable[Dict], flush: bool = True, is_upsert: bool = False, **kwargs):
        """Add arbitrary data into database"""
        insert_func = self.client.upsert if is_upsert else self.client.insert
        insert_func(collection_name=collection, data=list(data), timeout=self.config.timeout, **kwargs)
        if flush:
            self.client.flush(collection_name=collection, timeout=self.config.timeout)

    def add_entities(
        self, entities: Iterable[Entity], flush: bool = True, is_upsert: bool = False, skip_embed: bool = False
    ):
        """Add entities into database"""
        return self._add_data("entities", entities, flush, is_upsert, skip_embed)

    def add_relations(
        self, relations: Iterable[Relation], flush: bool = True, is_upsert: bool = False, skip_embed: bool = False
    ):
        """Add relations into database"""
        return self._add_data("relations", relations, flush, is_upsert, skip_embed)

    def add_episodes(
        self, episodes: Iterable[Episode], flush: bool = True, is_upsert: bool = False, skip_embed: bool = False
    ):
        """Add episodes into database"""
        return self._add_data("episodes", episodes, flush, is_upsert, skip_embed)

    def is_empty(self, collection: str) -> bool:
        """Check if a collection is empty"""
        return not self.client.get_collection_stats(collection).get("row_count")

    def query(
        self,
        collection: str,
        ids: Optional[List[Any]] = None,
        expr: Optional[query_expr.BaseFilter] = None,
        silence_errors: bool = False,
        **kwargs,
    ) -> List[Dict]:
        """Execute query on database

        Args:
            collection (str): Collection to query on: "entities", "relations", "episodes".
            ids (Optional[List[Any]], optional): List of uuids to fetch directly. Defaults to None.
            expr (Optional[query_expr.BaseFilter], optional): Filtering expression, ignored if ids is not None.\
                Defaults to None.
            silence_errors (bool): Supresses MilvusExceptions and return empty list instead. Defaults to False.
            **kwargs: Additional arguments to pass into query, such as "limit".

        Raises:
            ValueError: If "expr" and "ids" are both None, an integer "limit" argument must be supplied.

        Returns:
            List[Dict]: Query result.
        """
        if expr:
            expr_str = expr.to_str("milvus")
        elif "limit" not in kwargs and ids is None:
            raise ValueError('Argument "limit" must be set to positive integer when "expr" and "ids" are None')
        else:
            expr_str = None
        output_fields = kwargs.pop("output_fields", self.field_def[collection])
        silence_errors = kwargs.pop("throw_errors", False)
        if ids:
            query_method = self.client.get
            query_args = dict(
                collection_name=collection, ids=ids, output_fields=output_fields, timeout=self.config.timeout, **kwargs
            )
        else:
            query_method = self.client.query
            query_args = dict(
                collection_name=collection,
                filter=expr_str,
                output_fields=output_fields,
                timeout=self.config.timeout,
                **kwargs,
            )
        if silence_errors:
            try:
                return query_method(**query_args)
            except MilvusException:
                return []
        return query_method(**query_args)

    def delete(
        self, collection: str, ids: Optional[List[Any]] = None, expr: Optional[query_expr.BaseFilter] = None, **kwargs
    ) -> Dict:
        """Delete records from database

        Args:
            collection (str): Collection to perform deletion on: "entities", "relations", "episodes".
            ids (Optional[List[Any]], optional): List of uuids for records to delete. Defaults to None.
            expr (Optional[query_expr.BaseFilter], optional): Filtering expression, ignored if ids is not None.\
                Defaults to None.
            **kwargs: Additional arguments to pass into query.

        Raises:
            ValueError: If "expr" and "ids" are both None.

        Returns:
            Dict: Deletion result.
        """
        if ids:
            expr_str = query_expr.in_list("uuid", ids).to_str("milvus")
        elif expr:
            expr_str = expr.to_str("milvus")
        else:
            raise ValueError('Either "ids" or "expr" must supplied')
        return self.client.delete(collection, timeout=self.config.timeout, filter=expr_str, **kwargs)

    def close(self):
        """Close connection to the database"""
        try:
            self.client.close()
            self.embed_executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            """Ignore exception"""

    def _build_indices(self):
        """Build indices & collections for database"""
        self.embed_dim = self.config.embed_dim

        for col in ["entities", "relations", "episodes"]:
            if self.client.has_collection(col):
                if self.config.wipe_at_startup:
                    self.client.drop_collection(col)
                else:
                    self.client.load_collection(col)
                    continue

            schema, index_params = generate_schema_and_index(
                self.client,
                collection=col,
                storage_config=self.config.db_storage_config,
                embed_config=self.config.db_embed_config,
            )

            self.client.create_collection(
                collection_name=col,
                primary_field_name="uuid",
                id_type="varchar",
                schema=schema,
                index_params=index_params,
                vector_field_name="content_embedding",
                dimension=self.embed_dim,
                metric_type=self.config.db_embed_config.metric_type,
                auto_id=False,
            )

            self.client.load_collection(col)

    def _rank_results(
        self,
        query: str,
        candidates: List[Dict],
        reranker: Optional[BaseReranker],
        language: str,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """Internal helper function for ranking retrieval results (and re-ranking if reranker is supplied)"""
        if self.config.db_embed_config.metric_is_sim:
            candidates = [doc for doc in candidates if doc["distance"] >= min_score]
        else:
            candidates = [doc for doc in candidates if doc["distance"] <= min_score]
        if reranker:
            # Cross-encoder re-ranking
            self.rerank(query, candidates=candidates, reranker=reranker, language=language)
        else:
            # If is similarity not distance, higher is better, otherwise lower is better
            is_similarity = (self.config.db_embed_config.metric_type in {"IP", "COSINE"}) if self.config else True
            candidates.sort(key=lambda doc: doc["distance"], reverse=is_similarity)
        return candidates

    def _combined_rerank(
        self, query: str, results: Dict, reranker: Optional[BaseReranker], language: str, min_score: float = 0.0
    ):
        """Internal helper function for combined re-ranking (using relations to aid entity re-ranking)"""
        if reranker is None:
            return

        entities = [ent.copy() for ent in results["entities"]]
        relations = [rel.copy() for rel in results["relations"]]
        rel_uuids = {rel["uuid"]: rel for rel in relations}
        for ent in entities:
            ent["original_content"] = ent.get("content", "")
            content = []
            for rel_id in ent.get("relations", []):
                if rel_id in rel_uuids:
                    rel = rel_uuids[rel_id]
                    content.append((rel["content"], rel["distance"]))
            content.sort(key=lambda rel: rel[1], reverse=True)
            content = [(ent["original_content"], -1), ("-" * 10, -1)] + content
            mentions = len(content) - 2
            if mentions > 0:
                ent["content"] = "\n - ".join(line for line, _ in content)

        entities = self._rank_results(
            query, candidates=entities, reranker=reranker, language=language, min_score=min_score
        )

        for ent in entities:
            ent["content"] = ent["original_content"]
            del ent["original_content"]

        results["entities"] = entities

    def _expand_entities(self, expr: Optional[query_expr.BaseFilter], uuids: Set[str], **kwargs) -> Set:
        """Graph expansion method for entity retrieval (expand by relations)"""
        if uuids:
            final_expr = query_expr.in_list("lhs", uuids) | query_expr.in_list("rhs", uuids)
            if expr:
                final_expr = expr & final_expr
            results = self.client.query("relations", filter=final_expr.to_str("milvus"), output_fields=["lhs", "rhs"])
            expansion_results = {doc["lhs"] for doc in results}
            expansion_results.update(doc["rhs"] for doc in results)
            return expansion_results
        return set()

    def _expand_relations(
        self, expr: Optional[query_expr.BaseFilter], uuids: Set[str], lookup: Dict[str, Dict], **kwargs
    ) -> Set:
        """Graph expansion method for relation retrieval (expand by entities)"""
        if uuids:
            node_uuids = []
            for relation_uuid in uuids:
                relation = lookup[relation_uuid]
                node_uuids.append(relation["lhs"])
                node_uuids.append(relation["rhs"])

            final_expr = query_expr.in_list("uuid", set(node_uuids))
            if expr:
                final_expr = expr & final_expr
            results = self.client.query(
                "entities",
                filter=final_expr.to_str("milvus"),
                output_fields=["relations"],
            )

            expansion_results = []
            for entity in results:
                expansion_results.extend(entity["relations"])
            return set(expansion_results)
        return set()

    def _query_embedding(self, query: str) -> List[float]:
        if issubclass(self.config.embedding_cls, VarDimEmbeddingService):
            extra_kwargs = dict(max_retry=self.config.request_max_retry, retry_wait=self.config.request_retry_wait)
        else:
            extra_kwargs = {}
        return self.embedder.query_embedding(query, **extra_kwargs)

    def _add_data(
        self,
        collection: str,
        data: Iterable[BaseGraphObject],
        flush: bool = True,
        is_upsert: bool = False,
        skip_embed: bool = False,
    ):
        """Internal helper function for adding data to database"""
        t_start = time.time()
        data = list(data)
        if data:
            # Fetch all embedding tasks in the form of (obj, attribute, value) tuples
            to_embed = []
            if not skip_embed:
                for graph_object in data:
                    to_embed.extend(graph_object.fetch_embed_task())

            tasks: List[Future] = []
            support_batch = issubclass(self.config.embedding_cls, VarDimEmbeddingService)
            if support_batch:
                extra_kwargs = dict(max_retry=self.config.request_max_retry, retry_wait=self.config.request_retry_wait)
            else:
                extra_kwargs = {}
            embed_batches = list(batched(to_embed, self.config.embed_batch_size))
            for batch in embed_batches:
                query = [task_tuple[-1] for task_tuple in batch]  # task_tuple[-1] is query to embed
                if not support_batch:
                    query = query[0]
                tasks.append(self.embed_executor.submit(self.embedder.query_embedding, query, **extra_kwargs))

            for batch, future in zip(embed_batches, tasks):
                embed_result = future.result()
                if not embed_result:
                    raise ConnectionError("Embedding Service is not available")
                if not support_batch:
                    embed_result = [embed_result]
                for (obj, attribute, _), embedding in zip(batch, embed_result):
                    setattr(obj, attribute, embedding)

            data_processed: List[Dict] = []
            for item in data:
                if len(item.content) > self.config.db_storage_config.content:
                    item.content = item.content[: self.config.db_storage_config.content - 3] + "..."
                if hasattr(item, "name"):
                    item_name = getattr(item, "name")
                    if len(item_name) > self.config.db_storage_config.name:
                        setattr(item, "name", item_name[: self.config.db_storage_config.name - 3] + "...")
                entity_dict = {k: v for k, v in item.model_dump().items() if v is not None}
                data_processed.append(entity_dict)
            insert_func = self.client.upsert if is_upsert else self.client.insert
            try:
                insert_func(collection, data=data_processed, timeout=self.config.timeout)
            except MilvusException:
                # Maybe too much data
                logger.info("Milvus data addition failed, try batching with size of %d", self.config.embed_batch_size)
                try:
                    self.client.delete(collection, ids=[x.uuid for x in data])
                except Exception:
                    """Ignore exception"""
                for batch in batched(data_processed, self.config.embed_batch_size):
                    insert_func(collection, data=list(batch), timeout=self.config.timeout)
        if flush:
            self.client.flush(collection, timeout=self.config.timeout)
        logger.debug("Add graph memory [%s] took %.2fs", collection, round(time.time() - t_start, 2))

    def _raw_hybrid_search(
        self,
        query: str,
        k: int,
        collection: str,
        ranker_config: BaseRankConfig,
        skip_ranking: bool = False,
        query_embedding: Optional[List[float]] = None,
        expr: Union[None, str, query_expr.BaseFilter] = None,
        output_fields: Iterable[str] = ("id", "uuid", "content"),
        **kwargs,
    ) -> List[Dict]:
        """Internal method for performing hybrid search on a collection"""
        language: str = kwargs.pop("language", "en")
        reranker: Optional[BaseReranker] = kwargs.pop("reranker", None)
        min_score: float = kwargs.pop("min_score", 0.0)
        query_embedding = query_embedding or self._query_embedding(query)

        if isinstance(expr, query_expr.BaseFilter):
            expr = expr.to_str("milvus")
        if not isinstance(expr, str):
            expr = ""

        # Prepare sparse and dense vector search requests
        dense_req_name = AnnSearchRequest(
            [query_embedding],
            "name_embedding",
            self.dense_search_params,
            limit=k * 2,
            expr=expr,
        )
        dense_req_content = AnnSearchRequest(
            [query_embedding],
            "content_embedding",
            self.dense_search_params,
            limit=k * 2,
            expr=expr,
        )
        full_text_search_params = self.full_text_search_params.copy()
        sparse_req_content = AnnSearchRequest(
            [query],
            "content_bm25",
            full_text_search_params,
            limit=k * 2,
            expr=expr,
        )
        search_requests = [dense_req_name, dense_req_content, sparse_req_content]

        ranker_config = ranker_config.model_copy()
        if isinstance(ranker_config, WeightedRankConfig):
            if collection == "episodes":
                ranker_config.dense_name = 0
            elif collection == "relations":
                ranker_config.dense_name = 0
            weights = [
                ranker_config.dense_name,
                ranker_config.dense_content,
                ranker_config.sparse_content,
            ]
        else:
            weights = [float(w) for w in ranker_config.is_active]
            if collection == "episodes":
                weights[0] = weights[1] = 0
            elif collection == "relations":
                weights[0] = 0

        # Adjust search based on ranking config
        search_requests = [req for req, weight in zip(search_requests, weights) if weight > 0]
        ranker_cls = ranker_config.get_ranker_cls("milvus")
        ranker_args, ranker_kwargs = ranker_config.args
        ranker = ranker_cls(*ranker_args, **ranker_kwargs)

        # Execute hybrid search request
        result = self.client.hybrid_search(
            collection,
            search_requests,
            ranker=ranker,
            limit=k,
            output_fields=list(output_fields),
        )[0]

        if skip_ranking:
            return result

        return self._rank_results(query, candidates=result, reranker=reranker, language=language, min_score=min_score)
