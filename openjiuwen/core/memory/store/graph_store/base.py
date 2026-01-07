# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import datetime
import gc
import threading
import time
from concurrent.futures import Future, as_completed
from concurrent.futures import wait as wait_futures
from itertools import chain, cycle
from math import ceil
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import httpx

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.config.graph import (
    DEFAULT_STRATEGY,
    AddMemStrategy,
    EpisodeType,
    GraphConfig,
    LLMConfig,
    SearchConfig,
    query_expr,
)
from openjiuwen.core.memory.config.graph.result_ranking.rank_configs import WeightedRankConfig
from openjiuwen.core.memory.generation.graph.custom_types import ChatMessage
from openjiuwen.core.memory.generation.graph.entity_type_definition import AIEntity, EntityDef, HumanEntity
from openjiuwen.core.memory.generation.graph.extraction_models import (
    EntityDeclaration,
    EntityDuplication,
    EntitySummary,
    MergeRelations,
    RelationExtraction,
    RelevantFacts,
    TimezonePredictions,
)
from openjiuwen.core.memory.generation.graph.extraction_prompts import (
    dedupe_entity_list,
    dedupe_relation_list,
    extract_entity_attributes,
    extract_entity_declaration,
    extract_relation_declaration,
    extract_timezone,
    filter_relations_for_merge,
    merge_existing_entities,
)
from openjiuwen.core.memory.generation.graph.parse_response import ensure_list, parse_json
from openjiuwen.core.memory.generation.graph.prompts.entity_extraction.base import ensure_valid_language
from openjiuwen.core.memory.store.graph_store.api_services.llm_reranker import (
    AliyunRerankService,
    BaseReranker,
    GraphLLMClient,
    StandardRerankService,
)
from openjiuwen.core.memory.store.graph_store.backends.utils import batched
from openjiuwen.core.memory.store.graph_store.parse_llm_response import parse_all_relations, resolve_entities
from openjiuwen.core.utils.llm.messages import BaseMessage
from openjiuwen.core.utils.prompt.template.template import Template

from .api_services.embedding_service import EmbeddingService, VarDimEmbeddingService
from .backends.base import GraphBackendFactory
from .graph_objects import BaseGraphObject, Entity, Episode, Relation, update_entity
from .postprocess_graph_objects import (
    create_episode,
    parse_relation_uuids_to_remove,
    process_entities,
    process_relations,
    validate_entities_episodes,
)
from .states import EntityMerge, GraphMemState, GraphMemUpdate, classify_relations_extracted, persist_to_db
from .utils import format_list_of_messages, format_timestamp, msg2dict, safe_timestamp
from .validate_input import validate_add_memory_input


def embed_loop(embedding_service: EmbeddingService, queries: List[str]) -> List[List[float]]:
    """Since base EmbeddingService doesn't support bulk embedding"""
    return [embedding_service.query_embedding(query) for query in queries]


class GraphMemory:
    """Graph memory class that handles the addition & retrieval of graph memory"""

    def __init__(
        self,
        db_config: GraphConfig,
        db_kwargs: Optional[Dict] = None,
        llm_config: Optional[LLMConfig] = None,
        reranker: Optional[BaseReranker] = None,
        verify: Union[str, bool] = False,
        extraction_strategy: AddMemStrategy = DEFAULT_STRATEGY,
        llm_extra_kwargs: Optional[Dict] = None,
        language: Literal["cn", "en"] = "cn",
        debug: bool = False,
    ):
        # Set default values
        db_kwargs = db_kwargs or dict()
        if llm_config is None:
            llm_config = LLMConfig.default_config()
        if reranker is None:
            reranker_config = LLMConfig.default_config(is_reranker=True)
            if reranker_config is not None:
                rerank_cls_name = reranker_config.extras.pop("JIUWEN_GRAPH_MEM_RERANK_MODE", "standard")
                rerank_cls = AliyunRerankService if rerank_cls_name == "aliyun" else StandardRerankService
                reranker = rerank_cls.from_config(reranker_config)

        self.token_record: Dict[str, int] = dict(input_tokens=0, output_tokens=0)
        self.default_extraction_strategy = extraction_strategy
        self.reranker = reranker
        self.language = ensure_valid_language(language, db_config.db_storage_config.language)
        self.db_backend = GraphBackendFactory.from_config(config=db_config, **db_kwargs)
        self.executor = self.db_backend.embed_executor
        self.llm_client = GraphLLMClient.from_config(llm_config)
        self.llm_extra_kwargs = llm_extra_kwargs
        self.thread_lock = threading.Lock()
        self.user_locks: Dict[str, threading.Lock] = dict()
        self.debug = debug
        self.time_till_next_gc = 10
        self._search_strategies: Dict[str, Tuple[SearchConfig, SearchConfig, SearchConfig]] = dict(
            default=(
                SearchConfig(rank_config=WeightedRankConfig()),
                SearchConfig(min_score=0.1),
                SearchConfig(min_score=0.1),
            )
        )
        self._last_gc: float = 0.0
        self._clients = cycle(
            [
                httpx.Client(verify=verify, timeout=llm_config.timeout)
                for _ in range(self.db_backend.embed_executor._max_workers)
            ]
        )
        self._request_common_kwargs = dict(
            max_retry=self.db_backend.config.request_max_retry, retry_wait=self.db_backend.config.request_retry_wait
        )

    @staticmethod
    def _replace_one_side_of_relation(
        side: str,
        relation: Relation,
        tgt_uuid: str,
        entity_relation_updates: Dict[str, Dict[str, Relation]],
        state: GraphMemState,
    ):
        if relation.uuid not in entity_relation_updates[tgt_uuid]:
            state.relation_deferred_updates[tgt_uuid].append((relation, side, tgt_uuid))
            entity_relation_updates[tgt_uuid][relation.uuid] = relation
        else:
            state.faulty_relations[relation.uuid] = relation
            del entity_relation_updates[tgt_uuid][relation.uuid]
            for task in [task for task in state.relation_deferred_updates[tgt_uuid] if task[0] == relation]:
                state.relation_deferred_updates[tgt_uuid].remove(task)

    @staticmethod
    def _parse_relation_filtering_result(relations: List[Relation], state: GraphMemState):
        """Parse relation filtering result"""
        for task in as_completed(state.relation_filter_tasks):
            tgt_entity, new_relation_list = state.relation_filter_tasks[task]
            tgt_uuid = tgt_entity.uuid
            try:
                response = task.result()
                dedupe_entity = parse_json(response.content, output_schema=state.prompting.schema_relation_filter) or {}
                keep_ids = set(dedupe_entity.get("relevant_relations"))
                relations_filtered = [new_relation_list[i - 1] for i in keep_ids]
            except Exception:
                relations_filtered = new_relation_list
            state.merge_infos[tgt_uuid].new_relations = relations_filtered

        for tgt_uuid, merge_info in state.merge_infos.items():
            for relation, attr, val in state.relation_deferred_updates[tgt_uuid]:
                if relation in merge_info.new_relations:
                    setattr(relation, attr, val)
                    if relation not in state.mem_update_skip_embed.updated_relation:
                        state.mem_update_skip_embed.updated_relation.append(relation)
                else:
                    state.mem_update.removed_relation.add(relation.uuid)
                    state.to_remove.append(relation)

        classify_relations_extracted(relations, state)

    def attach_embedder(self, embedder: EmbeddingService):
        """Supply embedding service to use"""
        self.db_backend.attach_embedder(embedder)

    def attach_reranker(self, reranker: BaseReranker):
        """Supply cross-encoder re-ranker to use"""
        if isinstance(reranker, BaseReranker):
            self.reranker = reranker
        else:
            raise ValueError(
                f"Reranker must be instance of Qwen3Reranker or a subclass of it, got {type(reranker)} instead."
            )

    def register_search_strategy(
        self,
        name: str,
        search_entity: Optional[SearchConfig] = None,
        search_relation: Optional[SearchConfig] = None,
        search_episode: Optional[SearchConfig] = None,
        force: bool = False,
    ):
        """Register search strategy"""
        input_configs = [search_entity, search_relation, search_episode]
        if not all(arg is None or isinstance(arg, SearchConfig) for arg in input_configs):
            raise KeyError("Search config for entity/relation/episode must be an instance of SearchConfig or None")

        with self.thread_lock:
            if not name:
                raise KeyError("Search config cannot be registered as an empty value.")
            if name in self._search_strategies and not force:
                raise KeyError(f"Search config with name [{name}] already exists.")

            self._search_strategies[name] = (
                search_entity or SearchConfig(rank_config=WeightedRankConfig()),
                search_relation or SearchConfig(min_score=0.1),
                search_episode or SearchConfig(min_score=0.1),
            )

    def batch_embed(self, to_embed: List[str], task_queue: List[Future]):
        """Batch embed texts"""
        embed_batches = list(batched(to_embed, self.db_backend.config.embed_batch_size))
        for batch, client in zip(embed_batches, self._clients):
            task_queue.append(
                self.executor.submit(
                    self.db_backend.embedder.query_embedding, query=batch, client=client, **self._request_common_kwargs
                )
            )

    def ensure_thread_lock(self, user_id: str):
        """Ensure thread lock exists for current user_id"""
        with self.thread_lock:
            if user_id not in self.user_locks:
                self.user_locks[user_id] = threading.Lock()

    def add_memory(
        self,
        src_type: EpisodeType,
        user_id: str,
        content: Union[str, List[ChatMessage]],
        content_fmt_kwargs: Optional[Dict] = None,
        reference_time: Optional[datetime.datetime] = None,
    ) -> Tuple[GraphMemUpdate, GraphMemUpdate]:
        """Add memory episode to Memory Knowledge Graph

        Args:
            src_type (EpisodeType): type of episode: conversation/document/json.
            user_id (str): user id.
            content (Union[str, List[ChatMessage]]): content of episode.
            content_fmt_kwargs (Optional[Dict], optional): formatting arguments like {"user": "张三（用户）",\
                "assistant": "智能客服小李"}. Defaults to None.
            reference_time (Optional[datetime.datetime], optional): reference time for when the episode takes place,\
                leave blank will use current time. Defaults to None.

        Returns:
            Tuple[GraphMemUpdate, GraphMemUpdate]: returns the memory update with and without change in content
        """
        self.ensure_thread_lock(user_id=user_id)
        with self.user_locks[user_id]:
            state = self.init_state(reference_time)

            content = self.prepare_episodes(
                src_type,
                user_id,
                content,
                state,
                content_fmt_kwargs,
            )
            current_episode = create_episode(self.db_backend, user_id, content, state)
            content = format_timestamp(state.reference_timestamp) + "\n" + content

            # Timezone Predictions
            tz_task = self.executor.submit(
                self.sync_invoke,
                *extract_timezone(content=content, history=state.history, language=state.prompting.language),
            )

            # Extract Entity Declarations
            no_existing_entity, extracted_declarations = self._extract_entity_declarations(src_type, content, state)

            # Extract Relations
            response = tz_task.result()
            state.tasks.append(
                self.executor.submit(
                    self.sync_invoke,
                    *extract_relation_declaration(
                        relation_types=None,
                        entities=extracted_declarations,
                        reference_time=state.reference_timestamp,
                        tz_info=parse_json(
                            response.content,
                            output_schema=TimezonePredictions.response_format(state.prompting.language),
                        )
                        or [],
                        entity_types=state.entity_types,
                        content=content,
                        history=state.history,
                        language=state.prompting.relation_extraction_language,
                    ),
                )
            )

            # Find relevant existing entities
            self._fetch_relevant_entities(extracted_declarations, no_existing_entity, user_id, state)

            # Entity merging
            existing_entities_list: List[Dict] = [entity.model_dump() for entity in state.retrieved_entities.values()]
            if existing_entities_list:
                state.tasks.append(
                    self.executor.submit(
                        self.sync_invoke,
                        *dedupe_entity_list(
                            content,
                            candidate_entities=extracted_declarations,
                            existing_entities=existing_entities_list,
                            entity_types=state.entity_types,
                            history=state.history,
                            language=state.prompting.entity_dedupe_language,
                        ),
                    )
                )
            extracted_declarations = self._entity_merge(extracted_declarations, existing_entities_list, state)

            response = state.tasks.pop(0).result()
            relations, entities = parse_all_relations(
                ensure_list(
                    parse_json(
                        response.content,
                        output_schema=RelationExtraction.response_format(state.prompting.relation_extraction_language),
                    )
                    or []
                ),
                entities=extracted_declarations,
                entity_types=state.entity_types,
                created_at=state.reference_timestamp,
                user_id=user_id,
            )

            # Extract summary & attribute for entities
            entities = self._entity_enrich(entities, content, state)
            # Parse relation filtering result
            self._parse_relation_filtering_result(relations, state)
            # Bulk-embed relations & de-duplication
            self._handle_relation_dedupe(user_id, content, relations, state)
            self._update_entities_for_relation_removal(state, extracted_declarations)

            # Post-process relations & entities, then persist to database!
            process_relations(self.db_backend, entities, relations, state)
            process_entities(self.db_backend, entities, current_episode, state)
            validate_entities_episodes(entities, current_episode, state)
            persist_to_db(self.executor, self.db_backend, state, self.db_backend.config)
            # Clean up resources
            state.clear_references()
            del relations, entities, current_episode, response, existing_entities_list
            del extracted_declarations
            self.db_backend.refresh()

        # Check if garbage collection should be manually invoked
        with self.thread_lock:
            if self.time_till_next_gc >= 0:
                if time.time() - self._last_gc > self.time_till_next_gc:
                    self._last_gc = time.time()
                    gc.collect()

        return state.mem_update, state.mem_update_skip_embed

    def sync_invoke(
        self, kwargs: Dict, template: Template, output_model: Optional[Dict] = None, **extra
    ) -> BaseMessage:
        """Synchronous invoke of LLM clients

        Args:
            kwargs (Dict): Keyword arguments to fill into prompt template.
            template (Template): Prompt template to use.
            output_model (Optional[Dict], optional): Response format for structured output. Defaults to None.
            **extra: Extra arguments to supply in request, such as enable_thinking=False.

        Returns:
            BaseMessage: LLM response.
        """
        params = self.llm_client.assemble_invoke_params(kwargs, template, output_model)
        params["client"] = next(self._clients)
        if self.llm_extra_kwargs is not None:
            params.update(self.llm_extra_kwargs)
        params.update(self._request_common_kwargs)
        params["token_record"] = self.token_record
        params["record_lock"] = self.thread_lock
        params.update(extra)
        response = self.llm_client.invoke(**params)
        if self.debug:
            sep = f"\n{'=' * 20}\n"
            query = params["messages"][-1]["content"]
            debug_msg = f"TEMPLATE {template.name}{sep}{query}{sep}{response.content}"
            with self.thread_lock:
                logger.debug(debug_msg)
        return response

    def init_state(self, reference_time: Optional[datetime.datetime] = None) -> GraphMemState:
        """Initialize GraphMemState for current memory update request"""
        strategy = self.default_extraction_strategy

        state = GraphMemState(strategy=strategy, entity_types=[EntityDef(), HumanEntity(), AIEntity()])
        state.prompting.language = self.language
        state.prompting.entity_extraction_language = "cn" if strategy.chinese_entity else self.language
        state.prompting.relation_extraction_language = "cn" if strategy.chinese_relation else self.language
        state.prompting.entity_dedupe_language = "cn" if strategy.chinese_entity_dedupe else self.language
        state.prompting.schema_entity_extraction = EntitySummary.response_format(self.language)
        state.prompting.schema_entity_dedupe = EntityDuplication.response_format(state.prompting.entity_dedupe_language)
        state.prompting.schema_relation_merge = MergeRelations.response_format(self.language)
        state.prompting.schema_relation_filter = RelevantFacts.response_format(self.language)
        state.extras = dict(summary_target=str(strategy.summary_target))

        if reference_time is None:
            state.reference_timestamp = state.current_timestamp
        elif isinstance(reference_time, datetime.datetime):
            state.reference_timestamp = int(safe_timestamp(reference_time))
        else:
            raise ValueError("reference_time must be a valid datetime object")

        return state

    def prepare_episodes(
        self,
        src_type: EpisodeType,
        user_id: str,
        content: Union[str, List[ChatMessage]],
        state: GraphMemState,
        content_fmt_kwargs: Optional[Dict] = None,
    ) -> str:
        """Preprocess episode content & retrieve relevant history episodes"""
        validate_add_memory_input(
            self.db_backend.config.db_storage_config.user_id, src_type, user_id, content_fmt_kwargs
        )
        if not isinstance(content, str):
            if src_type == EpisodeType.conversation:
                try:
                    content = msg2dict(content)
                    if not all((isinstance(msg, dict) and "role" in msg and "content" in msg) for msg in content):
                        raise ValueError('The content is not a list of dict with keys "role" and "content"')
                except Exception as e:
                    raise ValueError("The content must be str or list of messages in dict or BaseMessage") from e
                content = format_list_of_messages(content, role_replace=content_fmt_kwargs)
            else:
                raise ValueError("The content must be str when source type is not conversation")
        content = content.strip()

        # Retrieve Relevant Histories
        result: List[Episode] = []
        recall_strategy = state.strategy.recall_episode
        maximize = self.db_backend.config.db_embed_config.metric_is_sim or recall_strategy.rank_config.higher_is_better
        if recall_strategy.top_k and not self.db_backend.is_empty("episodes"):
            # Assemble search query
            query_components = []
            if user_id:
                query_components.append(query_expr.filter_user(user_id))
            if recall_strategy.same_kind:
                query_components.append(query_expr.eq("type", src_type.name))
            if recall_strategy.exclude_future_results:
                query_components.append(query_expr.lte("valid_since", state.reference_timestamp))
            ep_search_query = query_expr.chain_filters(query_components)
            # Retrieve episodes
            result = self.db_backend.search(
                content,
                k=recall_strategy.top_k,
                collection="episodes",
                ranker_config=recall_strategy.rank_config,
                filter_expr=ep_search_query,
                language=state.prompting.language,
            )["episodes"]
            if maximize:
                result = [r for r in result if r["distance"] >= recall_strategy.min_score]
            else:
                result = [r for r in result if r["distance"] <= recall_strategy.min_score]
            result = [Episode(**ep["entity"]) for ep in result]
            result.sort(key=lambda ep: ep.valid_since)
        for ep in result:
            state.lookup_table.episodes[ep.uuid] = ep
        state.history = "\n---\n".join(format_timestamp(ep.created_at) + "\n" + ep.content for ep in result)
        return content

    def search(
        self,
        query: str,
        user_id: Union[str, list[str]],
        search_strategy: str = "default",
        entity: bool = True,
        relation: bool = True,
        episode: bool = True,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, List[Tuple[float, BaseGraphObject]]]:
        """Search in the memory graph"""
        if search_strategy not in self._search_strategies:
            raise KeyError(
                f"Strategy [{search_strategy}] not found, please register with register_search_configs method."
            )
        if not isinstance(user_id, list):
            user_id = [user_id]
        if not all(isinstance(uid, str) and len(uid) <= 32 for uid in user_id):
            raise ValueError("user_id must be a string of length <= 32 or a list of such strings")
        if not (isinstance(query_embedding, list) and any(isinstance(val, (int, float)) for val in query_embedding)):
            query_embedding = self.db_backend.embedder.query_embedding(query)
        tasks, result = {}, {}

        if entity:
            self._perform_search(0, user_id, search_strategy, tasks, dict(query=query, query_embedding=query_embedding))
        if relation:
            self._perform_search(1, user_id, search_strategy, tasks, dict(query=query, query_embedding=query_embedding))
        if episode:
            self._perform_search(2, user_id, search_strategy, tasks, dict(query=query, query_embedding=query_embedding))
        for task in as_completed(tasks):
            cls = dict(entity=Entity, relation=Relation, episode=Episode)[tasks[task]]
            returned_list = []
            for returned_dict in task.result():
                returned_list.append((returned_dict["distance"], cls(**returned_dict["entity"])))
            result[tasks[task]] = returned_list
        return result

    def _perform_search(self, col_idx: int, user_id: str, search_strategy: str, tasks: Dict[Future, str], kwargs: Dict):
        filter_by_user = query_expr.filter_user(user_id)
        names = ["entity", "relation", "episode"]
        config_e = self._search_strategies[search_strategy][col_idx]
        config_e = config_e.model_copy()
        config_e.filter_expr = config_e.filter_expr & filter_by_user if config_e.filter_expr else filter_by_user
        tasks[self.executor.submit(self._search, col=names[col_idx], search_config=config_e, **kwargs)] = names[col_idx]

    def _search(self, col: str, query: str, search_config: SearchConfig, query_embedding: Optional[List[float]] = None):
        return self.db_backend.search(
            query=query,
            k=search_config.top_k,
            collection=col,
            ranker_config=search_config.rank_config,
            bfs_depth=search_config.bfs_depth,
            bfs_k=search_config.bfs_k,
            filter_expr=search_config.filter_expr,
            output_fields=search_config.output_fields,
            language=search_config.language,
            query_embedding=query_embedding,
            reranker=self.reranker if search_config.rerank else None,
        )[col]

    def _extract_entity_declarations(
        self,
        src_type: EpisodeType,
        content: str,
        state: GraphMemState,
    ) -> Tuple[bool, List[Union[Entity, EntityDeclaration]]]:
        """Extract entity declarations"""
        prompt_entity_extraction = extract_entity_declaration(
            src_type=src_type,
            content=content,
            history=state.history,
            entity_types=state.entity_types,
            language=state.prompting.entity_extraction_language,
            extras=state.extras,
        )

        entity_names = {"user", "assistant", "User", "Assistant", "USER", "ASSISTANT"}
        response = self.sync_invoke(*prompt_entity_extraction)
        extracted_declarations = parse_json(response.content, output_schema=prompt_entity_extraction[-1]) or []

        if isinstance(extracted_declarations, dict):
            extracted_list = next(iter(extracted_declarations.values()))
            if isinstance(extracted_list, dict):
                extracted_declarations = [extracted_list]
            elif isinstance(extracted_list, list):
                extracted_declarations = extracted_list

        if isinstance(extracted_declarations, list):
            extracted_list = []
            for extraction in extracted_declarations:
                name = extraction.pop("name", "")
                if isinstance(name, str):
                    name = name.strip()
                else:
                    name = ""
                if name and name not in entity_names:
                    entity_names.add(name)
                    type_id = next(v for k, v in extraction.items() if isinstance(k, str) and "type" in k.casefold())
                    extracted_list.append(dict(name=name, entity_type_id=type_id))
            extracted_declarations = extracted_list
        else:
            extracted_declarations = []

        extracted_declarations: List[Union[Entity, EntityDeclaration]] = [
            EntityDeclaration(**ent) for ent in extracted_declarations if ent.get("name", "").strip()
        ]
        extracted_entity_names = [ent.name for ent in extracted_declarations]

        # Bluk-Embed Name of New Entities
        no_existing_entity = self.db_backend.is_empty("entities")
        if not no_existing_entity:
            if isinstance(self.db_backend.embedder, VarDimEmbeddingService):
                self.batch_embed(extracted_entity_names, task_queue=state.tasks)
            else:
                state.tasks.append(
                    self.executor.submit(
                        embed_loop, embedding_service=self.db_backend.embedder, queries=extracted_entity_names
                    )
                )
        return no_existing_entity, extracted_declarations

    def _fetch_relevant_entities(
        self,
        extracted_declarations: List[Union[Entity, EntityDeclaration]],
        no_existing_entity: bool,
        user_id: str,
        state: GraphMemState,
    ):
        """Find Relevant Existing Entities"""
        if not no_existing_entity:
            entity_embed_results = list(chain(*[t.result() for t in state.tasks[:-1]]))
            state.tasks = state.tasks[-1:]
            for entity, emb in zip(extracted_declarations, entity_embed_results):
                # Search for semantically relevant entities
                entity_type = (
                    state.entity_types[entity.entity_type_id]
                    if entity.entity_type_id < len(state.entity_types)
                    else None
                )
                filter_by_user = query_expr.filter_user(user_id)
                # If not set to only search in same kind/type, then also perform a type-less search
                if not state.strategy.recall_entity.same_kind:
                    result = self.db_backend.search(
                        entity.name,
                        k=state.strategy.recall_entity.top_k,
                        collection="entities",
                        ranker_config=state.strategy.recall_entity.rank_config,
                        filter_expr=filter_by_user,
                        query_embedding=emb,
                        language=state.prompting.language,
                    )["entities"]
                    if (
                        self.db_backend.config.db_embed_config.metric_is_sim
                        or state.strategy.recall_entity.rank_config.higher_is_better
                    ):
                        result = [r for r in result if r["distance"] >= state.strategy.recall_entity.min_score]
                    else:
                        result = [r for r in result if r["distance"] <= state.strategy.recall_entity.min_score]
                    for r in result:
                        state.retrieved_entities[r["entity"]["uuid"]] = state.lookup_table.get_entity(r["entity"])
                # Always perform a typed search
                if entity_type is not None:
                    result = self.db_backend.search(
                        entity.name,
                        k=state.strategy.recall_entity.top_k,
                        collection="entities",
                        ranker_config=state.strategy.recall_entity.rank_config,
                        filter_expr=filter_by_user & query_expr.MatchFilter(field="type", value=entity_type.name),
                        query_embedding=emb,
                        language=state.prompting.language,
                    )["entities"]
                else:
                    result = []
                if (
                    self.db_backend.config.db_embed_config.metric_is_sim
                    or state.strategy.recall_entity.rank_config.higher_is_better
                ):
                    result = [r for r in result if r["distance"] >= state.strategy.recall_entity.min_score]
                else:
                    result = [r for r in result if r["distance"] <= state.strategy.recall_entity.min_score]
                for r in result:
                    state.retrieved_entities[r["entity"]["uuid"]] = state.lookup_table.get_entity(r["entity"])

                # Match entities with same name
                exact_match = query_expr.MatchFilter(field="name", value=entity.name, match_mode="exact")
                infix_match = query_expr.MatchFilter(field="name", value=entity.name, match_mode="infix")
                result = self.db_backend.query(
                    collection="entities",
                    expr=query_expr.filter_user(user_id) & exact_match,
                    limit=ceil(state.strategy.recall_entity.top_k / 2),
                    silence_errors=True,
                ) + self.db_backend.query(
                    collection="entities",
                    expr=query_expr.filter_user(user_id) & infix_match,
                    limit=ceil(state.strategy.recall_entity.top_k / 2),
                    silence_errors=True,
                )
                for e in result:
                    state.retrieved_entities[e["uuid"]] = state.lookup_table.get_entity(e)

    def _resolve_entity_merges(self, merging_args: List[Tuple[Entity, List[Entity]]], state: GraphMemState):
        """Update the relations / episodes for entities to delete"""
        episodes_to_update = set()
        entity_relation_updates: Dict[str, Dict[str, Relation]] = {}
        map_src2tgt: Dict[str, str] = dict()
        for tgt_entity, src_entities in merging_args:
            tgt_uuid = tgt_entity.uuid
            state.merge_infos[tgt_uuid] = EntityMerge(target=tgt_entity, source={e.uuid: e for e in src_entities})
            alias = set(state.merge_infos[tgt_uuid].source.keys())  # UUIDs that point to same entitiy after merging
            alias.add(tgt_uuid)
            state.relation_deferred_updates[tgt_uuid] = []
            entity_relation_updates[tgt_uuid] = {}
            for src_entity in src_entities:
                map_src2tgt[src_entity.uuid] = tgt_uuid
                src_episode_uuids = [ep if isinstance(ep, str) else ep.uuid for ep in src_entity.episodes]
                tgt_entity.episodes.extend(src_episode_uuids)
                tgt_entity.episodes.extend(src_episode_uuids)
                episodes_to_update.update(src_episode_uuids)
                if not src_entity.relations:
                    continue
                self._resolve_each_relation(tgt_uuid, src_entity, alias, map_src2tgt, entity_relation_updates, state)
            tgt_entity.episodes = list(set(tgt_entity.episodes))
            tgt_entity.episodes = list(set(tgt_entity.episodes))

        state.mem_update.removed_relation.update(state.faulty_relations.keys())
        self._dispatch_entity_merge_tasks(episodes_to_update, entity_relation_updates, state)

    def _dispatch_entity_merge_tasks(
        self,
        episodes_to_update: Set[str],
        entity_relation_updates: Dict[str, Dict[str, Relation]],
        state: GraphMemState,
    ):
        """Dispatch the relation filtering tasks"""
        obj_cache = state.lookup_table
        if state.strategy.merge_filter:
            for tgt_uuid, relation_dict in entity_relation_updates.items():
                tgt_entity = obj_cache.entities[tgt_uuid]
                relation_list = [r for r in relation_dict.values() if r.uuid not in state.faulty_relations]
                state.merge_infos[tgt_uuid].new_relations = relation_list
                prompt_relation_filter = filter_relations_for_merge(
                    tgt_entity, relation_list, state.prompting.language, state.extras
                )
                task = self.executor.submit(self.sync_invoke, *prompt_relation_filter)
                state.relation_filter_tasks[task] = (tgt_entity, relation_list)
        # Finally, update the episodes
        if episodes_to_update:
            episodes = [
                obj_cache.get_episode(e) for e in self.db_backend.query("episodes", ids=list(episodes_to_update))
            ]
            state.mem_update_skip_embed.updated_episode.extend(episodes)

    def _resolve_each_relation(
        self,
        tgt_uuid: str,
        src_entity: Entity,
        alias: Set[str],
        map_src2tgt: Dict[str, str],
        entity_relation_updates: Dict[str, Dict[str, Relation]],
        state: GraphMemState,
    ):
        self_pointing: Set[str] = set()
        src_relations = [
            state.lookup_table.get_relation(r) for r in self.db_backend.query("relations", ids=src_entity.relations)
        ]
        for relation in src_relations:
            to_replace = src_entity.uuid
            lhs_rhs = {relation.lhs, relation.rhs}
            # Self-pointing -> remove it!
            if all(e_uuid in alias for e_uuid in lhs_rhs):
                state.faulty_relations[relation.uuid] = relation
                self_pointing.add(relation.uuid)
                to_replace = None
            # Should always terminate within one iteration, while loop for safety
            while to_replace in map_src2tgt and relation.uuid not in state.faulty_relations:
                if relation.lhs == to_replace:
                    self._replace_one_side_of_relation("lhs", relation, tgt_uuid, entity_relation_updates, state)
                    break
                if relation.rhs == to_replace:
                    self._replace_one_side_of_relation("rhs", relation, tgt_uuid, entity_relation_updates, state)
                    break
                to_replace = map_src2tgt[to_replace]
            else:
                if relation.uuid not in self_pointing:
                    relation_repr = f"[{relation.lhs}]-<{relation.uuid}>-[{relation.rhs}]"
                    logger.warning(
                        "[Graph Mem] Relation [%s] not connected to entity [%s] (caught remapping to -> [%s])\n%s",
                        relation.uuid,
                        src_entity.uuid,
                        tgt_uuid,
                        relation_repr,
                    )
                    state.faulty_relations[relation.uuid] = relation

    def _entity_merge(
        self, extracted_declarations: List, existing_entities_list: List[Dict], state: GraphMemState
    ) -> List:
        """Merge entities"""
        wait_futures(state.tasks)
        if existing_entities_list:
            response_resolve_entity = state.tasks.pop().result()
            existing_entities = [state.lookup_table.get_entity(entity_dict) for entity_dict in existing_entities_list]
            dedupe_entity = (
                parse_json(response_resolve_entity.content, output_schema=state.prompting.schema_entity_dedupe) or []
            )
            dedupe_entity = ensure_list(dedupe_entity)

            # If existing entities need merging:
            # - merging_args contains argument tuples for merging tasks: [(target, [source_1, source_2, ...]), ...]
            # - entity_uuids_to_remove contains uuid of entities to remove due to merging
            extracted_declarations, merging_args, entity_uuids_to_remove = resolve_entities(
                extracted_declarations, existing_entities, dedupe_entity
            )
            if not state.strategy.merge_entities:
                merging_args.clear()
            else:
                state.mem_update.removed_entity.update(entity_uuids_to_remove)
            non_blocking_tasks = []  # Merging tasks that would not block entity summary & attribute extractions
            # Dispatch the blocking tasks first
            for tgt, src_entities in merging_args:
                prompt_entity_merge = merge_existing_entities(tgt, src_entities, language=state.prompting.language)
                if tgt in extracted_declarations:
                    # Submit tasks that would block entity summary / attribute extraction
                    task = self.executor.submit(self.sync_invoke, *prompt_entity_merge)
                    state.pending_merge[tgt.uuid] = task
                    state.merging_tasks.append(task)
                    state.merging_tasks_entities[task] = tgt
                else:
                    # Non blocking tasks can wait
                    non_blocking_tasks.append((tgt, prompt_entity_merge))
            # Then the non-blocking tasks
            for tgt, prompt_entity_merge in non_blocking_tasks:
                task = self.executor.submit(self.sync_invoke, *prompt_entity_merge)
                state.merging_tasks.append(task)
                state.merging_tasks_entities[task] = tgt

            # Resolve relations after the entity merges
            self._resolve_entity_merges(merging_args, state)
        return extracted_declarations

    def _entity_enrich(self, entities: List[Entity], content: str, state: GraphMemState) -> List[Entity]:
        state.tasks.clear()
        # Classify entities
        entities_blocking: List[Entity] = []
        entities_non_blocking: List[Entity] = []
        for entity in entities:
            if entity.uuid in state.pending_merge:
                entities_blocking.append(entity)
            else:
                entities_non_blocking.append(entity)
        entities = entities_non_blocking + entities_blocking

        # Start non-blocking tasks first
        for entity in entities_non_blocking:
            prompt_entity_summary = extract_entity_attributes(
                entity=entity,
                content=content,
                history=state.history,
                language=state.prompting.language,
                extras=state.extras,
            )
            state.tasks.append(self.executor.submit(self.sync_invoke, *prompt_entity_summary))

        # Blocking tasks needs to wait
        for entity in entities_blocking:
            task = state.pending_merge[entity.uuid]
            response = task.result()
            state.merging_tasks.remove(task)
            update_entity(entity, response.content, state.prompting.schema_entity_extraction)
            prompt_entity_summary = extract_entity_attributes(
                entity=entity,
                content=content,
                history=state.history,
                language=state.prompting.language,
                extras=state.extras,
            )
            state.tasks.append(self.executor.submit(self.sync_invoke, *prompt_entity_summary))
        wait_futures(state.tasks)

        # Update entities
        for entity, future in zip(entities, state.tasks):
            response = future.result()
            update_entity(entity, response.content, state.prompting.schema_entity_extraction)
        state.tasks.clear()
        return entities

    def _handle_relation_dedupe(
        self,
        user_id: str,
        content: str,
        relations: List[Relation],
        state: GraphMemState,
    ):
        """Bulk-Embed Relations & De-Duplication"""
        # Filter out self-pointing relations (fact about object)
        for relation in state.to_remove:
            if relation in relations:
                relations.remove(relation)
        # Bulk-embed & de-duplicate
        if state.strategy.merge_relations and state.tmp_buffer and not self.db_backend.is_empty("relations"):
            if isinstance(self.db_backend.embedder, VarDimEmbeddingService):
                self.batch_embed(state.tmp_buffer, task_queue=state.tasks)
            else:
                state.tasks.append(
                    self.executor.submit(
                        embed_loop, embedding_service=self.db_backend.embedder, queries=state.tmp_buffer
                    )
                )
            self._relation_dedupe(user_id, content, relations, list(chain(*[t.result() for t in state.tasks])), state)

    def _relation_dedupe(
        self,
        user_id: str,
        content: str,
        relations: List[Relation],
        relation_embed_results: List[List[float]],
        state: GraphMemState,
    ):
        state.tasks.clear()
        dedupe_relation_tasks = []
        for relation, emb in zip(relations, relation_embed_results):
            current_relations: List[Relation] = []
            lhs_rhs = [
                e if isinstance(e, str) else (e.uuid if e.content.strip() else None)
                for e in (relation.lhs, relation.rhs)
            ]
            if not all(lhs_rhs):
                continue
            result = self.db_backend.search(
                relation.content,
                k=state.strategy.recall_relation.top_k,
                collection="relations",
                ranker_config=state.strategy.recall_relation.rank_config,
                filter_expr=query_expr.in_list("lhs", lhs_rhs)
                & query_expr.in_list("rhs", lhs_rhs)
                & query_expr.filter_user(user_id),
                query_embedding=emb,
                language=state.prompting.language,
            )["relations"]
            if (
                self.db_backend.config.db_embed_config.metric_is_sim
                or state.strategy.recall_relation.rank_config.higher_is_better
            ):
                result = [r for r in result if r["distance"] >= state.strategy.recall_relation.min_score]
            else:
                result = [r for r in result if r["distance"] <= state.strategy.recall_relation.min_score]

            for r in result:
                relation = state.retrieved_relations[r["entity"]["uuid"]] = state.lookup_table.get_relation(r["entity"])
                current_relations.append(relation)

            if current_relations:
                prompt_relation_dedupe = dedupe_relation_list(
                    content=content,
                    relation=relation,
                    existing_relations=current_relations,
                    existing_entities=[relation.lhs.model_dump(), relation.rhs.model_dump()],
                    history=state.history,
                    language=state.prompting.language,
                )
                state.tasks.append(self.executor.submit(self.sync_invoke, *prompt_relation_dedupe))
                dedupe_relation_tasks.append((relation, state.retrieved_relations, state.tasks[-1]))
        wait_futures(state.tasks)
        state.tasks.clear()
        parse_relation_uuids_to_remove(dedupe_relation_tasks, state)

    def _update_entities_for_relation_removal(
        self,
        state: GraphMemState,
        update_needs_embed: List[Union[Entity, EntityDeclaration]],
    ):
        """Update entities affected by relation removal"""
        entities_to_remove_relations_from = set()
        for relation in state.to_remove:
            entities_to_remove_relations_from.add(relation.lhs if isinstance(relation.lhs, str) else relation.lhs.uuid)
            entities_to_remove_relations_from.add(relation.rhs if isinstance(relation.rhs, str) else relation.rhs.uuid)

        if entities_to_remove_relations_from:
            entities_retrieved = [
                state.lookup_table.get_entity(e)
                for e in self.db_backend.query("entities", ids=list(entities_to_remove_relations_from))
            ]
            for entity in entities_retrieved:
                if entity.uuid in state.lookup_table.entities:
                    entity = state.lookup_table.entities[entity.uuid]
                update_without_embed = needs_re_embed = False
                for existing_entitiy in update_needs_embed:
                    if isinstance(existing_entitiy, Entity) and existing_entitiy.uuid == entity.uuid:
                        entity = existing_entitiy
                        needs_re_embed = True
                for relation_uuid in state.mem_update.removed_relation.intersection(entity.relations):
                    entity.relations.remove(relation_uuid)
                    if not needs_re_embed:
                        update_without_embed = True
                if (
                    update_without_embed
                    and entity not in state.mem_update_skip_embed.updated_entity
                    and entity.uuid not in state.mem_update.removed_entity
                ):
                    state.mem_update_skip_embed.updated_entity.append(entity)
