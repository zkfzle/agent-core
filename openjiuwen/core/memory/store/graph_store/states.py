# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Set, Tuple, Union

from openjiuwen.core.memory.config.graph import AddMemStrategy
from openjiuwen.core.memory.config.graph.config import EpisodeType, GraphConfig
from openjiuwen.core.memory.generation.graph.entity_type_definition import EntityDef
from openjiuwen.core.memory.generation.graph.extraction_models import (
    EntityDuplication,
    EntitySummary,
    MergeRelations,
    RelevantFacts,
)
from openjiuwen.core.memory.store.graph_store.backends.graph_backend import GraphBackend
from openjiuwen.core.memory.store.graph_store.backends.utils import batched
from openjiuwen.core.memory.store.graph_store.utils import get_current_utc_timestamp

from .api_services.embedding_service import EmbeddingService, VarDimEmbeddingService
from .graph_objects import BaseGraphObject, Entity, Episode, Relation


def nested_clear_dataclass(data_obj: Any):
    """Recursively call clear method on input dataclass's fields"""

    if not is_dataclass(data_obj):
        return
    for f in fields(data_obj):
        field_val = getattr(data_obj, f.name)
        if hasattr(field_val, "clear"):
            clear_method = getattr(field_val, "clear")
            if isinstance(clear_method, Callable):
                clear_method()


@dataclass
class LookupTables:
    """Lookup Tables for UUID-to-Entity/Relation"""

    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    episodes: Dict[str, Episode] = field(default_factory=dict)

    def get_entity(
        self,
        input_obj: Mapping,
    ) -> Entity:
        """Get entity in duplication-safe manner"""
        entity_id = input_obj["uuid"]
        entity = self.entities.get(entity_id)
        if entity is None:
            self.entities[entity_id] = entity = Entity(**input_obj)
        return entity

    def get_relation(self, input_obj: Mapping) -> Relation:
        """Get relation in duplication-safe manner"""
        relation_id = input_obj["uuid"]
        relation = self.relations.get(relation_id)
        if relation is None:
            self.relations[relation_id] = relation = Relation(**input_obj)
        return relation

    def get_episode(self, input_obj: Mapping) -> Episode:
        """Get episode in duplication-safe manner"""
        episode_id = input_obj["uuid"]
        episode = self.episodes.get(episode_id)
        if episode is None:
            self.episodes[episode_id] = episode = Episode(**input_obj)
        return episode

    def clear(self):
        """Clear references to prevent memory leak"""
        nested_clear_dataclass(self)


@dataclass
class EntityMerge:
    """Lookup Tables for UUID-to-Entity/Relation"""

    target: Entity
    source: Dict[str, Entity] = field(default_factory=dict)
    new_relations: List[Relation] = field(default_factory=list)
    relations_to_keep: Set[str] = field(default_factory=set)

    def clear(self):
        """Clear references to prevent memory leak"""
        nested_clear_dataclass(self)


@dataclass
class GraphMemUpdate:
    """Graph Memory Update"""

    added_episode: List[Episode] = field(default_factory=list)
    updated_episode: List[Episode] = field(default_factory=list)
    added_entity: List[Entity] = field(default_factory=list)
    updated_entity: List[Entity] = field(default_factory=list)
    added_relation: List[Relation] = field(default_factory=list)
    updated_relation: List[Relation] = field(default_factory=list)
    removed_entity: Set[str] = field(default_factory=set)
    removed_relation: Set[str] = field(default_factory=set)


@dataclass
class GraphMemPrompting:
    """Schema Definitions for Graph Memory Addition"""

    # Output Schema
    schema_entity_extraction: Dict[str, Any] = field(default_factory=EntitySummary.response_format)
    schema_entity_dedupe: Dict[str, Any] = field(default_factory=EntityDuplication.response_format)
    schema_relation_merge: Dict[str, Any] = field(default_factory=MergeRelations.response_format)
    schema_relation_filter: Dict[str, Any] = field(default_factory=RelevantFacts.response_format)

    # Prompt Language
    language: str = field(default="cn")
    entity_extraction_language: str = field(default="cn")
    relation_extraction_language: str = field(default="cn")
    entity_dedupe_language: str = field(default="cn")

    def clear(self):
        """Clear references to prevent memory leak"""
        nested_clear_dataclass(self)


@dataclass
class GraphMemState:
    """Current State of Graph Memory Addition"""

    # Task buffers (deferred update or concurrency)
    tasks: list[Future] = field(default_factory=list)
    merging_tasks: List[Future] = field(default_factory=list)
    merging_tasks_entities: Dict[Future, Entity] = field(default_factory=dict)
    pending_merge: Dict[str, Future] = field(default_factory=dict)
    relation_deferred_updates: Dict[str, List[Tuple[Relation, str, str]]] = field(default_factory=dict)
    relation_filter_tasks: Dict[Future, Tuple[Entity, List[Relation]]] = field(default_factory=dict)

    # General-purpose temporary buffers
    to_remove: List[Union[BaseGraphObject, str]] = field(default_factory=list)
    tmp_buffer: List = field(default_factory=list)

    # Special-purpose temporary buffers
    updated_entities_in_current_ep: List[Entity] = field(default_factory=list)
    retrieved_entities: Dict[str, Entity] = field(default_factory=dict)
    retrieved_relations: Dict[str, Relation] = field(default_factory=dict)
    faulty_relations: Dict[str, Relation] = field(default_factory=dict)
    merge_infos: Dict[str, EntityMerge] = field(default_factory=dict)

    # Changes to memory (accumulate to flush all at the end for safety)
    mem_update: GraphMemUpdate = field(default_factory=GraphMemUpdate)
    mem_update_skip_embed: GraphMemUpdate = field(default_factory=GraphMemUpdate)

    # Shared variables / dictionaries
    current_timestamp: int = field(default_factory=get_current_utc_timestamp)
    reference_timestamp: int = field(default=0)
    lookup_table: LookupTables = field(default_factory=LookupTables)
    extras: Dict[str, Any] = field(default_factory=dict)
    strategy: AddMemStrategy = field(default_factory=AddMemStrategy)
    prompting: GraphMemPrompting = field(default_factory=GraphMemPrompting)
    entity_types: List[EntityDef] = field(default_factory=list)
    episode_type: EpisodeType = field(default=EpisodeType.conversation)
    content: str = field(default="")
    history: str = field(default="")

    def clear_references(self):
        """Clear references to prevent memory leak"""
        for merge_info in self.merge_infos.values():
            nested_clear_dataclass(merge_info)
        nested_clear_dataclass(self)


def batch_embed(
    data: List[BaseGraphObject], embedding_service: EmbeddingService, executor: ThreadPoolExecutor, config: GraphConfig
) -> List[BaseGraphObject]:
    """Embed graph objects in batches"""

    not_embedded: List[BaseGraphObject] = []
    tasks: List[Future] = []
    embed_tasks = []
    for graph_object in data:
        embed_tasks.extend(graph_object.fetch_embed_task())
    _reset_embedding(embed_tasks)

    if isinstance(embedding_service, VarDimEmbeddingService):
        extra_kwargs = dict(max_retry=config.request_max_retry, retry_wait=config.request_retry_wait)
    else:
        extra_kwargs = {}
    embed_batches = list(batched(embed_tasks, config.embed_batch_size))
    for batch in embed_batches:
        query = [task_tuple[-1] for task_tuple in batch]  # task_tuple[-1] is query to embed
        if not isinstance(embedding_service, VarDimEmbeddingService):
            query = query[0]
        tasks.append(executor.submit(embedding_service.query_embedding, query, **extra_kwargs))

    for batch, future in zip(embed_batches, tasks):
        try:
            embed_result = future.result()
            if not embed_result:
                raise ConnectionError("Embedding Service is not available")
            if not isinstance(embedding_service, VarDimEmbeddingService):
                embed_result = [embed_result]
            _set_embed_results(batch, embed_result)
        except Exception:
            not_embedded.extend(task[0] for task in batch)
    return not_embedded


def persist_to_db(executor: ThreadPoolExecutor, db_backend: GraphBackend, state: GraphMemState, config: GraphConfig):
    """Persist data to database in a safe manner"""

    # For safety, validate mem_update_skip_embed.updated_entity first
    state.tmp_buffer.clear()
    for entity in state.mem_update_skip_embed.updated_entity:
        if entity.content_embedding is None or entity.name_embedding is None:
            state.tmp_buffer.append(entity)
            if entity not in state.mem_update.updated_entity:
                state.mem_update.updated_entity.append(entity)
    for entity in state.tmp_buffer:
        while entity in state.mem_update_skip_embed.updated_entity:
            state.mem_update_skip_embed.updated_entity.remove(entity)

    # Try to embed new entity, new relation and updated entity
    graph_objects_to_embed = (
        state.mem_update.added_entity + state.mem_update.added_relation + state.mem_update.updated_entity
    )
    retries = config.request_max_retry
    while retries > 0:
        graph_objects_to_embed = batch_embed(graph_objects_to_embed, db_backend.embedder, executor, config)
        if not graph_objects_to_embed:
            break
        retries -= 1
    else:
        raise ConnectionError("Unable to access embedding service")

    # Try to embed new episode
    graph_objects_to_embed = state.mem_update.added_episode
    retries = config.request_max_retry
    while retries > 0:
        graph_objects_to_embed = batch_embed(graph_objects_to_embed, db_backend.embedder, executor, config)
        if not graph_objects_to_embed:
            break
        # Maybe episode
        episode = state.mem_update.added_episode[0]
        episode.content = episode.content[: len(episode.content) // 2]
        retries -= 1
    else:
        raise ConnectionError("Unable to access embedding service for new episode, maybe exceeding context limit")

    db_backend.add_entities(entities=state.mem_update.added_entity, is_upsert=False, flush=False, skip_embed=True)
    db_backend.add_relations(relations=state.mem_update.added_relation, is_upsert=False, flush=False, skip_embed=True)
    db_backend.add_episodes(episodes=state.mem_update.added_episode, is_upsert=False, flush=False, skip_embed=True)
    db_backend.add_entities(entities=state.mem_update.updated_entity, is_upsert=True, flush=False, skip_embed=True)

    if state.mem_update_skip_embed.updated_episode:
        db_backend.add_episodes(
            episodes=state.mem_update_skip_embed.updated_episode, flush=False, is_upsert=True, skip_embed=True
        )
    if state.mem_update_skip_embed.updated_entity:
        db_backend.add_entities(
            state.mem_update_skip_embed.updated_entity, flush=False, is_upsert=True, skip_embed=True
        )
    if state.mem_update_skip_embed.updated_relation:
        db_backend.add_relations(
            state.mem_update_skip_embed.updated_relation, flush=False, is_upsert=True, skip_embed=True
        )
    if state.mem_update.removed_entity:
        db_backend.delete(collection="entities", ids=list(state.mem_update.removed_entity))
    if state.mem_update.removed_relation:
        db_backend.delete(collection="relations", ids=list(state.mem_update.removed_relation))


def classify_relations_extracted(relations: List[Relation], state: GraphMemState):
    """Classify relations due to entity merging"""
    # Classify relations to keep & remove
    for merge_info in state.merge_infos.values():
        for relation in merge_info.new_relations:
            lhs_uuid = getattr(relation.lhs, "uuid", relation.lhs)
            rhs_uuid = getattr(relation.rhs, "uuid", relation.rhs)
            if lhs_uuid != rhs_uuid:
                merge_info.relations_to_keep.add(relation.uuid)
            else:
                state.mem_update.removed_relation.add(relation.uuid)
        merge_info.target.relations = list(merge_info.relations_to_keep.union(merge_info.target.relations))
    # Classify relations extracted
    state.tmp_buffer.clear()  # relations to embed
    for relation in relations:
        # Record self-pointing relations (fact about object)
        relation.language = state.prompting.language
        if not relation.content.strip():
            state.to_remove.append(relation)
        elif relation.lhs == relation.rhs:
            content = relation.lhs.content.removesuffix("\n")
            relation.lhs.content = f"{content}\n- {relation.content}"
            state.to_remove.append(relation)
        else:
            state.tmp_buffer.append(relation.content)  # used in GraphMemory._relation_dedupe


def _set_embed_results(batch: Iterable[Tuple[object, str, Any]], embed_result: Iterable[List[float]]):
    for (obj, attribute, _), embedding in zip(batch, embed_result):
        setattr(obj, attribute, embedding)


def _reset_embedding(embed_tasks: Iterable[Tuple[object, str, Any]]):
    for obj, attribute, _ in embed_tasks:
        setattr(obj, attribute, None)
