# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any, Dict, List, Optional, Self, Sequence, Tuple, Union

from pydantic import BaseModel, Field, field_serializer, model_validator

from openjiuwen.core.memory.generation.graph.parse_response import parse_json

from .utils import get_current_utc_timestamp, get_uuid


class BaseGraphObject(BaseModel):
    """Base class for all graph objects with common properties."""

    uuid: str = Field(default_factory=get_uuid, description="Unique 64-bit identifier for the object")
    created_at: int = Field(
        default_factory=get_current_utc_timestamp,
        description="Timestamp when the object was created",
    )
    user_id: Optional[str] = Field(default=None, description="User identifier")
    type: Optional[str] = Field(default=None, description="Object type classification")
    language: str = Field(
        default="cn",
        pattern=r"^[ce]n$",
        description="Language identifier for multi-analyzer support (cn/en)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="JSON metadata for additional object properties",
    )
    content: Optional[str] = Field(default=None, description="Text content for full-text search")
    content_embedding: Optional[Sequence[float]] = Field(
        default=None, description="Dense vector embeddings for content semantic search", repr=False
    )
    content_bm25: Optional[Sequence[float]] = Field(
        default=None, description="BM25 sparse vector for full-text search", repr=False
    )

    @model_validator(mode="after")
    def set_default_values(self) -> Self:
        """Set default values"""
        if getattr(self, "valid_since", 0) == -1:
            setattr(self, "valid_since", self.created_at)
        if getattr(self, "metadata") is None:
            setattr(self, "metadata", dict())
        if getattr(self, "attributes", 0) is None:
            setattr(self, "attributes", dict())
        return self

    def fetch_embed_task(self) -> List[Tuple[Self, str, str]]:
        """Fetch embedding task 3-tuples (self, attribute_name, content_to_embed)"""
        return [(self, "content_embedding", self.content)]


class NamedGraphObject(BaseGraphObject):
    """Base class for graph objects with names."""

    name: Optional[str] = Field(default=None, description="Name field")


class Entity(NamedGraphObject):
    """Node / Entity representing entity nodes in graph."""

    type: str = Field(default="Entity", description="Object type classification")
    name_embedding: Optional[Sequence[float]] = Field(
        default=None, description="Dense vector embeddings for name semantic search", repr=False
    )
    relations: List[Union[BaseGraphObject, str]] = Field(
        default_factory=list, description="Array of relation IDs for this entity", repr=False
    )
    episodes: List[str] = Field(default_factory=list, description="Episodes where this entity is mentioned", repr=False)
    attributes: Optional[Dict] = Field(default_factory=dict, description="Entity attributes")

    def fetch_embed_task(self) -> List[Tuple[Self, str, str]]:
        """Fetch embedding task 3-tuples (self, attribute_name, content_to_embed)"""
        return [(self, "content_embedding", self.content), (self, "name_embedding", self.name)]

    @field_serializer("relations", "episodes")
    def serialize(self, graph_obj_list: List[Union[BaseGraphObject, str]], _info) -> List[str]:
        """Serialization helper"""
        return sorted({obj if isinstance(obj, str) else obj.uuid for obj in graph_obj_list})


class Relation(NamedGraphObject):
    """Edge / Relation entity representing relationships between entity nodes."""

    type: str = Field(default="Relation", description="Object type classification")
    valid_since: int = Field(default=-1, description="Timestamp when the relation becomes valid")
    valid_until: int = Field(default=-1, description="Timestamp when the relation expires")
    offset_since: int = Field(default=0, description="Timezone offset of valid_since")
    offset_until: int = Field(default=0, description="Timezone offset of valid_until")
    lhs: Union[BaseGraphObject, str] = Field(description="Left-hand side entity UUID")
    rhs: Union[BaseGraphObject, str] = Field(description="Right-hand side entity UUID")

    def update_connected_entities(self) -> Self:
        """Update the connected entities"""
        for field_name in ["lhs", "rhs"]:
            connected_node: Union[BaseGraphObject, str] = getattr(self, field_name)
            if isinstance(connected_node, BaseGraphObject):
                if (self not in connected_node.relations) and (self.uuid not in connected_node.relations):
                    connected_node.relations.append(self)
        return self

    @field_serializer("lhs", "rhs")
    def serialize(self, ent: Union[Entity, str], _info) -> str:
        """Serialization helper"""
        if isinstance(ent, str):
            return ent
        return ent.uuid


class Episode(BaseGraphObject):
    """Episode nodes with no name"""

    type: str = Field(default="Episode", description="Object type classification")
    valid_since: int = Field(default=-1, description="Timestamp when the episode becomes valid")
    entities: List[str] = Field(default_factory=list, description="Entities mentioned in this episode")

    @field_serializer("entities")
    def serialize(self, graph_obj_list: List[Union[BaseGraphObject, str]], _info) -> List[str]:
        """Serialization helper"""
        return sorted({obj if isinstance(obj, str) else obj.uuid for obj in graph_obj_list})


def update_entity(entity: Entity, response: str, extraction_schema: dict):
    """Update entity content based on LLM response"""
    extracted_entity_info = parse_json(response, output_schema=extraction_schema) or {}
    if isinstance(extracted_entity_info, list):
        extracted_entity_info = extracted_entity_info[0]
    if isinstance(extracted_entity_info, str):
        extracted_entity_info = dict(summary=extracted_entity_info)
    _parse_summary(entity, extracted_entity_info)
    _parse_attributes(entity, extracted_entity_info)


def _parse_summary(entity: Entity, extracted_entity_info: Dict[str, Any]):
    """Process extracted summary"""
    summary = extracted_entity_info.get("summary", "")
    if isinstance(summary, (list, set)):
        summary = "\n".join(line.strip() for line in summary)
    elif not isinstance(summary, str):
        summary = str(summary) if summary else ""
    summary = summary.strip()
    summary_cleaned = summary.casefold()
    if summary and all(null_word not in summary_cleaned for null_word in ["null", "none", "empty"]):
        entity.content = summary


def _parse_attributes(entity: Entity, extracted_entity_info: Dict[str, Any]):
    """Process extracted attributes"""
    attributes = extracted_entity_info.get("attributes")
    if isinstance(attributes, str):
        attributes = parse_json(attributes)
    elif isinstance(attributes, (list, set)):
        try:
            attributes = dict(attributes)
        except Exception:
            """Ignore exception"""
    if not isinstance(attributes, dict):
        attributes = {}  # Cannot parse, don't risk saving bad data to database
    if attributes:
        entity.attributes = attributes
