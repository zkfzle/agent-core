# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Tuple

from pymilvus import CollectionSchema, DataType, Function, FunctionType, MilvusClient
from pymilvus.milvus_client import IndexParams

from openjiuwen.core.memory.config.graph.database_config import DBEmbeddingConfig, DBStorageSize

# Use ICU analyzer for best multi-lingual support
icu_filter = ["asciifolding", "lowercase", {"type": "stemmer", "language": "english"}, "removepunct"]
icu_analyzer = {
    "tokenizer": "icu",
    "filter": icu_filter,
}
icu_analyzer_with_stopwords = icu_analyzer.copy()
icu_analyzer_with_stopwords["filter"] = icu_filter + [{"type": "stop", "stop_words": ["of", "to", "_english_"]}]
# White space analyzer for exact match
exact_match_analyzer = {"tokenizer": "whitespace"}


def generate_schema_and_index(
    milvus_client: MilvusClient,
    collection: str,
    storage_config: DBStorageSize,
    embed_config: DBEmbeddingConfig,
) -> Tuple[CollectionSchema, IndexParams]:
    """Generate the schema and index for milvus db"""
    schema = milvus_client.create_schema(enable_dynamic_field=False)
    index_params = milvus_client.prepare_index_params()

    # Common fields
    schema.add_field("uuid", DataType.VARCHAR, max_length=storage_config.uuid, auto_id=False, is_primary=True)
    schema.add_field("created_at", DataType.INT64)
    schema.add_field("user_id", DataType.VARCHAR, max_length=storage_config.user_id)
    schema.add_field(
        "type",
        DataType.VARCHAR,
        max_length=storage_config.type,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params=exact_match_analyzer,
    )
    schema.add_field("language", DataType.VARCHAR, max_length=storage_config.language)
    schema.add_field("metadata", DataType.JSON)

    # Add collection-specific fields
    match collection:
        case "entities":
            schema.add_field(
                "name",
                DataType.VARCHAR,
                max_length=storage_config.name,
                enable_analyzer=True,
                enable_match=True,
                analyzer_params=icu_analyzer,
            )
            schema.add_field("name_embedding", DataType.FLOAT_VECTOR, dim=embed_config.dim)
            schema.add_field("attributes", DataType.JSON)
            index_params.add_index(
                field_name="name_embedding",
                index_name="semantic_embedding_name",
                index_type=embed_config.index_type,
                metric_type=embed_config.metric_type,
                **embed_config.extra_configs,
            )
            schema.add_field(
                "relations",
                DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_length=storage_config.uuid,
                max_capacity=storage_config.relations,
            )
            schema.add_field(
                "episodes",
                DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_length=storage_config.uuid,
                max_capacity=storage_config.episodes,
            )

        case "relations":
            schema.add_field("valid_since", DataType.INT64)
            schema.add_field("valid_until", DataType.INT64)
            schema.add_field("offset_since", DataType.INT8)
            schema.add_field("offset_until", DataType.INT8)
            schema.add_field("name", DataType.VARCHAR, max_length=storage_config.name)
            schema.add_field("lhs", DataType.VARCHAR, max_length=storage_config.uuid)
            schema.add_field("rhs", DataType.VARCHAR, max_length=storage_config.uuid)

        case "episodes":
            schema.add_field("valid_since", DataType.INT64)
            schema.add_field(
                "entities",
                DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_length=storage_config.uuid,
                max_capacity=storage_config.entities,
            )

    # Content field (summary, fact, etc.)
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=storage_config.content,
        enable_analyzer=True,
        analyzer_params=embed_config.bm25_analyzer_settings or icu_analyzer_with_stopwords,
    )
    if embed_config.dim:
        schema.add_field("content_embedding", DataType.FLOAT_VECTOR, dim=embed_config.dim)
        index_params.add_index(
            field_name="content_embedding",
            index_name="semantic_embedding_content",
            index_type=embed_config.index_type,
            metric_type=embed_config.metric_type,
            **embed_config.extra_configs,
        )
    schema.add_field("content_bm25", DataType.SPARSE_FLOAT_VECTOR)
    bm25_func = Function(
        name="bm25_func",
        input_field_names=["content"],
        output_field_names=["content_bm25"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_func)
    index_params.add_index(
        field_name="content_bm25",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params=embed_config.bm25_config.model_dump(),
    )

    return schema, index_params
