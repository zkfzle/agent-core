# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# GraphRAG component code is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from collections.abc import Iterator
from typing import Any, Optional

from pymilvus import MilvusClient


def iter_index(
    client: MilvusClient,
    collection_name: str,
    batch_size: int = 256,
    filter_expr: str | None = None,
    **kwargs: Any,
) -> Iterator[list[dict]]:
    """Iterate over all documents in a Milvus collection using MilvusClient.

    Args:
        client (MilvusClient): MilvusClient instance.
        collection_name (str): Name of the collection to query.
        batch_size (int, optional): Maximum number of documents to return at a time. Defaults to 256.
        filter_expr (str | None, optional): Boolean expression to filter results. Defaults to None (all documents).
        kwargs: Additional args to pass to `query`, including:

        - output_fields (list[str] | None, optional): Fields to return. If None, returns all fields except those in\
            exclude_fields. Defaults to None.
        - exclude_fields (list[str] | None, optional): Fields to exclude from output. Defaults to None (excludes\
            "embedding" by default).

    Yields:
        Iterator[list[dict]]: An iterator of batches of documents.

    Example:
        >>> from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import milvus_manager
        >>> client = milvus_manager.get_client(uri="http://localhost:19530")
        >>> for batch in iter_index(client, "my_collection", batch_size=100):
        ...     for doc in batch:
        ...         print(doc)
        >>> milvus_manager.release()
    """
    # Default exclude embedding field (similar to ES behavior)
    exclude_fields: Optional[list[str]] = kwargs.pop("exclude_fields", None)
    output_fields: Optional[list[str]] = kwargs.pop("output_fields", None)
    if exclude_fields is None:
        exclude_fields = ["embedding"]

    # Get all field names from schema if output_fields not specified
    if output_fields is None:
        # Get collection schema to determine fields
        schema = client.describe_collection(collection_name)
        all_fields = [field["name"] for field in schema.get("fields", [])]
        output_fields = [f for f in all_fields if f not in exclude_fields]
    else:
        # Remove excluded fields from output_fields
        output_fields = [f for f in output_fields if f not in exclude_fields]

    # Get primary key field name
    schema = client.describe_collection(collection_name)
    pk_field = None
    for field in schema.get("fields", []):
        if field.get("is_primary", False):
            pk_field = field["name"]
            break

    if pk_field and pk_field not in output_fields:
        output_fields.insert(0, pk_field)

    # Use offset-based pagination since MilvusClient doesn't have query_iterator
    offset = 0
    while True:
        results = client.query(
            collection_name=collection_name,
            filter=filter_expr if filter_expr else "",
            output_fields=output_fields,
            limit=batch_size,
            offset=offset,
            **kwargs,
        )

        if not results:
            break

        yield results

        # If we got fewer results than batch_size, we've reached the end
        if len(results) < batch_size:
            break

        offset += batch_size
