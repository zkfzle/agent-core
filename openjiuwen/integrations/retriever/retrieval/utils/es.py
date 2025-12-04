# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import functools
from collections.abc import Iterator
from typing import Any

import requests
from elasticsearch import Elasticsearch


def iter_index(
    client: Elasticsearch,
    index: str,
    batch_size: int = 256,
    source_excludes: str | list[str] = "embedding",
    query: dict | None = None,
    **kwargs: Any,
) -> Iterator[list[dict]]:
    """Iterate over all documents in an Elasticsearch index.

    Args:
        client (Elasticsearch): Elasticsearch client.
        index (str): Index name.
        batch_size (int, optional): Maximum number of documents to return at a time. Defaults to 256.
        source_excludes (str | list[str], optional): Fields to be excluded. Defaults to "embedding".
        kwargs: Additonal args to `Elasticsearch.search`.

    Yields:
        Iterator[list[dict]]: An iterator of batches of `hits`.

    """
    _kwargs = {
        "track_total_hits": False,
        "sort": ["_doc"],
    }
    _kwargs.update(kwargs)

    search = functools.partial(
        client.search,
        index=index,
        size=batch_size,
        query=query if query else {"match_all": {}},
        source_excludes=source_excludes,
        **_kwargs,
    )

    response = search()

    hits = response["hits"]["hits"]

    if hits:
        yield hits
    else:
        return

    last_sort = hits[-1]["sort"]
    while True:
        response = search(search_after=last_sort)
        hits = response["hits"]["hits"]

        if not hits:
            break

        last_sort = hits[-1]["sort"]
        yield hits


def iter_index_compat(
    es_url: str,
    index: str,
    batch_size: int = 256,
    params: dict | None = None,
) -> Iterator[list[dict]]:
    """Iterate over all documents in an Elasticsearch index.

    Note:
        This function removes the dependency on Elasticsearch client and is directly implemented via HTTP requests.
        It intends to be used when your local Elasticsearch client is incompatible with the Elasticsearch server.

    Args:
        es_url (str): Elasticsearch url.
        index (str): Index name.
        params (dict, optional): Additional parameters for Elasticsearch's search request. Defaults to None.

    Yields:
        Iterator[list[dict]]: An iterator of batches of `hits`.

    Raises:
        RuntimeError: Failure of search requests.

    """
    # http://<ip>:<port>/<index>/_search
    url = "/".join(x.strip("/") for x in (es_url, index, "_search"))

    data = {
        "query": {"match_all": {}},
        "size": batch_size,
        "track_total_hits": False,
        "sort": ["_doc"],
    }
    if params:
        data.update(params)

    response = requests.post(url=url, json=data)
    if response.status_code != 200:
        raise RuntimeError(f"{response.text}")

    hits = response.json()["hits"]["hits"]

    if hits:
        yield hits
    else:
        return

    last_sort = hits[-1]["sort"]
    while True:
        data["search_after"] = last_sort
        response = requests.post(url=url, json=data)
        if response.status_code != 200:
            raise RuntimeError(f"{response.text}")

        hits = response.json()["hits"]["hits"]

        if not hits:
            break

        last_sort = hits[-1]["sort"]
        yield hits


def custom_query_num_candidates(
    query_body: dict[str, Any],
    query: str | None = None,
    *,
    num_candidates=500,
) -> dict[str, Any]:
    if "knn" in query_body:
        query_body["knn"]["num_candidates"] = num_candidates

    return query_body
