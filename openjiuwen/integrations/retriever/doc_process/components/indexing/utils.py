# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import itertools
from typing import Iterator

from elasticsearch import Elasticsearch

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.retrieval.utils import iter_index

empty_triples_count = 0



def prepare_triples(
    es: Elasticsearch, chunk2triples: dict[str, list[list[str]]], index_name: str, file_id: str = None
) -> Iterator[dict]:
    """Generate document dictionaries from Elasticsearch index and extracted triples (from API).
       This dictionary is used to then build the triplets index.

    Args:
        es (Elasticsearch): Elasticsearch client
        chunk2triples (dict[str, list[list[str]]]): Dictionary mapping chunks to extracted triples
        index_name (str): Name of the Elasticsearch index

    Yields:
        Iterator[dict]: Document dictionaries containing text and metadata
    """
    global empty_triples_count
    for item in itertools.chain.from_iterable(
        iter_index(
            client=es,
            index=index_name,
            batch_size=256,
            query={"term": {"metadata.file_id": file_id}} if file_id else None,
        )
    ):

        chunk_id = item["_id"]
        chunk = item["_source"]["content"]
        # Extract file_id from the text index metadata
        file_id = item["_source"].get("metadata", {}).get("file_id", "unknown")

        triples = chunk2triples.get(chunk, None)
        if triples is None:
            logger.warning("chunk=%r not found", chunk)
            continue

        if not triples:
            empty_triples_count += 1
            continue

        for triple in triples:
            if not isinstance(triple, list):
                raise TypeError(f"{type(triple)=}")

            if len(triple) == 0:
                continue

            triple = [str(x) for x in triple]

            yield {
                "text": " ".join(triple),
                "metadata": {
                    "chunk_id": chunk_id,
                    "triple": triple,
                    "file_id": file_id,  # Add file_id for deletion queries
                },
            }
        logger.debug("number of empty triples: %d", empty_triples_count)
