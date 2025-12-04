# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import List

from llama_index.core.schema import TextNode


def triples2str(triples: List[tuple[str, ...]]) -> str:
    """
    Utility function for converting an input list of triples into a properly formatted string.

    Args:
        triples (List[tuple[str, ...]]): _description_

    Returns:
        str: _description_
    """
    formatted_triples = []
    for triple in triples:
        formatted_triples.append(f"({' '.join(triple)})")
    return "\n".join(formatted_triples)


def chunks2str(passages: List[TextNode]) -> str:
    """
    Utility function for formatting returned passages into a single string.

    Args:
        passages (List[TextNode]): _description_

    Returns:
        str: _description_
    """

    formatted_passages = []
    for chunk in passages:
        passage = chunk.text
        lines = passage.split("\n")
        if "title" in chunk.metadata and chunk.metadata["title"] != "":
            title = chunk.metadata["title"]
            formatted_entry = f"Wikipedia Title: {title}\n{passage}"
        elif len(lines) >= 2 and ("title" not in chunk.metadata or chunk.metadata["title"] == ""):
            title = lines[0]
            passage = "\n".join(lines[1:])
            formatted_entry = f"Wikipedia Title: {title}\n{passage}"
        else:
            formatted_entry = passage
        formatted_passages.append(formatted_entry)
    return "\n\n".join(formatted_passages)
