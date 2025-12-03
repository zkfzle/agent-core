# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
from typing import List, Union

from llama_index.core.schema import TextNode

from openjiuwen.core.common.logging import logger

from .utils import chunks2str, triples2str


_READ_PROMPT = """
Your task is to find facts that help answer an input question.

You should present these facts as knowledge triples, which are structured as ("subject", "predicate", "object").
Example:
Question: When was Neville A. Stanton's employer founded?
Facts: ("Neville A. Stanton", "employer", "University of Southampton"), ("University of Southampton", "founded in", "1862")

Now you are given some documents:
{docs}

{instruction}
Note: if the information you are given is insufficient, output only the relevant facts you can find.

Question: {query}
Facts: {facts}
"""


_PATTERN = re.compile(r"\((?:\"|').*?(?:\"|')\)")


def get_read_prompt(query: str, passages: List[TextNode], triples: Union[List[str], None]) -> str:

    facts = triples2str(triples) if triples is not None else ""
    instruction = (
        (
            "Based on these documents and preliminary facts provided below, "
            "find additional supporting fact(s) that may help answer the following question."
        )
        if facts
        else "Based on these documents, find supporting fact(s) that may help answer the following question."
    )
    return _READ_PROMPT.format(docs=chunks2str(passages=passages), query=query, instruction=instruction, facts=facts)


def postproc_read(completion: str) -> List[tuple[str, ...]]:
    """
    Extract triples from completion and ensure they are unique.
    """
    triples_set = set()
    for x in _PATTERN.findall(completion):
        try:
            triple = eval(x)
            if not isinstance(triple, tuple):
                continue
            triple = tuple(map(str, triple))  # converted each element in triple to string -- just in case
            triples_set.add(triple)
        except Exception as e:
            logger.warning("%s: fail to extract triple; error: %r", postproc_read.__name__, e)
            continue

    triples = list(triples_set)
    if not triples:
        logger.warning("%s: no triples exacted from %r", postproc_read.__name__, completion)

    return triples
