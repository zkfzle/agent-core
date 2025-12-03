# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
from typing import List, Union

from .utils import triples2str

_REWRITE_PROMPT = """

# Task Description:
You will be presented with an input question and a set of known facts.
These facts might be insufficient for answering the question for some reason.
Your task is to analyze the question given the provided facts and determine what else information is needed for the next step.

# Example:
Question: What region of the state where Guy Shepherdson was born, contains SMA Negeri 68?
Facts: ("Guy Shepherdson", "born in", "Jakarta")
Reason: The provided facts only indicate that Guy Shepherdson was born in Jakarta, but they do not provide any information
about the region of the state that contains SMA Negeri 68.
Next Question: What region of Jakarta contains SMA Negeri 68?

# Your Task:
Question: {query}
Facts: {triples}
Reason: {reason}
Next Question: """

_PATTERN = re.compile(r"(?<=^Next Question:).*", flags=re.IGNORECASE | re.MULTILINE)


def get_rewrite_prompt(query: str, triples: Union[List[str], None], reason: str) -> str:
    triples_str = triples2str(triples) if triples is not None else ""
    return _REWRITE_PROMPT.format(query=query, triples=triples_str, reason=reason)


def postproc_rewrite(completion: str) -> str:
    """
    Extract new question from the given `completion`.
    """

    tmp_match = _PATTERN.search(completion)
    if not tmp_match:
        return completion.strip()

    return tmp_match.group().strip()
