# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
from typing import List, Tuple

from openjiuwen.core.common.logging import logger


_REASON_PROMPT = """
# Task Description:
You are given an input question and a set of known facts:
Question: {query}
Facts: {triples}

Your reply must follow the required format:
1. If the provided facts contain the answer to the question, your should reply as follows:
Answerable: Yes
Answer: ...

2. If not, you should explain why and reply as follows:
Answerable: No
Why: ...

# Your reply: """

_PATTERN = re.compile(
    r"^Answerable:\s*(?P<answerable>Yes|No).*(^Answer:(?P<answer>.*)|^Why:(?P<why>.*))",
    flags=re.MULTILINE | re.IGNORECASE | re.DOTALL,
)


def get_reason_prompt(query: str, triples: List[str]):
    """ """
    if triples is None:
        raise ValueError(f"{get_reason_prompt.__name__} expects non-empty triples!")
    return _REASON_PROMPT.format(query=query, triples=triples)


def postproc_reason(completion: str) -> Tuple[bool, str]:
    """ """
    tmp_match = _PATTERN.search(completion)
    if not tmp_match:
        logger.warning("%s: no matching pattern from completion=%r", postproc_reason.__name__, completion)
        return False, ""

    answerable = tmp_match.group("answerable")
    if answerable is None:
        logger.warning("%s: failed to match `Answerable` from completion=%r", postproc_reason.__name__, completion)
        return False, ""

    is_answerable = answerable.lower() == "yes"

    if is_answerable:
        answer = tmp_match.group("answer") or ""
        answer = answer.strip()
        return is_answerable, answer

    why = tmp_match.group("why") or ""
    why = why.strip()
    return is_answerable, why
