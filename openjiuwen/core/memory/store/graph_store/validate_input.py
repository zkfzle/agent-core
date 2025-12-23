# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

__all__ = ["validate_add_memory_input"]

from typing import Dict, Optional, Tuple

from openjiuwen.core.memory.config.graph.config import EpisodeType


def validate_add_memory_input(
    user_id_max_length: int,
    src_type: EpisodeType,
    user_id: str,
    content_fmt_kwargs: Optional[Dict] = None,
) -> Tuple[str, str]:
    """Preprocess episode content & retrieve relevant history episodes"""

    # Validate content_fmt_kwargs
    if content_fmt_kwargs is None:
        content_fmt_kwargs = {}
    else:
        if not (content_fmt_kwargs and isinstance(content_fmt_kwargs, dict)):
            raise ValueError("When supplied, content_fmt_kwargs must be of type Dict[str, str] and not empty")
    if not all(isinstance(k, str) and isinstance(v, str) and k and v for k, v in content_fmt_kwargs.items()):
        raise ValueError("content_fmt_kwargs must have non-empty keys and values of string type")

    # Data Preparation
    if not isinstance(src_type, EpisodeType):
        raise ValueError("src_type must be an EpisodeType Enum")
    if not (isinstance(user_id, str) and len(user_id) <= user_id_max_length):
        raise ValueError(f"user_id must be a string of length <= {user_id_max_length} (preferably UUID4)")
