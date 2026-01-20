# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from pydantic import Field
from typing import Any, Dict

from openjiuwen.core.foundation.llm import BaseMessage


class OffloadMessage(BaseMessage):
    """
    A message representing content that has been offloaded from the active context.

    This message type acts as a lightweight placeholder (hint) for content that
    was moved to external storage to manage token budget. It preserves essential
    metadata for retrieval while occupying minimal space in the LLM's context window.

    The actual content is stored externally and can be fully restored using the
    `offload_handle` with an appropriate reloader tool when needed.

    Attributes:
        offload_type: Storage backend type. Either "memory" for in-memory cache
            (e.g., Redis, LRU dict) or "filesystem" for persistent file storage.
        offload_handle: Unique identifier for retrieving the offloaded content.
            UUID string for memory storage, or absolute/relative file path for
            filesystem storage.
        metadata: Additional context about the offloaded content, such as
            original token count, timestamp, content type, or compression info.

        {'role': 'assistant', 'content': '[[HANDLE:abc123]]', 'offload_id': '/tmp/conv_001/turn_42.json'}
    """

    offload_type: str = ...
    offload_handle: str = ...
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Serialize the offload message to a dictionary format.

        Returns a compact representation suitable for API transmission or
        persistence, with the offload_handle exposed as 'offload_id' for
        compatibility with external reloading mechanisms.

        Returns:
            Dictionary containing role, content hint, and offload identifier.
        """
        result = {
            "role": self.role,
            "content": self.content,
            "offload_id": self.offload_handle
        }
        return result
