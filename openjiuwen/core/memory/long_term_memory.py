# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from datetime import datetime
from enum import Enum
from typing import Any, Tuple
from pydantic import BaseModel, Field

from openjiuwen.core.common.utils.singleton import Singleton
from openjiuwen.core.foundation.llm import BaseMessage
from openjiuwen.core.memory.config.config import MemoryEngineConfig, MemoryScopeConfig
from openjiuwen.core.memory.store import BaseKVStore, BaseSemanticStore, BaseDbStore


class MemoryType(Enum):
    USER_PROFILE = "USER_PROFILE"


class MemInfo(BaseModel):
    mem_id: str = Field(default="", description="memory id")
    content: str = Field(default="", description="memory content")
    type: MemoryType = Field(default=MemoryType.USER_PROFILE, description="memory type")


class MemResult(BaseModel):
    mem_info: MemInfo = Field(default=None, description="memory information")
    score: float = Field(default=0.0, description="memory score of relevance")


class LongTermMemory(metaclass=Singleton):
    """
    Abstract base class for memory engine.

    Defines the core interface for memory storage and retrieval operations.
    Provides unified memory management functionality including conversation memory,
    user variables, semantic search, and persistence.

    Concrete implementations should handle memory operations across multiple storage
    backends (KV store, semantic store, database store).
    """

    DEFAULT_VALUE: str = "__default__"

    def __init__(self):
        """
        Initialize the memory engine
        """
        pass

    def register_store(self, kv_store: BaseKVStore, semantic_store: BaseSemanticStore,
                 db_store: BaseDbStore):
        """
        Register store instance.

        Args:
            kv_store: Key-value store for fast structured data access
            semantic_store: Semantic storage for vector-based similarity search
            db_store: Database store for persistent data storage
        """
        pass

    def set_config(self, config: MemoryEngineConfig):
        """
        Set configuration.

        Args:
            config: memory engine configuration parameters
        """
        pass

    def set_scope_config(self, scope_id: str, config: MemoryScopeConfig):
        """
        Set memory configuration for a specified scope.

        Allows different groups to have customized memory retention policies,
        storage limits, and behavior settings.

        Args:
            scope_id: Unique identifier for the scope
            config: Group-specific memory configuration
        """
        pass

    async def add_messages(
            self,
            messages: list[BaseMessage],
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE,
            session_id: str = DEFAULT_VALUE,
            timestamp: datetime | None = None,
            gen_mem: bool = True,
            gen_mem_with_history_msg_num: int = 3
    ):
        """
        Add messages to memory storage.

        Args:
            messages: List of message objects to store
            user_id: Unique identifier for the user
            scope_id: Unique identifier for the scope
            session_id: Optional session identifier for grouping related messages
            timestamp: When the messages were created
            gen_mem: Whether to trigger memory generation.
            gen_mem_with_history_msg_num: History message num to use in memory generation.d
        """
        pass

    async def get_recent_messages(
            self,
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE,
            session_id: str = DEFAULT_VALUE,
            num: int = 10
    ) -> list[BaseMessage]:
        """
        Get recent messages.

        Args:
            user_id: Unique identifier for the user
            scope_id: Unique identifier for the scope
            session_id: Optional session identifier for grouping related messages
            num: message num

        Returns:
            Message list in order of writing.
        """
        pass

    async def get_message_by_id(self, msg_id: str) -> Tuple[BaseMessage, datetime]:
        """
        Retrieve a specific message by its unique identifier.

        Args:
            msg_id: Unique identifier of the message to retrieve

        Returns:
            Tuple of (message object, creation timestamp)
        """
        pass

    async def delete_mem_by_id(
            self,
            mem_id: str,
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE
    ):
        """
        Delete a specific memory by ID.

        Args:
            user_id: Unique identifier for the user
            scope_id: Unique identifier for the scope
            mem_id: Unique identifier of the memory to delete
        """
        pass

    async def delete_mem_by_user_id(
            self,
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE
    ):
        """
        Delete all type memories for a user with scope id.

        Useful for implementing "forget me" functionality or cleaning up user data.

        Args:
            user_id: User identifier whose memories should be deleted
            scope_id: Unique identifier for the scope
        """
        pass

    async def update_mem_by_id(
            self,
            mem_id: str,
            memory: str,
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE
            ):
        """
        Update the content of an existing memory entry.

        Args:
            mem_id: Unique identifier of the memory to update
            memory: New content for the memory
            user_id: Unique identifier for the user
            scope_id: Unique identifier for the scope
        """
        pass

    async def get_user_variables(
        self,
        names: list[str] | str | None = None,
        user_id: str = DEFAULT_VALUE,
        scope_id: str = DEFAULT_VALUE
    ) -> dict[str, str]:
        """
        Retrieve a specific user variable by name.

        User variables store persistent user preferences, settings, or context
        that should be maintained across conversations.

        Args:
            names: Name of the variable(s) to get, return all variables when is None
            user_id: User identifier
            scope_id: Unique identifier for the scope

        Returns:
            Mapping of variable names to values
        """
        pass

    async def search_user_mem(
            self,
            query: str,
            num: int,
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE,
            threshold: float = 0.3
    ) -> list[MemResult]:
        """
        Search user memories using semantic similarity.

        Performs vector-based similarity search to find relevant memories
        based on the query content rather than exact keyword matching.

        Args:
            query: Search query text
            num: Maximum number of results to return
            user_id: User identifier to search within
            scope_id: Unique identifier for the scope
            threshold: Similarity threshold (0.0-1.0) for filtering results

        Returns:
            List of memory result
        """
        pass

    async def user_mem_total_num(
            self,
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE,
    ) -> int:
        """
        return total number of user memory
        """
        pass

    async def get_user_mem_by_page(
            self,
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE,
            page_size: int = 10,
            page_idx: int = 0
    ) -> list[MemInfo]:
        """
        List user memories with pagination support.

        Retrieves memories in chronological order, suitable for displaying
        conversation history or memory browsing interfaces.

        Args:
            user_id: User identifier to search within
            scope_id: Unique identifier for the scope
            page_size: Number of memories per page
            page_idx: Page index

        Returns:
            List of memory information
        """
        pass

    async def update_user_variables(
            self,
            variables: dict[str, str],
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE
    ):
        """
        Update user variables.

        Args:
            variables: variable name to value pairs
            user_id: User identifier to search within
            scope_id: Unique identifier for the scope
        """
        pass

    async def delete_user_variables(
            self,
            names: list[str],
            user_id: str = DEFAULT_VALUE,
            scope_id: str = DEFAULT_VALUE
    ):
        """
        Delete user variables.

        Args:
            names: Name of the variables to delete
            user_id: User identifier to search within
            scope_id: Unique identifier for the scope
        """
        pass
