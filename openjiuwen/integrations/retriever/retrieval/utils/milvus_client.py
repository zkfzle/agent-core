# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Singleton Milvus client manager."""

import atexit
import threading
from typing import Optional

from pymilvus import MilvusClient

from openjiuwen.core.common.logging import logger


class MilvusClientManager:
    """Thread-safe singleton for managing a single MilvusClient instance."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._client: Optional[MilvusClient] = None
        self._uri: Optional[str] = None
        self._ref_count = 0

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._client = None
                    cls._instance._ref_count = 0
                    cls._instance._uri = None
        return cls._instance

    def get_client(
        self,
        uri: str,
        token: str = None,
        timeout: float = 60.0,
    ) -> MilvusClient:
        """
        Get or create a MilvusClient. Reuses existing client.
        """
        with self._lock:
            # Ensure client exists and matches URI
            if self._client is None or self._uri != uri:
                if self._client is not None:
                    logger.info(
                        f"Closing old client at: {self._uri=} since URI changed to {uri=}"
                    )
                    try:
                        self._client.close()
                    except Exception as e:
                        logger.warning(
                            f"Issue with closing old client at: {self._uri=}!"
                        )
                self._client = MilvusClient(uri=uri, token=token, timeout=timeout)
                self._uri = uri
                self._ref_count = 0

            self._ref_count += 1
            logger.debug(f"Re-using MilvusClient (refs={self._ref_count})")
            return self._client

    def release(self) -> None:
        """Decrement ref count. Close when count reaches 0."""
        with self._lock:
            if self._client is None:
                return

            self._ref_count -= 1
            logger.debug(f"Released MilvusClient (refs={self._ref_count})")

            if self._ref_count <= 0:
                logger.info("Closing MilvusClient")
                try:
                    self._client.close()
                except Exception as e:
                    logger.warning(f"Error closing MilvusClient: {e}")
                self._client = None
                self._uri = None
                self._ref_count = 0

    def close(self) -> None:
        """Force close the client."""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    """Should ignore the error"""
                    pass
                self._client = None
                self._uri = None
                self._ref_count = 0

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._client is not None


# Global singleton instance
milvus_manager = MilvusClientManager()


@atexit.register
def _cleanup_client():
    """Automatically close client when program exits."""
    logger.debug("Cleaning up MilvusClient on exit...")
    milvus_manager.close()
