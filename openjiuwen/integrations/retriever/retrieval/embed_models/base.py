# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from abc import ABCMeta, abstractmethod
from typing import Any


class EmbedModel(metaclass=ABCMeta):
    @abstractmethod
    def embed_docs(
        self,
        texts: list[str],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Embed documents."""
        pass

    @abstractmethod
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a single query."""
        pass

    def embed_queries(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Embed queries (default: map over embed_query)."""
        return [self.embed_query(x, **kwargs) for x in texts]

    def get_embedding_dimension(self) -> int:
        return len(self.embed_query("X"))
