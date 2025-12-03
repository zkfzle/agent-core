# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any

import requests

from openjiuwen.core.common.logging import logger

from .base import EmbedModel



class APIEmbedModel(EmbedModel):
    """
    通用 HTTP embedding 客户端：
    - payload: {"model": <model_name>, "input": <text or list>}（可附加 kwargs）
    - headers: 默认 application/json，可选 Authorization: Bearer <api_key>
    - 返回支持以下格式之一：
        {"embedding": [...]}
        {"embeddings": [...]}
        {"data": [{"embedding": [...]}, ...]}
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        extra_headers: dict | None = None,
        max_batch_size: int = 1,
    ):
        self.api_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_batch_size = max_batch_size
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            self._headers.update(extra_headers)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        if not text.strip():
            raise ValueError("Empty text provided for embedding")
        return self._get_embeddings(text, **kwargs)[0]

    def embed_docs(
        self,
        texts: list[str],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("Empty texts list provided")
        non_empty = [t for t in texts if t.strip()]
        if len(non_empty) != len(texts):
            raise ValueError(f"{len(texts) - len(non_empty)} chunks are empty while embedding")
        if not non_empty:
            raise ValueError("All texts are empty after filtering")
        bsz = batch_size or self.max_batch_size or 1
        all_embeddings: list[list[float]] = []
        for i in range(0, len(non_empty), bsz):
            j = i + bsz
            batch = non_empty[i:j]
            all_embeddings.extend(self._get_embeddings(batch, **kwargs))
        return all_embeddings

    def embed_queries(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            raise ValueError("Empty texts list provided")
        return self.embed_docs(texts, **kwargs)

    def _get_embeddings(self, text: str | list[str], **kwargs) -> list[list[float]]:
        payload = {"model": self.model_name, "input": text, **kwargs}
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.api_url, json=payload, headers=self._headers, timeout=self.timeout)
                resp.raise_for_status()
                result = resp.json()
                if "embedding" in result:
                    emb = result["embedding"]
                elif "embeddings" in result:
                    emb = result["embeddings"]
                elif "data" in result and isinstance(result["data"], list):
                    emb = []
                    for item in result["data"]:
                        if "embedding" in item:
                            emb.append(item["embedding"])
                    if not emb:
                        raise ValueError(f"No embeddings field found in data items: {result}")
                else:
                    raise ValueError(f"No embeddings in response: {result}")
                return emb
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to get embedding after {self.max_retries} attempts") from e
                logger.warning("Embedding request failed (attempt %s/%s): %s", attempt + 1, self.max_retries, e)
        raise RuntimeError("Unreachable code in _get_embeddings")