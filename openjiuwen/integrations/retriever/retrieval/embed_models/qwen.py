# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any

import requests

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.retrieval.embed_models import EmbedModel


class QwenEmbedModel(EmbedModel):
    """
    针对 Qwen/DashScope 的 embedding 客户端。
    - 兼容官方 embedding 接口返回格式：
      { "output": { "embeddings": [{"embedding": [...]}] }, ... }
    - 也兼容 OpenAI 兼容模式返回的 data[*].embedding / embeddings / embedding 字段。
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        extra_headers: dict | None = None,
        max_batch_size: int = 8,
    ):
        if not base_url:
            raise ValueError("QwenEmbedModel requires base_url")
        self.api_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_batch_size = max_batch_size

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._headers = headers

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Empty text provided for embedding")
        return self._fetch_embeddings(text, **kwargs)[0]

    def embed_docs(
        self,
        texts: list[str],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("Empty texts list provided")

        non_empty = [t for t in texts if t and t.strip()]
        if len(non_empty) != len(texts):
            raise ValueError(f"{len(texts) - len(non_empty)} chunks are empty while embedding")
        if not non_empty:
            raise ValueError("All texts are empty after filtering")

        bsz = batch_size or self.max_batch_size or 1
        all_embeddings: list[list[float]] = []
        for i in range(0, len(non_empty), bsz):
            j = i + bsz
            batch = non_empty[i:j]
            all_embeddings.extend(self._fetch_embeddings(batch, **kwargs))
        return all_embeddings

    def embed_queries(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            raise ValueError("Empty texts list provided")
        return self.embed_docs(texts, **kwargs)

    def _fetch_embeddings(self, text: str | list[str], **kwargs: Any) -> list[list[float]]:
        payload = {"model": self.model_name, "input": text, **kwargs}
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.api_url, json=payload, headers=self._headers, timeout=self.timeout)
                resp.raise_for_status()
                return self._parse_embeddings(resp.json())
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to get Qwen embedding after {self.max_retries} attempts") from e
                logger.warning("Qwen embedding request failed (attempt %s/%s): %s", attempt + 1, self.max_retries, e)
        raise RuntimeError("Unreachable code in _fetch_embeddings")

    @staticmethod
    def _parse_embeddings(result: Any) -> list[list[float]]:
        """
        支持 DashScope 原生响应和 OpenAI 兼容响应：
        - {"output": {"embeddings": [{"embedding": [...]}]}}
        - {"data": [{"embedding": [...]}]}
        - {"embeddings": [...]}
        - {"embedding": [...]}
        """
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected embedding response type: {type(result)}")

        output = result.get("output")
        if isinstance(output, dict):
            embeddings = output.get("embeddings")
            if isinstance(embeddings, list):
                parsed = [item["embedding"] for item in embeddings if isinstance(item, dict) and "embedding" in item]
                if parsed:
                    return parsed
                raise ValueError(f"No embedding field found in output.embeddings items: {result}")

        if "data" in result and isinstance(result["data"], list):
            parsed = [item["embedding"] for item in result["data"] if isinstance(item, dict) and "embedding" in item]
            if parsed:
                return parsed
            raise ValueError(f"No embeddings field found in data items: {result}")

        if "embeddings" in result:
            embeddings = result["embeddings"]
            if embeddings and isinstance(embeddings, list):
                # Support both list[list[float]] and list[float] (single embedding)
                if embeddings and isinstance(embeddings[0], (list, tuple)):
                    return list(embeddings)
                return [embeddings]

        if "embedding" in result:
            return [result["embedding"]]

        raise ValueError(f"No embeddings in response: {result}")