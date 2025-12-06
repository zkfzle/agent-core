#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Any

import requests
import aiohttp

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.embed_models.base import EmbedModel


class QwenEmbedModel(EmbedModel):
    """
    an embedding client for Qwen/DashScope.
    - compatible with official embedding interface return format：
      { "output": { "embeddings": [{"embedding": [...]}] }, ... }
    - it is also compatible with OpenAI return data[*].embedding / embeddings / embedding field.
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

    async def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Empty text provided for embedding")
        embs = await self._fetch_embeddings(text, **kwargs)
        return embs[0]

    async def _fetch_embeddings(self, text: str | list[str], **kwargs: Any) -> list[list[float]]:
        payload = {"model": self.model_name, "input": text, **kwargs}
        for attempt in range(self.max_retries):
            try:
                resp = await self._post_async(payload)
                resp.raise_for_status()
                json_data = await resp.json()
                return await self._parse_embeddings(json_data)
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to get Qwen embedding after {self.max_retries} attempts") from e
                logger.warning("Qwen embedding request failed (attempt %s/%s): %s", attempt + 1, self.max_retries, e)
        raise RuntimeError("Unreachable code in _fetch_embeddings")

    @staticmethod
    async def _parse_embeddings(result: Any) -> list[list[float]]:
        """
        support DashScope native response and OpenAI compatible response：
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

    async def _post_async(self, payload):
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.post(self.api_url, json=payload, timeout=self.timeout) as resp:
                return resp
