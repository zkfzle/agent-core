#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
from typing import Any

import asyncio
import requests
import aiohttp
from llama_index.embeddings.dashscope import DashScopeEmbedding

from openjiuwen.core.common.logging import logger

from .base import EmbedModel


class APIEmbedModel:
    _qwen_model_names = {"text-embedding-v2", "text-embedding-v3"}

    def __new__(
        cls,
        model_name: str,
        base_url: str,
        timeout: int,
        max_retries: int,
        extra_headers: dict | None = None,
        api_key: str | None = None,
        *args,
        **kwargs,
    ):
        if model_name not in cls._qwen_model_names:
            return SiliconflowEmbedClient(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                extra_headers=extra_headers or {},
                *args,
                **kwargs,
            )
        return QwenEmbedClient(model_name=model_name, api_key=api_key, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        # the actual instance is created in __new__; this is only for type compatibility checking.
        pass


class APIEmbedClient(EmbedModel):
    """
    general HTTP embedded client：
    - payload: {"model": <model_name>, "input": <text or list>}（attached kwargs）
    - headers: default application/json，optional Authorization: Bearer <api_key>
    - returns support one of the following formats：
        {"embedding": [...]}
        {"embeddings": [...]}
        {"data": [{"embedding": [...]}, ...]}
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_batch_size: int = 1,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.max_batch_size = max_batch_size

    async def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Empty text provided for embedding")
        embs = await self._get_embeddings(text, **kwargs)
        return embs[0]

    async def _get_embeddings(self, text: str | list[str], **kwargs) -> list[list[float]]:
        payload = {"model": self.model_name, "input": text, **kwargs}
        for attempt in range(self.max_retries):
            try:
                resp = await self._post_async(payload)
                resp.raise_for_status()
                result = await resp.json()
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
                logger.warning(
                    "Embedding request failed (attempt %s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
        raise RuntimeError("Unreachable code in _get_embeddings")

    async def _post_async(self, payload):
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.post(self.api_url, json=payload, timeout=self.timeout) as resp:
                return resp


class SiliconflowEmbedClient(APIEmbedClient):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout: int = 60,
        max_retries: int = 3,
        api_key: str | None = None,
        extra_headers: dict | None = None,
        max_batch_size: int = 8,
    ):
        super().__init__(model_name=model_name, api_key=api_key, max_batch_size=max_batch_size)
        self.api_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = {"Content-Type": "application/json"}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            self._headers.update(extra_headers)

    async def _get_embeddings(self, text: str | list[str], **kwargs) -> list[list[float]]:
        payload = {"model": self.model_name, "input": text, **kwargs}
        for attempt in range(self.max_retries):
            try:
                resp = await self._post_async(payload)
                resp.raise_for_status()
                result = await resp.json()
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
                logger.warning(
                    "Embedding request failed (attempt %s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
        raise RuntimeError("Unreachable code in _get_embeddings")

    async def _post_async(self, payload):
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.post(self.api_url, json=payload, timeout=self.timeout) as resp:
                return resp


class QwenEmbedClient(APIEmbedClient):
    def __init__(self, model_name: str, max_batch_size: int = 8, api_key: str | None = None):
        super().__init__(model_name=model_name, api_key=api_key, max_batch_size=max_batch_size)
        os.environ["DASHSCOPE_API_KEY"] = self.api_key
        self.embedder = DashScopeEmbedding(model_name=self.model_name)

    async def _get_embeddings(self, text: str | list[str]) -> list[list[float]]:
        if isinstance(text, str):
            text = [text]
        returned_embed = await asyncio.to_thread(self.embedder.get_text_embedding_batch, text)
        return returned_embed
