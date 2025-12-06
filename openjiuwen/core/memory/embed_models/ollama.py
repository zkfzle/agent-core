#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Any

import aiohttp
import requests

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.embed_models import EmbedModel


class OllamaEmbedModel(EmbedModel):
    """Ollama embedding model implementation."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        hf_tokenizer_name: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        extra_headers: dict | None = None,
    ):
        """
        Initialize Ollama embedder.

        Args:
            model_name: Name of the embedding model in Ollama
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.embed_url = f"{self.base_url}/api/embed"
        self._headers = extra_headers or {}

        # Initialize tokenizer if provided
        if hf_tokenizer_name:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        else:
            self._tokenizer = None

        # Test connection and model availability
        self._verify_model_availability()

    @property
    def tokenizer(self):
        return self._tokenizer

    def _verify_model_availability(self):
        """Verify that Ollama is running and the model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if the model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            if self.model_name not in model_names:
                raise ValueError(
                    f"Model '{self.model_name}' not found in available models: {model_names}. "
                    f"Make sure to pull the model first: ollama pull {self.model_name}"
                )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Could not connect to Ollama at {self.base_url}. Is Ollama running?") from e

    async def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        if not text.strip():
            raise ValueError("Empty text provided for embedding")
        embs = await self._get_ollama_embedding(text, **kwargs)
        return embs[0]

    async def _get_ollama_embedding(self, text: str | list[str], **kwargs) -> list[list[float]]:
        """Get ollama embedding"""
        if not text:
            raise ValueError("Empty text or list provided for embedding")

        payload = {"model": self.model_name, "input": text, "truncate": False, **kwargs}

        for attempt in range(self.max_retries):
            try:
                response = await self._post_async(payload)
                response.raise_for_status()
                result = await response.json()
                if "embeddings" not in result:
                    raise ValueError(f"No embeddings in response: {result}")

                return result["embeddings"]

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to get embedding after {self.max_retries} attempts") from e
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")

        raise RuntimeError("This should never be reached")

    async def _post_async(self, payload):
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.post(self.embed_url, json=payload, timeout=self.timeout) as resp:
                return resp
