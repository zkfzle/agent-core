# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any

import requests

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.retrieval.embed_models import EmbedModel


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

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        if not text.strip():
            raise ValueError("Empty text provided for embedding")

        return self._get_ollama_embedding(text, **kwargs)[0]

    def embed_docs(
        self,
        texts: list[str],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("Empty texts list provided")

        # Filter out empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        if len(non_empty_texts) != len(texts):
            raise ValueError(f"{len(texts) - len(non_empty_texts)} chunks are empty while embedding")

        if not non_empty_texts:
            raise ValueError("All texts are empty after filtering")

        # Process in batches if batch_size is specified
        if batch_size is not None and batch_size > 0:
            all_embeddings = []
            for i in range(0, len(non_empty_texts), batch_size):
                j = i + batch_size
                batch_texts = non_empty_texts[i:j]
                batch_embeddings = self._get_ollama_embedding(batch_texts, **kwargs)
                all_embeddings.extend(batch_embeddings)
            embeddings = all_embeddings
        else:
            embeddings = self._get_ollama_embedding(non_empty_texts, **kwargs)

        return embeddings

    def embed_queries(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            raise ValueError("Empty texts list provided")

        return self._get_ollama_embedding(texts, **kwargs)

    def _get_ollama_embedding(self, text: str | list[str], **kwargs) -> list[list[float]]:
        """Get ollama embedding"""
        if not text:
            raise ValueError("Empty text or list provided for embedding")

        payload = {"model": self.model_name, "input": text, "truncate": False, **kwargs}

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.embed_url, json=payload, headers=self._headers, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                if "embeddings" not in result:
                    raise ValueError(f"No embeddings in response: {result}")

                return result["embeddings"]

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to get embedding after {self.max_retries} attempts") from e
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")

        raise RuntimeError("This should never be reached")