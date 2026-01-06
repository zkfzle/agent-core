# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import json
import os
import threading
from typing import Any, Dict, List, Optional, Self, Union

import numpy as np
import requests
from pydantic import BaseModel, Field

from .api_requests import sync_request_with_retry

QueryLike = Union[str, List[str]]


class EmbeddingConfig(BaseModel):
    """Embedding Service Config"""

    model_name: str
    api_url: str
    api_key: str
    timeout: Union[int, float] = Field(default=60.0, gt=0)
    dim: int = Field(alias="embedding_dim", default=1024)

    model_config = {"validate_assignment": True}

    @classmethod
    def default_config(cls) -> Self:
        """Get default config based on environment variables JIUWEN_GRAPH_MEM_EMBED_*"""
        component = "EMBED"
        variable_prefix = f"JIUWEN_GRAPH_MEM_{component}_"
        instance_kwargs = dict()

        for env_var_name, attr_name in zip(["URL", "KEY", "MODEL"], ["api_url", "api_key", "model_name"]):
            env_var = variable_prefix + env_var_name
            val = os.environ.get(env_var, None)
            if val is None:
                raise ValueError(f"Default config for {component} incorrect: {env_var} is not set.")
            instance_kwargs[attr_name] = val
        instance = cls(**instance_kwargs)

        env_var = variable_prefix + "CONFIG"
        val = os.environ.get(env_var, None)
        if val is not None:
            try:
                config_dict = json.loads(val)
                if not isinstance(config_dict, dict):
                    raise ValueError("Config is not supplied as a JSON object!")
            except json.JSONDecodeError as e:
                raise ValueError(f"Default config for {component} incorrect: {env_var} is not valid json.") from e
            for k, v in config_dict.items():
                if hasattr(instance, k):
                    setattr(instance, k, v)
                else:
                    raise ValueError(f"Default config for {component} incorrect: {env_var} has invalid key {k}.")
        return instance


class EmbeddingService:
    """A Barebone Embedding Class for OpenAI-compatible API Service"""

    def __init__(self, config: Optional[EmbeddingConfig] = None, **kwargs):
        if config is None:
            config = EmbeddingConfig.default_config()
        self.model_name = config.model_name
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.embedding_dim: int = config.dim
        self.timeout = config.timeout

    def query_embedding(self, query: str) -> List[float]:
        """Query embedding service"""
        payload = {"model": self.model_name, "prompt": query}
        response = requests.post(
            self.api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout
        )
        return response.json().get("embedding", [])


class VarDimEmbeddingService(EmbeddingService):
    """EmbeddingService Implementation for OpenAI-compatible Service with variable dimensions"""

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        client: Any = None,
        projection_seed: int = 42,
        matryoshka: bool = True,
    ):
        super().__init__(config)
        self.client = client or requests
        self.projection_seed = projection_seed
        self.matryoshka = matryoshka
        self.projection_matrix: Dict[int, np.ndarray] = dict()
        self.lock = threading.Lock()

    @staticmethod
    def parse_output(response: Dict):
        """Parse output for OpenAI standard embedding response"""
        embeddings = [
            (obj.get("index", 0), obj["embedding"]) for obj in response.get("data", []) if obj["object"] == "embedding"
        ]
        embeddings.sort(key=lambda x: x[0])
        return [emb[1] for emb in embeddings]

    def query_embedding(
        self, query: QueryLike, client: Any = None, max_retry: int = 5, retry_wait: float = 0.1, **kwargs
    ) -> List[Union[float, List[float]]]:
        """Query embedding service, support bulk-embedding.

        Args:
            query (QueryLike): str or List[str], queries.
            client (Any, optional): request client, recommend httpx. Defaults to None.
            max_retry (int, optional): maximum number of retries. Defaults to 5.
            retry_wait (float, optional): base waiting time for retries. Defaults to 0.1.

        Returns:
            List[Union[float, List[float]]]: return List[float] if query is str, List[List[float]] if bulk-embedding.
        """
        if client is None:
            client = self.client

        params = self._assemble_payload(query, **kwargs)
        headers = self._assemble_headers()
        response = sync_request_with_retry(
            client=client,
            max_retry=max_retry,
            retry_wait=retry_wait,
            task="Embedding",
            timeout=self.timeout,
            url=self.api_url,
            json=params,
            headers=headers,
        )
        embeddings = self.parse_output(response) if response else []

        # Dimension reduction if required
        if any(len(emb) > self.embedding_dim for emb in embeddings):
            returned_dim = len(embeddings[0])
            if self.matryoshka:  # Matryoshka Embedding: simply truncate
                embeddings = np.asarray([vec[: self.embedding_dim] for vec in embeddings], dtype=float)
                embeddings /= embeddings.sum(axis=1, keepdims=True)
                embeddings = embeddings.tolist()
            else:  # Random Projection
                with self.lock:
                    proj = self.projection_matrix.get(returned_dim)
                    if proj is None:
                        rng = np.random.default_rng(self.projection_seed)
                        proj = rng.normal(0, 1 / np.sqrt(self.embedding_dim), size=(returned_dim, self.embedding_dim))
                        self.projection_matrix[returned_dim] = proj
                embeddings = [(np.asarray(emb, dtype=float) @ proj).tolist() for emb in embeddings]
        return embeddings[0] if isinstance(query, str) else embeddings

    def _assemble_payload(self, query: QueryLike, **kwargs):
        payload = {
            "model": self.model_name,
            "input": query,
            "dimensions": self.embedding_dim,
            "encoding_format": "float",
            **kwargs,
        }
        return payload

    def _assemble_headers(self) -> Dict:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
