# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import os.path
import socket
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Self, Type, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from openjiuwen.core.memory.store.graph_store.api_services.embedding_service import (
    EmbeddingConfig,
    EmbeddingService,
    VarDimEmbeddingService,
)
from openjiuwen.core.memory.store.graph_store.local_storage import DEFAULT_GRAPH_STORAGE_DIR

from .database_config import DBEmbeddingConfig, DBStorageSize


class EpisodeType(Enum):
    """Type of possible sources (conversation/document/json)"""

    conversation: int = 0
    document: int = 1
    json: int = 2


class GraphConfig(BaseModel):
    """Configuration of Graph Memory Backend"""

    uri: Optional[str] = Field(default=None)
    name: str = Field(default="")
    user: str = Field(default="")
    password: str = Field(default="")
    token: str = Field(default="")
    backend: str = Field(default="milvus")
    timeout: Union[int, float] = Field(default=15.0, gt=0)
    extras: dict = Field(default_factory=dict)
    worker_threads: int = Field(default=30, ge=0)
    embed_dim: int = Field(default=-1, init=False)
    embed_batch_size: int = Field(default=10, ge=1)
    embedding_cls: Type[EmbeddingService] = Field(default=VarDimEmbeddingService)
    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig.default_config)
    db_storage_config: DBStorageSize = Field(default_factory=DBStorageSize)
    db_embed_config: DBEmbeddingConfig = Field(default_factory=DBEmbeddingConfig)
    wipe_at_startup: bool = Field(default=False)
    request_max_retry: int = Field(
        default=5, description="Max number of retries for sending embedding / chat completion requests"
    )
    request_retry_wait: float = Field(
        default=0.1, description="Wait between retries for sending embedding / chat completion requests (second) "
    )

    @field_validator("extras", mode="before")
    @classmethod
    def check_extras(cls, value: Union[Any, Dict]):
        """Check the extras field"""
        if isinstance(value, Mapping) and all(isinstance(k, str) for k in value.keys()):
            return value
        raise ValueError("Extras must be a dictionary with string keys.")

    @model_validator(mode="after")
    def check_validity(self):
        """Check if configuration is valid"""
        self.embed_dim = self.embedding_config.dim
        setattr(self.db_embed_config, "dim", self.embed_dim)
        if self.uri is None:
            self.uri = os.path.join(DEFAULT_GRAPH_STORAGE_DIR, "local_graph_db" + ".db")
        uri_is_file_path = "://" not in self.uri
        if uri_is_file_path:
            file_dir = os.path.dirname(self.uri)
            if file_dir:
                os.makedirs(file_dir, exist_ok=True)
        else:
            try:
                cleaned_uri = self.uri.split("//")[-1]
                with socket.create_connection(cleaned_uri.split(":"), timeout=self.timeout):
                    return self
            except Exception as e:
                raise ValueError(f"Graph DB config uri did not respond within {self.timeout} seconds.") from e
        if not issubclass(self.embedding_cls, VarDimEmbeddingService):
            self.embed_batch_size = 1  # base EmbeddingService doesn't implement support for batching
        return self


class LLMConfig(BaseModel):
    api_key: str = Field(default="", description="API Key")
    api_base: str = Field(default="", description="Base URL")
    model_name: str = Field(default="", alias="model", description="Model name")
    temperature: float = Field(default=0.95, description="Generation temperature")
    top_p: float = Field(default=0.1, description="Generation top-p")
    timeout: Union[int, float] = Field(default=60.0, description="API access timeout", gt=0)
    structured_output: bool = Field(default=True, description="Whether structured output is supported")
    extras: Dict[str, Any] = Field(default_factory=dict, description="Any other configs to pass in")

    @field_validator("model_name", mode="before")
    @classmethod
    def handle_model_name(cls, v, values):
        """Check the model field"""
        if not v and "model" in values.data:
            return values.data["model"]
        return v

    @classmethod
    def default_config(cls, is_reranker: bool = False) -> Optional[Self]:
        """Get default config based on environment variables JIUWEN_GRAPH_MEM_LLM_* or JIUWEN_GRAPH_MEM_RERANK_*"""
        component = "RERANK" if is_reranker else "LLM"
        variable_prefix = f"JIUWEN_GRAPH_MEM_{component}_"
        if is_reranker and os.environ.get(variable_prefix + "ENABLE", "false").strip().lower() not in ["true", "1"]:
            return None
        instance = cls()

        for env_var_name, attr_name in zip(["URL", "KEY", "MODEL"], ["api_base", "api_key", "model_name"]):
            env_var = variable_prefix + env_var_name
            val = os.environ.get(env_var, None)
            if val is None:
                raise ValueError(f"Default config for {component} incorrect: {env_var} is not set.")
            setattr(instance, attr_name, val)

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
                    instance.extras[k] = v
        if is_reranker:
            rerank_mode = os.environ.get("JIUWEN_GRAPH_MEM_RERANK_MODE", "standard")
            if rerank_mode not in ["standard", "aliyun"]:
                raise ValueError(f'Default config for {component} incorrect: {env_var} is not "standard" or "aliyun".')
            instance.extras["JIUWEN_GRAPH_MEM_RERANK_MODE"] = rerank_mode
        return instance

    class Config:
        populate_by_name = True
        validate_assignment = True
        extra = "forbid"
