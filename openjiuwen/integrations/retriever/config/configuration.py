# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Config loader (SDK-friendly, explicit config only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from openjiuwen.core.common.logging import logger


def _require(cfg: dict, key: str) -> Any:
    if key not in cfg or cfg.get(key) is None:
        raise ValueError(f"Missing required config field: {key}")
    return cfg.get(key)


def _to_bool(val: Any, default: Any = None) -> Any:
    """Lenient bool parser with default fallback."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


class GraphRAGConfig:
    """Configuration holder."""

    def __init__(self, yaml_data: dict):
        self.raw = yaml_data or {}
        cfg = self.raw

        # 基础设施（必填）
        self.es_url = _require(cfg, "es_url")
        self.embed_api_type = _require(cfg, "embed_api_type")  # e.g., custom_api/ollama
        self.embed_api_base = _require(cfg, "embed_api_base")
        self.embed_api_key = cfg.get("embed_api_key")
        self.embed_model_name = _require(cfg, "embed_model_name")
        self.embed_timeout = int(_require(cfg, "embed_timeout"))
        self.embed_max_retries = int(_require(cfg, "embed_max_retries"))
        self.embed_max_batch_size = int(cfg.get("embed_max_batch_size") or 1)

        self.llm_api_type = _require(cfg, "llm_api_type")
        self.llm_api_key = _require(cfg, "llm_api_key")
        self.llm_api_base = _require(cfg, "llm_api_base")
        self.llm_model_name = _require(cfg, "llm_model_name")
        self.llm_temperature = float(_require(cfg, "llm_temperature"))
        self.llm_max_retries = int(_require(cfg, "llm_max_retries"))
        self.llm_timeout = int(_require(cfg, "llm_timeout"))

        # 以下是运行时选项的默认值（调用参数优先覆盖，建议调用接口时显示传入）
        self.index_type = cfg.get("index_type")
        # use_graph_index 不再要求调用方/配置提供，仅在内部按需要动态写入
        self.use_graph_index = None
        self.use_grag_agent = _to_bool(cfg.get("use_grag_agent"), default=None)
        self.use_sync = _to_bool(cfg.get("use_sync"), default=None)

        self.concurrent_llm_requests_limit = (
            int(cfg.get("concurrent_llm_requests_limit"))
            if cfg.get("concurrent_llm_requests_limit") is not None
            else None
        )
        self.concurrent_indexing_requests = (
            int(cfg.get("concurrent_indexing_requests"))
            if cfg.get("concurrent_indexing_requests") is not None
            else None
        )

        self.hide_local_urls = _to_bool(cfg.get("hide_local_urls"), default=None)
        self.enable_text_preprocessing = _to_bool(cfg.get("enable_text_preprocessing"), default=None)
        self.remove_urls = _to_bool(cfg.get("remove_urls"), default=None)
        self.remove_emails = _to_bool(cfg.get("remove_emails"), default=None)
        self.normalize_special_characters = _to_bool(cfg.get("normalize_special_characters"), default=None)
        self.normalize_whitespace = _to_bool(cfg.get("normalize_whitespace"), default=None)
        self.preserve_single_newline = _to_bool(cfg.get("preserve_single_newline"), default=None)
        self.chunk_unit = str(cfg.get("chunk_unit")).lower() if cfg.get("chunk_unit") else None

        self.batch_size = int(cfg.get("batch_size")) if cfg.get("batch_size") is not None else None
        self.chunk_size = int(cfg.get("chunk_size")) if cfg.get("chunk_size") is not None else None
        self.chunk_overlap = int(cfg.get("chunk_overlap")) if cfg.get("chunk_overlap") is not None else None

        self.data_dir = cfg.get("data_dir")
        self.input_data_file = cfg.get("input_data_file")

        self.chunk_es_index = cfg.get("chunk_es_index")
        self.triple_es_index = cfg.get("triple_es_index")

        self.precomputed_chunks = _to_bool(cfg.get("precomputed_chunks"), default=False)

    # --------- helpers ---------
    @staticmethod
    def _root_dir() -> Path:
        """RAG package root."""
        return Path(__file__).resolve().parents[2]

    def get_data_path(self, filename: str) -> Path:
        """Return data path resolved against RAG root if relative."""
        p = Path(filename)
        if p.is_absolute():
            return p
        base = self._root_dir()
        return (base / self.data_dir / filename).resolve()

    def get_full_data_path(self, filename: str) -> Path:
        return self.get_data_path(filename)

    def print_config(self) -> None:
        fields = [
            "es_url",
            "embed_api_type",
            "embed_api_base",
            "embed_model_name",
            "index_type",
            "use_graph_index",
            "chunk_es_index",
            "triple_es_index",
            "batch_size",
            "chunk_size",
            "chunk_overlap",
            "data_dir",
            "input_data_file",
            "precomputed_chunks",
            "llm_api_type",
            "llm_model_name",
            "llm_timeout",
        ]
        logger.info("Configuration:")
        for f in fields:
            logger.info("  %s: %s", f, getattr(self, f, None))


def load_config(path: str | None = None) -> GraphRAGConfig:
    """Load config from explicit path only; missing path or fields will raise."""
    if not path:
        raise ValueError("Config path is required for load_config (no default search).")
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {cfg_path}: {e}") from e

    return GraphRAGConfig(yaml_data)


# No default instance; callers must load explicitly.
CONFIG = None

__all__ = ["GraphRAGConfig", "load_config", "CONFIG"]
