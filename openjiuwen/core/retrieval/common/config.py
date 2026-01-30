# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Configuration Classes

All configuration classes are unified in this file.
"""

import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr

from openjiuwen.core.foundation.llm.schema.mode_info import BaseModelInfo
from openjiuwen.core.retrieval.common.document import _raise_validation_error_with_info


class KnowledgeBaseConfig(BaseModel):
    """Knowledge base configuration"""

    kb_id: str = Field(..., description="Knowledge base identifier")
    index_type: Literal["hybrid", "bm25", "vector"] = Field(default="hybrid", description="Index type")
    use_graph: bool = Field(default=False, description="Whether to use graph index")
    chunk_size: int = Field(default=512, description="Chunk size")
    chunk_overlap: int = Field(default=50, description="Chunk overlap")


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""

    top_k: int = Field(default=5, description="Number of results to return")
    score_threshold: Optional[float] = Field(default=None, description="Score threshold")
    use_graph: Optional[bool] = Field(
        default=None, description="Whether to use graph retrieval (None uses default config)"
    )
    agentic: bool = Field(default=False, description="Whether to use Agentic retrieval")
    graph_expansion: bool = Field(default=False, description="Whether to enable graph expansion")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter conditions")


class IndexConfig(BaseModel):
    """Index configuration"""

    index_name: str = Field(..., description="Index name")
    index_type: Literal["hybrid", "bm25", "vector"] = Field(default="hybrid", description="Index type")


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""

    database_name: str = Field(default="", pattern=r"^[A-Za-z0-9_]*$", description="Database name")
    collection_name: str = Field(..., description="Collection name")
    distance_metric: Literal["cosine", "euclidean", "dot"] = Field(default="cosine", description="Distance metric")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""

    model_name: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API Base URL")
    api_key: Optional[str] = Field(None, description="API Key")


class RerankerConfig(BaseModelInfo):
    """Reranker model configuration"""

    streaming: Literal[False] = Field(default=False, init=False, repr=False)
    timeout: float = Field(default=10, gt=0)

    _yes_id: int = PrivateAttr(default=None)
    _no_id: int = PrivateAttr(default=None)
    _yes_text: str = PrivateAttr(default=None)
    _no_text: str = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)

    @property
    def initialized(self) -> bool:
        """Check if RerankerConfig is initialized for Chat Completions Reranking"""
        return self._initialized

    def set_tokens_for_reranker(self, yes_id: int, no_id: int, yes_text: str = "yes", no_text: str = "no"):
        """Set token config for reranking"""
        if not all(isinstance(token_id, int) for token_id in [yes_id, no_id]):
            _raise_validation_error_with_info(
                "token_id_must_be_int",
                '"yes_id" and "no_id" must be int',
                dict(yes_id=yes_id, no_id=no_id),
            )
        if not all(token_text and isinstance(token_text, str) for token_text in [yes_text, no_text]):
            _raise_validation_error_with_info(
                "token_text_must_be_str",
                '"yes_text" and "no_text" must be non-empty str',
                dict(yes_text=yes_text, no_text=no_text),
            )
        self._yes_id, self._no_id = yes_id, no_id
        self._yes_text, self._no_text = yes_text, no_text
        self._initialized = True

    def load_tokens_from_huggingface(self, hf_repo: str, filename: str = "tokenizer.json", token: Any = None, **kwargs):
        """Load token config for reranking from huggingface"""
        hf_repo = hf_repo.rstrip("/")
        try:
            import huggingface_hub

            with tempfile.TemporaryDirectory() as tmp_dir:
                huggingface_hub.hf_hub_download(
                    repo_id=hf_repo, filename=filename, local_dir=tmp_dir, token=token, **kwargs
                )
                tmp_file = Path(tmp_dir) / filename
                tokenizer_content = tmp_file.read_text(encoding="utf-8")
                token_ids, token_texts = [None] * 2, [None] * 2
                for i, yes_no_token in enumerate([r'"(yes)"', r'"(no)"']):
                    token_match = re.search(yes_no_token + r"\s*:\s*([0-9]+)", tokenizer_content)
                    if token_match is None:
                        token_match = re.search(
                            yes_no_token + r"\s*:\s*([0-9]+)", tokenizer_content, flags=re.IGNORECASE
                        )
                    if token_match is None:
                        yes_no_token = yes_no_token[2:-2]
                        _raise_validation_error_with_info(
                            "unable_to_find_token",
                            f"Failed to find {yes_no_token} in {filename}",
                            dict(filename=filename, missing_token=yes_no_token),
                        )
                    token_ids[i] = int(token_match.group(2))
                    token_texts[i] = token_match.group(1)
                self._yes_id, self._no_id = token_ids
                self._yes_text, self._no_text = token_texts
                self._initialized = True

        except Exception as e:
            _raise_validation_error_with_info(
                "unable_to_download_from_huggingface",
                f'Failed to download file huggingface: "{hf_repo}/{filename}", error: {e}',
                dict(hf_repo=hf_repo, filename=filename, token=token),
            )
