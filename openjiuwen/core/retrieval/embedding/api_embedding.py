# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
API 嵌入模型实现

通用 HTTP 嵌入客户端实现。
"""
from typing import Any, List, Optional
import asyncio

import requests

from openjiuwen.core.common.logging import logger
from openjiuwen.core.retrieval.embedding.base import Embedding
from openjiuwen.core.retrieval.common.config import EmbeddingConfig


class APIEmbedding(Embedding):
    """
    Universal HTTP embedding client:
    - payload: {"model": <model_name>, "input": <text or list>} (can attach kwargs)
    - headers: default application/json, optional Authorization: Bearer <api_key>
    - returns support one of the following formats:
        {"embedding": [...]}
        {"embeddings": [...]}
        {"data": [{"embedding": [...]}, ...]}
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        timeout: int = 60,
        max_retries: int = 3,
        extra_headers: Optional[dict] = None,
        max_batch_size: int = 8,
    ):
        self.config = config
        self.model_name = config.model_name
        self.api_key = config.api_key
        self.max_batch_size = max_batch_size
        self.api_url = config.base_url or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = {"Content-Type": "application/json"}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            self._headers.update(extra_headers)
        
        # Cache dimension
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """返回嵌入维度
        
        通过 embed_query 自适应获取维度，
        最多重试10次。
        """
        if self._dimension is not None:
            return self._dimension
        
        # 通过 embed_query 自适应获取维度，最多重试10次
        max_attempts = 10
        test_text = "test"
        
        for attempt in range(max_attempts):
            try:
                # 尝试获取或创建事件循环
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 如果事件循环正在运行，创建新的事件循环
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            embedding = loop.run_until_complete(self.embed_query(test_text))
                        finally:
                            loop.close()
                    else:
                        embedding = loop.run_until_complete(self.embed_query(test_text))
                except RuntimeError:
                    # 没有事件循环，创建新的
                    embedding = asyncio.run(self.embed_query(test_text))
                
                # 成功获取嵌入向量，提取维度并缓存
                self._dimension = len(embedding)
                logger.debug(f"Determined embedding dimension: {self._dimension} via embed_query")
                return self._dimension
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(
                        f"Failed to determine embedding dimension after {max_attempts} attempts: {e}"
                    )
                    raise RuntimeError(
                        f"Failed to determine embedding dimension after {max_attempts} attempts"
                    ) from e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} to determine dimension failed: {e}"
                )
        
        raise RuntimeError("Unreachable code in dimension property")

    async def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        if not text.strip():
            raise ValueError("Empty text provided for embedding")
        embeddings = await self._get_embeddings(text, **kwargs)
        return embeddings[0]

    async def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        if not texts:
            raise ValueError("Empty texts list provided")
        non_empty = [t for t in texts if t.strip()]
        if len(non_empty) != len(texts):
            raise ValueError(
                f"{len(texts) - len(non_empty)} chunks are empty while embedding"
            )
        if not non_empty:
            raise ValueError("All texts are empty after filtering")
        # Respect caller batch_size but never exceed configured max_batch_size
        bsz = batch_size or self.max_batch_size or 1
        if self.max_batch_size:
            bsz = min(bsz, self.max_batch_size)
        all_embeddings: List[List[float]] = []
        for i in range(0, len(non_empty), bsz):
            j = i + bsz
            batch = non_empty[i:j]
            all_embeddings.extend(await self._get_embeddings(batch, **kwargs))
        return all_embeddings

    async def _get_embeddings(
        self, text: str | List[str], **kwargs
    ) -> List[List[float]]:
        """获取嵌入向量"""
        import asyncio
        
        payload = {"model": self.model_name, "input": text, **kwargs}
        
        for attempt in range(self.max_retries):
            try:
                resp = await asyncio.to_thread(
                    requests.post,
                    self.api_url,
                    json=payload,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                result = resp.json()
                if "embedding" in result:
                    emb = result["embedding"]
                    if isinstance(emb[0], list):
                        embeddings = emb
                    else:
                        embeddings = [emb]
                elif "embeddings" in result:
                    embeddings = result["embeddings"]
                elif "data" in result and isinstance(result["data"], list):
                    embeddings = []
                    for item in result["data"]:
                        if "embedding" in item:
                            embeddings.append(item["embedding"])
                    if not embeddings:
                        raise ValueError(
                            f"No embeddings field found in data items: {result}"
                        )
                else:
                    raise ValueError(f"No embeddings in response: {result}")
                
                # 如果维度还未确定，从结果中获取并缓存
                if self._dimension is None and embeddings and embeddings[0]:
                    self._dimension = len(embeddings[0])
                    logger.debug(f"Determined embedding dimension: {self._dimension}")
                
                return embeddings
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Failed to get embedding after {self.max_retries} attempts"
                    ) from e
                logger.warning(
                    "Embedding request failed (attempt %s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
        raise RuntimeError("Unreachable code in _get_embeddings")
