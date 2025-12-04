# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABCMeta, abstractmethod

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from transformers import PreTrainedTokenizerBase

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHAR_CHUNK_SIZE = 200
DEFAULT_CHAR_CHUNK_OVERLAP = 40


class TextSplitter(metaclass=ABCMeta):
    @abstractmethod
    def split(self, text: TextNode) -> list[TextNode]:
        pass


class CharSplitter(TextSplitter):
    """基于字符长度的简单分段器，不依赖 tokenizer。"""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        super().__init__()
        size = chunk_size or DEFAULT_CHAR_CHUNK_SIZE
        overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHAR_CHUNK_OVERLAP
        # 限制范围，避免 step 变 0 或负数
        overlap = max(0, min(overlap, size - 1))
        self.chunk_size = max(1, size)
        self.chunk_overlap = overlap

    def split(self, doc: TextNode) -> list[TextNode]:
        text = doc.text or ""
        # 保持元数据、排除字段，以便后续索引/删除
        meta = doc.metadata or {}
        excluded_embed_metadata_keys = doc.excluded_embed_metadata_keys or list(meta.keys())
        excluded_llm_metadata_keys = doc.excluded_llm_metadata_keys or list(meta.keys())

        res: list[TextNode] = []
        step = self.chunk_size - self.chunk_overlap if self.chunk_size > self.chunk_overlap else self.chunk_size
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            res.append(
                TextNode(
                    text=text[start:end],
                    metadata=meta,
                    excluded_embed_metadata_keys=excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=excluded_llm_metadata_keys,
                    start_char_idx=start,
                    end_char_idx=end,
                )
            )
            start += step
        return res


class LlamaindexSplitter(TextSplitter):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        splitter_config: dict | None = None,
    ) -> None:
        """Wrapper of llamaindex's splitter.

        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer.
            chunk_size (int | None, optional): Chunk size to split documents into passages. Defaults to None.
                Note: this is based on tokens produced by the tokenizer of embedding model.
                If None, set to the maximum sequence length of the embedding model.
            chunk_overlap (int | None, optional): Window size for passage overlap. Defaults to None.
                If None, set to `chunk_size // 5`.
            splitter_config (dict, optional): Other arguments to SentenceSplitter. Defaults to None.

        """
        super().__init__()
        self._tokenizer = tokenizer

        if not isinstance(splitter_config, dict):
            splitter_config = {
                "paragraph_separator": "\n",
            }

        if self._tokenizer:
            tokenizer_fn = self._tokenizer.tokenize
            if chunk_size is None:
                chunk_size = self._tokenizer.max_len_single_sentence
            else:
                chunk_size = min(chunk_size, self._tokenizer.max_len_single_sentence)
        else:
            tokenizer_fn = None  # use default tiktoken tokenizer if None
            if chunk_size is None:
                chunk_size = DEFAULT_CHUNK_SIZE

        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap or chunk_size // 5,
            tokenizer=tokenizer_fn,
            **splitter_config,
        )

    def split(self, doc: TextNode) -> list[TextNode]:
        # Note: we don't want to consider the length of metadata for chunking
        if not doc.excluded_embed_metadata_keys:
            doc.excluded_embed_metadata_keys = list(doc.metadata.keys())

        if not doc.excluded_llm_metadata_keys:
            doc.excluded_llm_metadata_keys = list(doc.metadata.keys())

        return self._splitter.get_nodes_from_documents([doc])
