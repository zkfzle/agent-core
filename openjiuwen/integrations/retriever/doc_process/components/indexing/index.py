# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import uuid
from typing import Any, Dict, Optional

from llama_index.core.schema import Document, TextNode

from openjiuwen.core.common.logging import logger
from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.doc_process.components.chunking.text_preprocessor import (
    PreprocessingPipeline,
    SpecialCharacterNormalizer,
    URLEmailRemover,
    WhitespaceNormalizer,
)
from openjiuwen.integrations.retriever.doc_process.components.chunking.text_splitter import (
    CharSplitter,
    LlamaindexSplitter,
    TextSplitter,
)
from openjiuwen.integrations.retriever.doc_process.components.indexing.base_indexer import BaseIndexer
from openjiuwen.integrations.retriever.doc_process.components.parsing.local_file_parser import parse_file
from openjiuwen.integrations.retriever.retrieval.utils import load_jsonl, load_jsonl_as_iterator



class TextIndexer(BaseIndexer):
    def __init__(self, *args, preprocessing_pipeline: Optional[PreprocessingPipeline] = None, **kwargs):
        """
        Initialize TextIndexer with optional text preprocessing pipeline.

        Args:
            preprocessing_pipeline: Optional PreprocessingPipeline to apply to text before indexing
            *args, **kwargs: Arguments passed to parent BaseIndexer
        """
        super().__init__(*args, **kwargs)
        self.preprocessing_pipeline = preprocessing_pipeline

    def preprocess(self, doc: dict, splitter: TextSplitter) -> list[TextNode]:
        # global doc id here
        metadata = {
            "title": doc["title"],
            "paragraph_id": doc["paragraph_id"],
            "file_id": doc["doc_id"],  # Add file_id to metadata for deletion queries
        }

        # Apply text preprocessing if pipeline is configured
        paragraph_text = doc["paragraph_text"]
        if self.preprocessing_pipeline:
            paragraph_text = self.preprocessing_pipeline.process(paragraph_text)

        doc = Document(
            text=paragraph_text,
            metadata=metadata,
            excluded_embed_metadata_keys=list(metadata.keys()),
            excluded_llm_metadata_keys=list(metadata.keys()),
        )

        if hasattr(self.embed_model, "tokenizer_threading_lock"):
            # HuggingFace tokenizers are not thread-safe (Runtime Error: Already borrowed)
            with self.embed_model.tokenizer_threading_lock:
                return splitter.split(doc)
        return splitter.split(doc)

    def get_metadata_mappings(self, **kwargs: Any) -> dict:
        analyzer = kwargs.get("analyzer")
        return {
            "properties": {
                "title": ({"type": "text", "analyzer": analyzer} if analyzer else {"type": "text"}),
                "paragraph_id": {"type": "keyword"},
                "file_id": {"type": "keyword"},  # Add file_id mapping for deletion queries
            }
        }


async def delete_text_entries(doc_id: str):
    """
    Delete text entries from the text index for a given document ID
    This function handles the deletion process for text chunks in the ES index

    Args:
        doc_id: The document ID to delete
    """
    try:
        # Delete from text index
        logger.info(f"🔍 Deleting from text index...")
        text_indexer = TextIndexer(
            es_index=CONFIG.chunk_es_index,
            es_url=CONFIG.es_url,
        )
        text_deleted_count = await text_indexer.async_delete_nodes(doc_id)

        # Log results
        logger.info(f"✅ Text chunk deletion completed successfully:")
        logger.info(f"   Text index: {text_deleted_count} chunks deleted")

    except Exception as e:
        logger.error(f"❌ Error during text chunk deletion: {e}")
        raise e


def process_precomputed_chunks(from_file: Dict[str, str]) -> list[Dict[str, str]]:
    if (not from_file["filepath"].endswith(".jsonl")) and (not from_file["filepath"].endswith(".txt")):
        raise ValueError("Precomputed chunks file must be a .jsonl or .txt file.")

    dataset = []
    for item in load_jsonl_as_iterator(from_file["filepath"]):
        doc_data = {
            "title": item.get("title", ""),
            "doc_id": from_file["id"],
            "paragraph_id": str(uuid.uuid4()),
            "paragraph_text": item.get("content", ""),
        }
        dataset.append(doc_data)

    return dataset


async def index(
    from_file: Optional[Dict[str, str]] = None,
    config_obj=None,
    embed_model=None,
):
    cfg = config_obj or CONFIG
    chunk_size = cfg.chunk_size
    chunk_overlap = cfg.chunk_overlap
    chunk_unit = getattr(cfg, "chunk_unit", None) or "token"

    logger.info(
        "Building text index: chunk_size=%s overlap=%s unit=%s index_type=%s",
        chunk_size,
        chunk_overlap,
        chunk_unit,
        cfg.index_type,
    )

    # 根据索引类型决定是否计算向量；splitter 仍需 tokenizer
    tokenizer_model = embed_model or getattr(cfg, "embed_model_instance", None)
    embed_model = tokenizer_model
    if tokenizer_model is None and (cfg.index_type in ("vector", "hybrid")):
        raise ValueError("embed_model is required when index_type is vector/hybrid (SDK 不再自动创建 embedding)")

    # index_type: vector/bm25/hybrid
    index_type = (cfg.index_type or "hybrid").lower()
    if index_type in ("vector", "hybrid"):
        embed_model = embed_model or tokenizer_model
    # bm25 情况不计算向量，但仍使用 tokenizer 做分块

    # Prevent splitter if we're reading in precomputed chunks
    if cfg.precomputed_chunks:
        chunk_overlap = 0
        tokenizer = getattr(embed_model, "tokenizer", None)
        if tokenizer and hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length < float("inf"):
            chunk_size = tokenizer.model_max_length
            logger.info(
                "Input files are configured to be read as precomputed chunks; "
                f"setting chunk size to tokenizer max length: {chunk_size}"
            )
        else:
            chunk_size = 100000
            logger.warning(
                "Input files are configured to be read as precomputed chunks; setting chunk size to 100000, overlap 0; "
                "ensure that the chunks fit your embedding model/tokenizer."
            )

    # 初始化分块器（即使 bm25 也需要 tokenizer）
    if chunk_unit == "char":
        splitter: TextSplitter = CharSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        splitter = LlamaindexSplitter(
            tokenizer=tokenizer_model.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # 初始化文本预处理管道
    preprocessing_pipeline = None
    if cfg.enable_text_preprocessing and not cfg.precomputed_chunks:
        logger.info("📝 Configuring text preprocessing pipeline...")
        preprocessing_pipeline = PreprocessingPipeline()

        if cfg.remove_urls or cfg.remove_emails:
            url_email_remover = URLEmailRemover(remove_urls=cfg.remove_urls, remove_emails=cfg.remove_emails)
            preprocessing_pipeline.add_preprocessor(url_email_remover)
            logger.info(f"   ✓ URL/Email remover added (URLs: {cfg.remove_urls}, Emails: {cfg.remove_emails})")

        if cfg.normalize_special_characters:
            special_char_normalizer = SpecialCharacterNormalizer()
            preprocessing_pipeline.add_preprocessor(special_char_normalizer)
            logger.info(f"   ✓ Special character normalizer added")

        if cfg.normalize_whitespace:
            whitespace_normalizer = WhitespaceNormalizer(preserve_single_newline=cfg.preserve_single_newline)
            preprocessing_pipeline.add_preprocessor(whitespace_normalizer)
            logger.info(f"   ✓ Whitespace normalizer added (preserve newlines: {cfg.preserve_single_newline})")

        logger.info(f"   Total preprocessors: {len(preprocessing_pipeline)}")
    else:
        logger.info("📝 Text preprocessing is disabled")

    # 初始化索引器
    es = TextIndexer(
        es_index=cfg.chunk_es_index,
        es_url=cfg.es_url,
        embed_model=embed_model,
        splitter=splitter,
        preprocessing_pipeline=preprocessing_pipeline,
    )
    # vector-only 时关闭倒排（仅向量）；bm25/hybrid 默认保留文本索引
    index_type = (cfg.index_type or "hybrid").lower()
    if index_type == "vector":
        setattr(es, "vector_only", True)

    logger.info("Reading data...")
    if from_file is not None:
        if cfg.precomputed_chunks:
            dataset = process_precomputed_chunks(from_file)
            logger.info(
                "✅ Loaded %d precomputed chunks from file: %s (ID: %s)",
                len(dataset),
                from_file["filepath"],
                from_file["id"],
            )
        else:
            dataset = await parse_file(from_file["filepath"], from_file["filename"], from_file["id"])
            logger.info(
                "✅ Loaded %d documents from file: %s (ID: %s)", len(dataset), from_file["filepath"], from_file["id"]
            )
    else:
        # 获取数据文件路径
        data_path = cfg.get_data_path(cfg.input_data_file)
        dataset = load_jsonl(data_path)

    logger.info("Building text index...")
    await es.build_index(
        dataset,
        batch_size=cfg.batch_size,
        debug=False,
    )

    logger.info("Text index build completed.")