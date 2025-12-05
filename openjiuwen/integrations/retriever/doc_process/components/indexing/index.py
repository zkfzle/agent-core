# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import uuid
from typing import Any, Dict, Optional

from llama_index.core.schema import Document, TextNode
from pymilvus import MilvusException

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
from openjiuwen.integrations.retriever.doc_process.components.indexing.milvus_wrapper import BaseMilvusIndexer
from openjiuwen.integrations.retriever.doc_process.components.parsing.local_file_parser import parse_file
from openjiuwen.integrations.retriever.retrieval.utils import load_jsonl, load_jsonl_as_iterator
from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import milvus_manager


class MilvusTextIndexer(BaseMilvusIndexer):
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


async def delete_text_entries(doc_id: str, config_obj=None):
    """
    Delete text entries from the text index for a given document ID
    This function handles the deletion process for text chunks

    Args:
        doc_id: The document ID to delete
    """
    cfg = config_obj or CONFIG

    client = milvus_manager.get_client(
        uri=cfg.milvus_uri,
        token=getattr(cfg, "milvus_token", None),
    )

    try:
        logger.info(f"🔍 Deleting from text index...")

        filter_expr = f'document_id == "{doc_id}"'
        try:
            result = client.delete(
                collection_name=cfg.chunk_es_index,
                filter=filter_expr,
            )
            # MilvusClient.delete returns dict with delete count or int
            if isinstance(result, dict):
                return result.get("delete_count", 0)
            return int(result) if result else 0
        except MilvusException as exc:
            logger.error("Failed to delete document %s from Milvus: %s", doc_id, exc)
            return 0

    except Exception as e:
        logger.error("删除文本索引时出错: %s", e)
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
    batch_size = cfg.batch_size or 128

    logger.info(
        "开始构建文本索引: chunk_size=%s overlap=%s unit=%s index_type=%s",
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
                "已配置为预生成分块，自动将 chunk_size 调整为 tokenizer 上限: %s",
                chunk_size,
            )
        else:
            chunk_size = 100000
            logger.warning(
                "已配置为预生成分块；将 chunk_size 设为 100000、overlap=0；请确认与 embedding/tokenizer 限制兼容。"
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
        logger.info("Configuring text preprocessing pipeline...")
        preprocessing_pipeline = PreprocessingPipeline()

        if cfg.remove_urls or cfg.remove_emails:
            url_email_remover = URLEmailRemover(remove_urls=cfg.remove_urls, remove_emails=cfg.remove_emails)
            preprocessing_pipeline.add_preprocessor(url_email_remover)
            logger.info(
                "   URL/Email remover added (URLs: %s, Emails: %s)",
                cfg.remove_urls,
                cfg.remove_emails,
            )

        if cfg.normalize_special_characters:
            special_char_normalizer = SpecialCharacterNormalizer()
            preprocessing_pipeline.add_preprocessor(special_char_normalizer)
            logger.info("   Special character normalizer added")

        if cfg.normalize_whitespace:
            whitespace_normalizer = WhitespaceNormalizer(preserve_single_newline=cfg.preserve_single_newline)
            preprocessing_pipeline.add_preprocessor(whitespace_normalizer)
            logger.info(
                "   Whitespace normalizer added (preserve newlines: %s)",
                cfg.preserve_single_newline,
            )

        logger.info(f"   Total preprocessors: {len(preprocessing_pipeline)}")
    else:
        logger.info("未开启文本预处理")

    # 初始化索引器
    text_indexer = MilvusTextIndexer(
        collection_name=cfg.chunk_es_index,
        uri=cfg.milvus_uri,
        token=cfg.milvus_token,
        embed_model=embed_model,
        splitter=splitter,
        preprocessing_pipeline=preprocessing_pipeline,
    )
    # vector-only 时关闭倒排（仅向量）；bm25/hybrid 默认保留文本索引
    index_type = (cfg.index_type or "hybrid").lower()
    if index_type == "vector":
        setattr(text_indexer, "vector_only", True)
    else:
        setattr(text_indexer, "vector_only", False)

    logger.info("读取输入数据...")
    if from_file is not None:
        if cfg.precomputed_chunks:
            dataset = process_precomputed_chunks(from_file)
            logger.info(
                "✅ 读取预生成分块 %d 条，文件: %s (ID: %s)",
                len(dataset),
                from_file["filepath"],
                from_file["id"],
            )
        else:
            dataset = await parse_file(from_file["filepath"], from_file["filename"], from_file["id"])
            logger.info(
                "✅ 读取原始文档 %d 条，文件: %s (ID: %s)",
                len(dataset),
                from_file["filepath"],
                from_file["id"],
            )
    else:
        # 获取数据文件路径
        data_path = cfg.get_data_path(cfg.input_data_file)
        dataset = load_jsonl(data_path)

    logger.info("开始写入文本索引...")
    await asyncio.to_thread(
        text_indexer.build_index,
        dataset,
        batch_size=batch_size,
        debug=False,
    )

    logger.info("文本索引构建完成。")
