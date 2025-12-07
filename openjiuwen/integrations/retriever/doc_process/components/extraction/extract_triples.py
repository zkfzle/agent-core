# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import json

import anyio

from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.doc_process.components.extraction.llm_openie import PROMPT as PROMPT_TEMPLATE
from openjiuwen.integrations.retriever.doc_process.components.extraction.llm_openie import LLMOpenIE
from openjiuwen.integrations.retriever.retrieval.llms.client import get_model_name
from openjiuwen.integrations.retriever.retrieval.utils.milvus import iter_index
from openjiuwen.integrations.retriever.retrieval.utils.milvus_client import milvus_manager


def _concurrent_limit(cfg=None) -> int | float:
    """Parse concurrent limit from config; fallback to 200 on invalid input."""
    val = None
    if cfg is not None:
        val = getattr(cfg, "concurrent_llm_requests_limit", None)
    val = val or 200
    try:
        return int(val)
    except Exception:
        return 200


async def process_chunk(chunk, llm_client, limiter, save_path=None, model_name: str = "gpt-4o-mini"):
    # Limit the number of concurrent LLM requests
    async with limiter:
        prompt = PROMPT_TEMPLATE.format(passage=chunk["content"], wiki_title=chunk["title"])
        messages = [{"role": "user", "content": prompt}]

        completion = await llm_client.ainvoke(model_name=model_name, messages=messages, temperature=0.0)
        _, triples_list = LLMOpenIE.match_entities_triples(completion.content)
        buffer = {chunk["content"]: [list(triple) for triple in triples_list]}

        if save_path is not None:
            with open(save_path, "a") as f:
                f.write(json.dumps(buffer, ensure_ascii=False) + "\n")

        return buffer


async def process_data(data, llm_client, limiter, save_path, model_name: str = "gpt-4o-mini") -> dict:
    tasks = [
        asyncio.create_task(process_chunk(chunk, llm_client, limiter, save_path, model_name=model_name))
        for chunk in data
    ]

    total = len(tasks)
    step = max(1, min(100, total))  # 每 100 个（或更小总量时每批次）打印一次
    chunk2triples = {}
    for idx, task in enumerate(asyncio.as_completed(tasks), start=1):
        res = await task
        chunk2triples.update(res)
        if idx % step == 0 or idx == total:
            logger.info("三元组抽取进度: %d/%d chunks", idx, total)
    return chunk2triples


def load_index(
    milvus_uri: str, milvus_token: str, chunk_es_index: str, chunk_file_path: str = None, file_id: str = None
) -> list[dict]:
    """Load chunks from Milvus collection using MilvusClient API."""
    client = milvus_manager.get_client(
        uri=milvus_uri,
        token=milvus_token,
    )

    # Build filter expression if file_id is provided
    filter_expr = None
    if file_id is not None:
        filter_expr = f'document_id == "{file_id}"'

    chunks = []
    logger.debug("Downloading chunks...")

    # Iterate over all matching records using iter_index
    try:
        for batch in iter_index(
            client=client,
            collection_name=chunk_es_index,
            output_fields=["content", "metadata"],
            filter_expr=filter_expr,
            batch_size=256,
        ):
            for item in batch:
                metadata = item.get("metadata", {})
                title = metadata.get("title", "") if isinstance(metadata, dict) else ""
                content = item.get("content", "")
                chunks.append({"title": title, "content": content})

    except Exception as e:
        logger.error(f"Failed to query Milvus: {e}")
        raise

    if chunk_file_path is not None:
        with open(chunk_file_path, "w+") as f:
            for chunk in chunks:
                f.write(json.dumps({"title": chunk["title"], "content": chunk["content"]}, ensure_ascii=False))
                f.write("\n")

    # Release client reference
    try:
        milvus_manager.release()
    except Exception:
        """Should ignore the error"""
        pass

    return chunks


async def extract_triples(
    file_id: str = None,
    chunk_file_path: str = None,
    output_path: str = None,
    config_obj=None,
    llm_client: BaseModelClient | None = None,
):
    logger.info("正在提取三元组...")
    cfg = config_obj or CONFIG
    if cfg is None:
        raise ValueError("config_obj (GraphRAGConfig) is required")
    chunk_src = (
        f"ES 索引 {cfg.chunk_es_index}（未指定 chunk_file_path）"
        if chunk_file_path is None
        else f"chunk 文件: {chunk_file_path}"
    )
    out_desc = "写入 ES triple 索引" if output_path is None else f"输出文件: {output_path}"
    logger.info(f"   Chunk来源: {chunk_src}")
    logger.info(f"   输出去向: {out_desc}")
    logger.info("")

    llm_client = llm_client or getattr(cfg, "llm_client_instance", None)
    if llm_client is None:
        raise ValueError("llm_client is required (SDK 不再自动创建 LLM 客户端)")

    chunks = load_index(
        cfg.milvus_uri, cfg.milvus_token, cfg.chunk_es_index, chunk_file_path=chunk_file_path, file_id=file_id
    )
    model_name = get_model_name(cfg)
    limiter = anyio.CapacityLimiter(_concurrent_limit(cfg))
    chunk2triples = await process_data(
        chunks,
        llm_client,
        limiter,
        output_path,
        model_name=model_name,
    )

    logger.info("三元组提取完成！")
    return chunk2triples
