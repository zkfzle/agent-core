# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import json

import anyio
from elasticsearch import Elasticsearch

from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.integrations.retriever.config.configuration import CONFIG
from openjiuwen.integrations.retriever.doc_process.components.extraction.llm_openie import PROMPT as PROMPT_TEMPLATE
from openjiuwen.integrations.retriever.doc_process.components.extraction.llm_openie import LLMOpenIE
from openjiuwen.integrations.retriever.retrieval.llms.client import get_model_name
from openjiuwen.integrations.retriever.retrieval.utils import iter_index


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

    chunk2triples = {}
    for task in await asyncio.gather(*tasks):
        chunk2triples.update(task)
    return chunk2triples


def load_index(es_host: str, es_index: str, chunk_file_path: str = None, file_id: str = None) -> list[dict]:
    es = Elasticsearch(es_host)

    query = None
    if file_id is not None:
        query = {"bool": {"filter": [{"term": {"metadata.file_id": file_id}}]}}

    chunks = []
    logger.debug("Downloading chunks...")
    for batch in iter_index(es, es_index, query=query):
        for item in batch:
            chunks.append({"title": item["_source"]["metadata"]["title"], "content": item["_source"]["content"]})

    if chunk_file_path is not None:
        with open(chunk_file_path, "w+") as f:
            for chunk in chunks:
                f.write(json.dumps({"title": chunk["title"], "content": chunk["content"]}, ensure_ascii=False))
                f.write("\n")

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
    logger.info(f"   ES URL: {cfg.es_url}")
    logger.info(f"   ES 索引: {cfg.chunk_es_index}")
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

    chunks = load_index(cfg.es_url, cfg.chunk_es_index, chunk_file_path=chunk_file_path, file_id=file_id)
    model_name = get_model_name(cfg)
    limiter = anyio.CapacityLimiter(_concurrent_limit(cfg))
    chunk2triples = await process_data(
        chunks,
        llm_client,
        limiter,
        output_path,
        model_name=model_name,
    )

    logger.info("三元组提取完成。")
    return chunk2triples
