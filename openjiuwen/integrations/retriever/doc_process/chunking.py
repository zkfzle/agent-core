# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
异步切分接口 chunk_doc（仅做分段，不做向量/索引写入）。

行为：
- 输入为前端已解析好的段落列表（parse_doc 的输出），按字符长度进行滑窗切分（chunk_size / chunk_overlap_percent）。
- 可选预处理：normalize_whitespace / remove_url_email。
- 只返回内存结果，不访问 ES、不做 embedding。

返回格式 List[dict]:
    {
        "chunk_id": "<uuid>",
        "doc_id": "<传入的 doc_id>",
        "chunk_text": "<文本片段>",
        "start_char_idx": <int>,
        "end_char_idx": <int>,
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openjiuwen.integrations.retriever.doc_process.components.chunking.text_preprocessor import (
    PreprocessingPipeline,
    URLEmailRemover,
    WhitespaceNormalizer,
)


def _build_preprocess_pipeline(options: Optional[Dict]) -> PreprocessingPipeline:
    """根据选项构建预处理流水线。"""
    options = options or {}
    preprocessors = []
    if options.get("normalize_whitespace"):
        preprocessors.append(WhitespaceNormalizer(preserve_single_newline=False))
    if options.get("remove_url_email"):
        preprocessors.append(URLEmailRemover())
    return PreprocessingPipeline(preprocessors)


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """按字符滑窗切分，返回包含 start/end 的列表（不含 doc_id）。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能为负数")
    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if chunk:
            chunks.append(
                {
                    "chunk_id": str(uuid4()),
                    "chunk_text": chunk,
                    "start_char_idx": start,
                    "end_char_idx": end,
                }
            )
        start += step
    return chunks


async def chunk_doc(
    paragraphs: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap_percent: float = 0.0,
    doc_id: str = "",
    preprocess_options: Optional[Dict] = None,
) -> List[Dict]:
    """
    异步切分文档（按字符），仅返回内存结果。

    Args:
        paragraphs: 已解析的段落列表（必填），每项至少包含 paragraph_text，可带 doc_id/paragraph_id。
        chunk_size: 每个 chunk 的最大字符数（可选，默认 500）。
        chunk_overlap_percent: 相邻 chunk 的重叠百分比 0-100（可选，默认 0）。
        doc_id: 文档 ID，必填。
        preprocess_options: 预处理选项字典（可选，默认 None）：
            - normalize_whitespace: bool
            - remove_url_email: bool

    Returns:
        List[dict]，字段见模块说明。
    """
    pipeline = _build_preprocess_pipeline(preprocess_options)

    if not doc_id:
        raise ValueError("doc_id is required for chunk_doc")
    # 计算重叠字符数
    overlap = int(max(0.0, min(chunk_overlap_percent, 100.0)) * float(chunk_size) / 100.0)
    overlap = min(overlap, max(chunk_size - 1, 0))
    # 拼接自然段，中间以双换行保留段落边界感
    texts = []
    for p in paragraphs or []:
        t = p.get("paragraph_text", "")
        if pipeline.preprocessors:
            t = pipeline(t)
        texts.append(t)
    full_text = "\n\n".join(texts).strip()

    chunks = _split_text(full_text, chunk_size=chunk_size, chunk_overlap=overlap)
    # 回填 doc_id
    resolved_doc_id = doc_id
    for c in chunks:
        c["doc_id"] = resolved_doc_id
    return chunks


__all__ = ["chunk_doc"]
