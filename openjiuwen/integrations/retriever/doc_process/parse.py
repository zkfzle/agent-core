# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
单文件解析入口（Parsing Stage）。

功能
-----
- 解析单个源文件为“自然段”列表，不做固定长度切分，不写索引。
- PDF：PyMuPDFReader，通常每页/段落一条记录。
- DOC：antiword/catdoc 解析，整文件一条记录。
- DOCX/TXT/MD：默认文本 reader，通常整文件一条记录。
- 预切分 JSONL 不在此处理，应在后续 chunking/index 阶段用 precomputed_chunks=True。

返回
-----
List[dict]，每条至少包含：
{
  "doc_id": <调用方传入的 doc_id（必填）>,
  "paragraph_id": "<uuid>",
  "paragraph_text": "<原始段落文本>"
}

调用
-----
await parse_doc(doc, parsing_strategy=None, doc_id=None)
"""

import os
from typing import Any, Dict, List, Optional

from openjiuwen.integrations.retriever.doc_process.components.parsing import local_file_parser


async def parse_doc(
    doc: str,
    parsing_strategy: Optional[Any] = None,
    doc_id: str = "",
) -> List[Dict[str, Any]]:
    """异步解析：解析单个文件为段落列表（不做 chunk 切分，不写索引）。

    Args:
        doc: 文件绝对路径（必填）。
        parsing_strategy: 占位参数，当前未使用（可选，默认 None）。
        doc_id: 文档 ID，必填。

    Returns:
        List[dict]，每条包含 paragraph_id、paragraph_text、doc_id。
    """
    if not os.path.exists(doc):
        raise FileNotFoundError(f"file not found: {doc}")

    fname = os.path.basename(doc)
    if not doc_id:
        raise ValueError("doc_id is required for parse_doc")

    rows = await local_file_parser.parse_file(doc, fname, doc_id)
    # 规范字段：透传 doc_id，不再返回 space_id/title
    for r in rows:
        r.pop("title", None)
        r.pop("space_id", None)
        r["doc_id"] = doc_id
    return rows


__all__ = ["parse_doc"]
