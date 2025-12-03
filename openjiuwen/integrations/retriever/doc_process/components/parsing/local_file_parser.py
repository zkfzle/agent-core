# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import asyncio
import os
import uuid
from typing import Dict, List, Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.file import PyMuPDFReader

from openjiuwen.core.common.logging import logger

UTF8_ENV = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}


async def _run(argv: list[str], timeout: float = 20.0) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **UTF8_ENV},
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    return proc.returncode, out, err


async def _extract_doc_async(path) -> str:
    # antiword -> utf-8; fall back to catdoc if output looks suspicious
    code, out, err = await _run(["antiword", "-m", "UTF-8.txt", "-w", "0", str(path)])
    text = out.decode("utf-8", "replace") if code == 0 else ""
    if code != 0 or ("\x00" in text) or (not text.strip()):
        code, out, err = await _run(["catdoc", "-d", "utf-8", str(path)])
        if code != 0:
            raise RuntimeError(err.decode("utf-8", "ignore") or "doc parse failed")
        text = out.decode("utf-8", "replace")
    return text


class DocToTextAsyncReader(BaseReader):
    async def aload_data(self, file, extra_info: Optional[Dict] = None) -> List[Document]:
        logger.info(f"Using custom .doc extractor for {file}")
        text = await _extract_doc_async(file)
        meta = {"source": str(file)}
        if extra_info:
            meta.update(extra_info)  # Keep metadata from SimpleDirectoryReader
        return [Document(text=text, metadata=meta)]

    def load_data(self, file, *args, **kwargs):
        raise Exception(f"Processing failed for {file}! Synchronous .doc loading not implemented!")


async def parse_file(filepath, filename, file_id):
    """
    filepath: location of file
    filename: name/title of the file
    file_id: unique identifier for the file
    """

    if not os.path.exists(filepath):
        raise RuntimeError(f"File {filepath} was not found when parsing!!! Exiting!!")

    reader = SimpleDirectoryReader(
        input_files=[filepath],
        file_extractor={
            ".pdf": PyMuPDFReader(),  # More robust PDF Parser
            ".doc": DocToTextAsyncReader(),  # Special processing logic for older .doc files
        },
    )

    dataset = []
    for document in await reader.aload_data():
        doc_data = {
            "title": filename,
            "doc_id": file_id,
            "paragraph_id": str(uuid.uuid4()),
            "paragraph_text": document.text,
        }
        dataset.append(doc_data)

    return dataset
