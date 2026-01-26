# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Document Data Models

Contains Document and TextChunk data models.
"""

import base64
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Self, overload

from pydantic import BaseModel, Field
from pydantic_core import PydanticCustomError

NOT_SET = None


class Document(BaseModel):
    """Document data model"""

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Document ID")
    text: str = Field(..., description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class TextChunk(BaseModel):
    """Text chunk data model"""

    id_: str = Field(..., description="Chunk ID")
    text: str = Field(..., description="Chunk text content")
    doc_id: str = Field(..., description="Parent document ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: list[float] | None = Field(default=None, description="Chunk embedding vector")

    @classmethod
    def from_document(cls, doc: Document, chunk_text: str, id_: str = "") -> "TextChunk":
        """Create TextChunk from Document"""
        return cls(
            id_=id_ if id_ else str(uuid.uuid4()),
            text=chunk_text,
            doc_id=doc.id_,
            metadata=doc.metadata,
        )


class MultimodalDocument(Document):
    """Multimodal Document data model"""

    text: Optional[str] = Field(default=None, max_length=0, init=False)
    data: list[tuple[Literal["text", "image", "audio", "video"], str, str]] = Field(default_factory=list, init=False)

    @property
    def content(self) -> list[dict[str, Any]]:
        """Get the whole content field"""
        content = []
        for kind, data, data_id in self.data:
            match kind:
                case "text":
                    content.append({"type": "text", "text": data})
                case "image" | "video":
                    content.append(
                        {
                            "type": f"{kind}_url",
                            f"{kind}_url": {"url": data},
                            "uuid": data_id,
                        }
                    )
                case "audio":
                    file_format = re.match(r"data:audio/([a-z0-9_]+);base64,", data).group(1)
                    content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data, "format": file_format},
                            "uuid": data_id,
                        }
                    )
        return content

    @overload
    def add_field(self, kind: Literal["text"], data: str) -> Self:
        """Add a text field to current multimodal document, you can chain add_field calls together, for example:

        `doc = MultimodalDocument().add_field("text", "hello world").add_field("image", ...)`

        Args:
            kind (Literal["text", "image", "audio", "video"]): modality of the new field.
            data (str, optional): Base64-encoded str of the data.

        Returns:
            Self: the current MultimodalDocument instance
        """

    @overload
    def add_field(
        self,
        kind: Literal["image", "audio", "video"],
        data: str = NOT_SET,
        file_path: Path = NOT_SET,
        data_id: str = "",
    ) -> Self: ...

    def add_field(
        self,
        kind: Literal["text", "image", "audio", "video"],
        data: str = NOT_SET,
        file_path: Path = NOT_SET,
        data_id: str = "",
    ) -> Self:
        """Add a data field to current multimodal document, you can chain add_field calls together, for example:

        `doc = MultimodalDocument().add_field("text", "hello world").add_field("image", ...)`

        Args:
            kind (Literal["text", "image", "audio", "video"]): modality of the new field.
            data (str, optional): Base64-encoded str of the data.
            file_path (Path, optional): a valid file path to a multimodal file.
            data_id (str, optional): uuid for multimodal caching, leave blank if unsure.

        Returns:
            Self: the current MultimodalDocument instance
        """
        kind, data = _load_multimodal_data(kind, data, file_path)
        if data_id:
            if len(data_id) > 32:
                raise PydanticCustomError(
                    "invalid_uuid_provided",
                    'MultimodalDocument.add_field received invalid "data_id", uuid is a string of length 32',
                    {"data_id": data_id},
                )
        else:
            data_id = uuid.uuid4().hex
        self.data.append((kind, data, data_id))
        return self


def _load_multimodal_data(
    kind: Literal["text", "image", "audio", "video"],
    data: str = NOT_SET,
    file_path: Path = NOT_SET,
) -> tuple[Literal["text", "image", "audio", "video"], str]:
    error_context = {"kind": kind, "file_path": file_path, "data": data}
    if kind not in ["text", "image", "audio", "video"]:
        raise PydanticCustomError(
            "unknown_kind",
            f'Unknown kind of multimodal file: {kind}, supported option: ["text", "image", "audio", "video"]',
            error_context,
        )
    if file_path is None and data is None:
        raise PydanticCustomError(
            f"no_{kind}_source_provided",
            f"MultimodalDocument.add_field received no data of {kind} type",
            error_context,
        )
    if file_path is not None and data is not None:
        raise PydanticCustomError(
            f"too_many_{kind}_source_provided",
            'MultimodalDocument.add_field cannot accept both "file_path" and "data", please only set one',
            error_context,
        )
    if isinstance(data, str):
        if kind == "text" or re.match(f"data:{kind}/([a-z0-9_]+);base64,", data):
            return kind, data
        raise PydanticCustomError(
            f"invalid_{kind}_data_provided",
            f'MultimodalDocument.add_field received invalid "data", this value should start with "data:{kind}/"',
            error_context,
        )
    if not isinstance(file_path, Path):
        raise PydanticCustomError(
            f"invalid_{kind}_file_path_provided",
            'MultimodalDocument.add_field received invalid "file_path", this value should be a Path',
            error_context,
        )
    if not file_path.is_file():
        raise PydanticCustomError(
            f"{kind}_path_invalid",
            f"Unable to open {kind} file at {file_path}",
            error_context,
        )
    file_ext = file_path.suffix.casefold().replace(".jpg", ".jpeg").removeprefix(".")
    b64_prefix = f"data:{kind}/{file_ext};base64,"
    try:
        if kind == "text":
            return kind, file_path.read_text(encoding="utf-8")
        return kind, b64_prefix + base64.b64encode(file_path.read_bytes()).decode()
    except Exception as e:
        raise PydanticCustomError(
            f"error_loading_{kind}",
            f"Unable to load {kind} file into base64: {e}",
            error_context,
        ) from e
