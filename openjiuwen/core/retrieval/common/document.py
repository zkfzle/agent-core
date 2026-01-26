# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Document Data Models

Contains Document and TextChunk data models.
"""

import base64
from pathlib import Path
import uuid
from typing import Any, Dict, Optional, Self

from pydantic import BaseModel, Field, model_validator
from pydantic_core import PydanticCustomError

NOT_SET = None
SUPPORTED_IMAGE_TYPES = {
    "png", "jpeg", "gif", "bmp", "tiff"
}


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
    text: Optional[str] = Field(default=None, description="Document text content")
    image_path: Path = Field(
        default=NOT_SET, description="Valid file path to a png/jpg image",
    )
    image_b64: str = Field(
        default=NOT_SET, description="Base64-encoded str of a png/jpg image", pattern=r"data:image/[a-z]+;base64,",
    )
    lazy_load: bool = Field(
        default=False, description="Use lazy loading for fetching image content (get when needed)",
    )
    image: str = Field(default=None, init=False)
    audio: str = Field(default=None, init=False)

    @model_validator(mode="after")
    def check_image(self) -> Self:
        """Check image input"""
        error_context = {"image_path": self.image_path, "image_b64": self.image_b64}
        if self.image_path is None and self.image_b64 is None:
            raise PydanticCustomError(
                "no_image_source_provided",
                'MultimodalDocument require either "image_path" or "image_b64" to be provided, for pure text documents '
                "please use the Document class",
                error_context,
            )
        if self.image_path is not None and self.image_b64 is not None:
            raise PydanticCustomError(
                "too_many_image_source_provided",
                'MultimodalDocument cannot accept both "image_path" and "image_b64", please only set one.',
                error_context,
            )
        if not self.lazy_load:
            self._load_image()
        return self

    def get_content(self) -> list[dict[str, Any]]:
        """Get image in base64 string format"""
        if self.image is None:
            self._load_image()
        content = [{"type": "image_url", "image_url": {"url": self.image}}]
        if self.text:
            content.append({"type": "text", "text": self.text})
        return content

    def _load_image(self) -> None:
        """Load image"""
        if self.image_b64 is not None:
            self.image = self.image_b64
        elif self.image_path is not None:
            if not self.image_path.is_file():
                raise PydanticCustomError(
                    "image_path_invalid",
                    f"Unable to open image file at {self.image_path}",
                    {"image_path": self.image_path},
                )
            file_ext = self.image_path.suffix.casefold().replace(".jpg", ".jpeg").removeprefix(".")
            if file_ext not in SUPPORTED_IMAGE_TYPES:
                raise PydanticCustomError(
                    "unknown_file_extension",
                    f"Unknown file format: {file_ext}",
                    {"image_path": self.image_path},
                )
            b64_prefix = f"data:image/{file_ext};base64,"
            try:
                self.image = b64_prefix + base64.b64encode(self.image_path.read_bytes()).decode()
            except Exception as e:
                raise PydanticCustomError(
                    "error_loading_image",
                    f"Unable to load image file into base64: {e}",
                    {"image_path": self.image_path},
                ) from e
