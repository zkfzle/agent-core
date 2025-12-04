# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import json
import os
import shutil
import uuid
from pathlib import Path


def load_json(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(fp):
    data = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) > 0:
                data.append(json.loads(line))
    return data


def load_jsonl_as_iterator(fp):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) > 0:
                yield json.loads(line)


def save_json(fp, data):
    if not isinstance(fp, Path):
        fp = Path(fp)

    fp.parent.mkdir(parents=True, exist_ok=True)

    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_jsonl(fp, data):
    if not isinstance(fp, Path):
        fp = Path(fp)

    fp.parent.mkdir(parents=True, exist_ok=True)

    with open(fp, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def copy_file_to_storage(local_dir: str, filename: str, file) -> tuple[str, str]:
    """
    Copy a file to local storage and generate a unique ID

    Args:
        local_dir: Directory to copy the file to
        filename: Original filename provided by user
        file: File object (e.g., UploadFile.file or any file-like object)

    Returns:
        tuple: (file_id, local_filename) where file_id is a UUID and local_filename is the storage filename
    """
    # Generate unique file ID
    file_id = str(uuid.uuid4())

    # Get upload path from configuration
    upload_dir = Path(local_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save file to local storage with ID as filename
    file_extension = Path(filename).suffix
    local_filename = f"{file_id}{file_extension}"
    local_file_path = upload_dir / local_filename

    # Save the uploaded file
    with open(local_file_path, "wb") as buffer:
        shutil.copyfileobj(file, buffer)
        buffer.flush()
        os.fsync(buffer.fileno())

    return file_id, local_filename
