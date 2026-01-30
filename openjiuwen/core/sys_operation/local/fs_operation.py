# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import asyncio
import datetime
import os
import pathlib
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal, List, AsyncIterator, Iterator

import aiofiles
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.sys_operation.fs import BaseFsOperation
from openjiuwen.core.sys_operation.base import OperationMode
from openjiuwen.core.sys_operation.registry import operation
from openjiuwen.core.sys_operation.result import (
    ReadFileResult, WriteFileResult, \
    UploadFileResult, DownloadFileResult, ListFilesResult, ListDirsResult, SearchFilesResult, \
    ReadFileStreamResult, DownloadFileStreamResult, UploadFileStreamResult,
    ReadFileData, ReadFileChunkData, WriteFileData, UploadFileData, DownloadFileData,
    FileSystemItem, FileSystemData, SearchFilesData, DownloadFileChunkData, UploadFileChunkData
)

SAFE_PATH_PATTERN = re.compile(r'[^\w.-]')


@dataclass(frozen=True)
class _ListItemsSpec:
    path: str

    include_files: bool = True
    include_dirs: bool = True

    recursive: bool = False
    max_depth: Optional[int] = None

    sort_by: Literal["name", "modified_time", "size"] = "name"
    sort_descending: bool = False

    file_types: Optional[List[str]] = None


@dataclass
class _ReadParams:
    """Read parameters for file operations."""

    path: str
    head: Optional[int] = None
    tail: Optional[int] = None
    line_range: Optional[Tuple[int, int]] = None
    is_stream: bool = False
    encoding: str = "utf-8"
    mode: Literal['text', 'bytes'] = "text"
    file_path: Optional[pathlib.Path] = None


@operation(name="fs", mode=OperationMode.LOCAL, description="local fs operation")
class FsOperation(BaseFsOperation):
    """File system operation"""

    async def read_file(
            self,
            path: str,
            *,
            mode: Literal['text', 'bytes'] = "text",
            head: Optional[int] = None,
            tail: Optional[int] = None,
            line_range: Optional[Tuple[int, int]] = None,
            encoding: str = "utf-8",
            chunk_size: int = 8192,
            options: Optional[Dict[str, Any]] = None
    ) -> ReadFileResult:
        """
        Asynchronously read file with specified mode and parameters.
        Mutually exclusive parameters: Only one of head, tail, or line_range can be specified.

        Args:
            path: Full or relative path to the file to read (required).
            mode: Reading mode - "text" (line-based, default) or "bytes" (raw bytes).
            head: Number of lines to read from the start (text mode only).
                  0 is equivalent to None.
            tail: Number of lines to read from the end (text mode only).
                  0 is equivalent to None.
            line_range: Specific line range to read (start, end) - 1-indexed, inclusive (text mode only).
                  If start <= 0 or end <= 0 or start > end, returns empty content.
            encoding: Character encoding for text mode (default: utf-8).
            chunk_size: Buffer size for bytes mode reading (default: 8192 bytes).
            options: Extended configuration options (dict, optional).

        Returns:
            ReadFileResult: Structured result.
        """
        try:
            # Validate parameters and resolve path
            read_params = _ReadParams(
                path=path,
                head=head,
                tail=tail,
                line_range=line_range,
                is_stream=False,
                encoding=encoding,
                mode=mode
            )
            validated_params, error_result = await self._validate_and_resolve_path(
                read_params, "read_file"
            )
            if error_result:
                return error_result

            # Extract validated parameters
            head = validated_params.head
            tail = validated_params.tail
            line_range = validated_params.line_range
            file_path = validated_params.file_path

            if mode == "bytes":
                # Check for text mode only parameters in binary mode
                if head is not None or tail is not None or line_range is not None:
                    return self._create_error_result(
                        "read_file",
                        f"Parameters 'head', 'tail', and 'line_range' are only supported in text mode",
                        ReadFileResult
                    )
                final_content = await self._read_bytes(file_path, chunk_size)
            elif head is None and tail is None and line_range is None:
                # Fast path for full text reading
                async with aiofiles.open(file_path, mode="r", encoding=encoding) as f:
                    final_content = await f.read()
            else:
                # Reuse read_file_stream for slicing logic
                lines = []
                async for res in self.read_file_stream(
                        path, mode=mode, head=head, tail=tail, line_range=line_range, encoding=encoding
                ):
                    if res.code != 0:
                        return ReadFileResult(code=res.code, message=res.message)
                    lines.append(res.data.chunk_content)
                final_content = "\n".join(lines)

            data = ReadFileData(path=str(file_path), content=final_content, mode=mode)
            return ReadFileResult(
                code=StatusCode.SUCCESS.code,
                message=StatusCode.SUCCESS.errmsg,
                data=data
            )
        except Exception as e:
            return self._create_error_result("read_file", str(e), ReadFileResult)

    async def read_file_stream(
            self,
            path: str,
            *,
            mode: Literal['text', 'bytes'] = "text",
            head: Optional[int] = None,
            tail: Optional[int] = None,
            line_range: Optional[Tuple[int, int]] = None,
            encoding: str = "utf-8",
            chunk_size: int = 8192,
            options: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[ReadFileStreamResult]:
        """
        Asynchronously read file streaming with specified mode and parameters.
        Mutually exclusive parameters: Only one of head, tail, or line_range can be specified.

        Args:
            path: Full or relative path to the file to read (required).
            mode: Reading mode - "text" (line-based, default) or "bytes" (raw bytes).
            head: Number of lines to read from the start (text mode only).
                  0 is equivalent to None.
            tail: Number of lines to read from the end (text mode only).
                  0 is equivalent to None.
            line_range: Specific line range to read (start, end) - 1-indexed, inclusive (text mode only).
                  If start <= 0 or end <= 0 or start > end, returns empty content.
            encoding: Character encoding for text mode (default: utf-8).
            chunk_size: Buffer size for bytes mode reading (default: 8192 bytes).
            options: Extended configuration options (dict, optional).

        Returns:
            AsyncIterator[ReadFileStreamResult]: Streaming structured results, line-by-line or chunk-by-chunk.
        """
        try:
            read_params = _ReadParams(
                path=path,
                head=head,
                tail=tail,
                line_range=line_range,
                is_stream=True,
                encoding=encoding,
                mode=mode
            )

            validated_params, error_result = await self._validate_and_resolve_path(
                read_params, "read_file_stream"
            )
            if error_result:
                yield error_result
                return

            # Extract validated parameters
            head = validated_params.head
            tail = validated_params.tail
            line_range = validated_params.line_range
            file_path = validated_params.file_path

            # bytes mode
            if mode != "text":
                if head is not None or tail is not None or line_range is not None:
                    yield self._create_error_result(
                        "read_file_stream",
                        "Parameters 'head', 'tail', and 'line_range' are only supported in text mode",
                        ReadFileStreamResult,
                    )
                    return

                async for chunk in self._read_bytes_stream(file_path, chunk_size):
                    yield chunk
                return

            # text mode
            async for chunk in self._stream_text_file(validated_params):
                yield chunk

        except Exception as e:
            yield self._create_error_result("read_file_stream", str(e), ReadFileStreamResult)

    async def write_file(
            self,
            path: str,
            content: str | bytes,
            *,
            mode: Literal['text', 'bytes'] = "text",
            prepend_newline: bool = True,
            append_newline: bool = False,
            create_if_not_exist: bool = True,
            permissions: str = "644",
            encoding: str = "utf-8",
            options: Optional[Dict[str, Any]] = None
    ) -> WriteFileResult:
        """
        Asynchronously writes content to a file with flexible configuration.

        Args:
            path: Full or relative path to the file to write (required).
            content: Data to write to the file (string for text mode, bytes for binary mode).
            mode: Writing mode: "text" (for string content) or "bytes" (for binary data) (default: "text").
            prepend_newline: Add a newline character (`\n`) before the content (text mode only; default: True).
            append_newline: Add a newline character (`\n`) after the content (text mode only; default: False).
            create_if_not_exist: Auto-create the file if it doesn't exist (default: True).
            permissions: Octal file permissions (Unix/Linux only; ignored on Windows) (default: "644").
            encoding: Character encoding for text mode (default: utf-8).
            options: Extended configuration options (dict, optional).

        Returns:
            WriteFileResult: Structured result.
        """
        try:
            file_path = self._resolve_path(path, create_parent=True)
            if file_path.is_dir():
                return self._create_error_result("write_file", f"Target path is a directory: {file_path}", WriteFileResult)
            if not create_if_not_exist and not file_path.exists():
                return self._create_error_result("write_file", f"File does not exist: {file_path}", WriteFileResult)

            if mode == "text":
                txt = str(content)
                if prepend_newline:
                    txt = "\n" + txt
                if append_newline:
                    txt = txt + "\n"

                data_bytes = txt.encode(encoding)
            else:
                data_bytes = content if isinstance(content, (bytes, bytearray)) else bytes(content)

            async with aiofiles.open(file_path, mode="wb") as f:
                await f.write(data_bytes)

            self._apply_permissions(file_path, permissions)
            return WriteFileResult(
                code=StatusCode.SUCCESS.code,
                message=StatusCode.SUCCESS.errmsg,
                data=WriteFileData(path=str(file_path), size=len(data_bytes), mode=mode)
            )
        except Exception as e:
            return self._create_error_result("write_file", str(e), WriteFileResult)

    async def upload_file(
            self,
            local_path: str,
            target_path: str,
            *,
            overwrite: bool = False,
            create_parent_dirs: bool = True,
            preserve_permissions: bool = True,
            chunk_size: int = 1024 * 1024,
            options: Optional[Dict[str, Any]] = None
    ) -> UploadFileResult:
        """
        Asynchronous file upload (semantics: local file → target path).

        Args:
            local_path: Local source file path (required, e.g. /tmp/local_file.txt).
            target_path: Upload destination path (required, e.g. /mnt/storage/file.txt or sandbox:/opt/bucket/file.txt).
            overwrite: Whether to overwrite existing target file (default: False).
            create_parent_dirs: Whether to auto-create target parent directories (default: True).
            preserve_permissions: Whether to preserve file permissions (default: True, Unix/Linux only).
            chunk_size: Chunk size for cross-filesystem transfers (default: 1MB, bytes).
            options: Extended configuration options (dict, optional).

        Returns:
            UploadFileResult: Structured result.
        """
        try:
            src = pathlib.Path(local_path).expanduser().resolve()
            dst = self._resolve_path(target_path, create_parent=create_parent_dirs)
            if not src.is_file():
                return self._create_error_result("upload_file", f"Source not found: {src}", UploadFileResult)
            if dst.exists() and not overwrite:
                return self._create_error_result("upload_file", f"Target exists: {dst}", UploadFileResult)

            size = await self._transfer_file(src, dst, chunk_size)
            if preserve_permissions:
                self._copy_permissions(src, dst)
            return UploadFileResult(
                code=StatusCode.SUCCESS.code,
                message=StatusCode.SUCCESS.errmsg,
                data=UploadFileData(local_path=str(src), target_path=str(dst), size=size)
            )
        except Exception as e:
            return self._create_error_result("upload_file", str(e), UploadFileResult)

    async def upload_file_stream(
            self,
            local_path: str,
            target_path: str,
            *,
            overwrite: bool = False,
            create_parent_dirs: bool = True,
            preserve_permissions: bool = True,
            chunk_size: int = 1024 * 1024,
            options: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[UploadFileStreamResult]:
        """
        Asynchronous file upload streaming(semantics: local file → target path).

        Args:
            local_path: Local source file path (required, e.g. /tmp/local_file.txt).
            target_path: Upload destination path (required, e.g. /mnt/storage/file.txt or sandbox:/opt/bucket/file.txt).
            overwrite: Whether to overwrite existing target file (default: False).
            create_parent_dirs: Whether to auto-create target parent directories (default: True).
            preserve_permissions: Whether to preserve file permissions (default: True, Unix/Linux only).
            chunk_size: Chunk size for cross-filesystem transfers (default: 1MB, bytes).
            options: Extended configuration options (dict, optional).

        Returns:
            AsyncIterator[UploadFileStreamResult]: Streaming structured results, chunk-by-chunk.
        """
        try:
            src = pathlib.Path(local_path).expanduser().resolve()
            dst = self._resolve_path(target_path, create_parent=create_parent_dirs)
            if not src.is_file():
                yield self._create_error_result("upload_file_stream", f"Source not found: {src}",
                                                UploadFileStreamResult)
                return
            if dst.exists() and not overwrite:
                yield self._create_error_result("upload_file_stream", f"Target exists: {dst}", UploadFileStreamResult)
                return

            async with aiofiles.open(src, mode="rb") as src_f, aiofiles.open(dst, mode="wb") as dst_f:
                index = 0
                while True:
                    chunk_bytes = await src_f.read(chunk_size)
                    if not chunk_bytes:
                        break

                    await dst_f.write(chunk_bytes)
                    yield UploadFileStreamResult(
                        code=StatusCode.SUCCESS.code,
                        message=StatusCode.SUCCESS.errmsg,
                        data=UploadFileChunkData(
                            local_path=str(src), target_path=str(dst), chunk_size=len(chunk_bytes), chunk_index=index,
                            is_last_chunk=False
                        )
                    )
                    index += 1

            if preserve_permissions:
                self._copy_permissions(src, dst)
        except Exception as e:
            yield self._create_error_result("upload_file_stream", str(e), UploadFileStreamResult)

    async def download_file(
            self,
            source_path: str,
            local_path: str,
            *,
            overwrite: bool = False,
            create_parent_dirs: bool = True,
            preserve_permissions: bool = True,
            chunk_size: int = 1024 * 1024,
            options: Optional[Dict[str, Any]] = None
    ) -> DownloadFileResult:
        """
        Asynchronous file download (semantics: source file → local destination path).

        Args:
            source_path: Source file path (required, e.g. /mnt/storage/file.txt or sandbox:/opt/bucket/file.txt).
            local_path: Local destination file path (required, e.g. /home/user/downloads/file.txt).
            overwrite: Whether to overwrite existing target file (default: False).
            create_parent_dirs: Whether to auto-create target parent directories (default: True).
            preserve_permissions: Whether to preserve file permissions (default: True, Unix/Linux only).
            chunk_size: Chunk size for cross-filesystem transfers (default: 1MB, bytes).
            options: Extended configuration options (dict, optional).

        Returns:
            DownloadFileResult: Structured result.
        """
        try:
            src = self._resolve_path(source_path)
            dst = pathlib.Path(local_path).expanduser().resolve()
            if not src.is_file():
                return self._create_error_result("download_file", f"Source not found: {src}", DownloadFileResult)
            if dst.exists() and not overwrite:
                return self._create_error_result("download_file", f"Destination exists: {dst}", DownloadFileResult)
            if create_parent_dirs:
                dst.parent.mkdir(parents=True, exist_ok=True)

            size = await self._transfer_file(src, dst, chunk_size)
            if preserve_permissions:
                self._copy_permissions(src, dst)
            return DownloadFileResult(
                code=StatusCode.SUCCESS.code,
                message=StatusCode.SUCCESS.errmsg,
                data=DownloadFileData(source_path=str(src), local_path=str(dst), size=size)
            )
        except Exception as e:
            return self._create_error_result("download_file", str(e), DownloadFileResult)

    async def download_file_stream(
            self,
            source_path: str,
            local_path: str,
            *,
            overwrite: bool = False,
            create_parent_dirs: bool = True,
            preserve_permissions: bool = True,
            chunk_size: int = 1024 * 1024,
            options: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[DownloadFileStreamResult]:
        """
        Asynchronous file download streaming(semantics: source file → local destination path).

        Args:
            source_path: Source file path (required, e.g. /mnt/storage/file.txt or sandbox:/opt/bucket/file.txt).
            local_path: Local destination file path (required, e.g. /home/user/downloads/file.txt).
            overwrite: Whether to overwrite existing target file (default: False).
            create_parent_dirs: Whether to auto-create target parent directories (default: True).
            preserve_permissions: Whether to preserve file permissions (default: True, Unix/Linux only).
            chunk_size: Chunk size for cross-filesystem transfers (default: 1MB, bytes).
            options: Extended configuration options (dict, optional).

        Returns:
            AsyncIterator[DownloadFileStreamResult]: Streaming structured results, chunk-by-chunk.
        """
        try:
            src = self._resolve_path(source_path)
            dst = pathlib.Path(local_path).expanduser().resolve()
            if not src.is_file():
                yield self._create_error_result("download_file_stream", f"Source not found: {src}",
                                                DownloadFileStreamResult)
                return
            if dst.exists() and not overwrite:
                yield self._create_error_result("download_file_stream", f"Destination exists: {dst}",
                                                DownloadFileStreamResult)
                return
            if create_parent_dirs:
                dst.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(src, mode="rb") as src_f, aiofiles.open(dst, mode="wb") as dst_f:
                index = 0
                while True:
                    chunk_bytes = await src_f.read(chunk_size)
                    if not chunk_bytes:
                        break

                    await dst_f.write(chunk_bytes)
                    yield DownloadFileStreamResult(
                        code=StatusCode.SUCCESS.code,
                        message=StatusCode.SUCCESS.errmsg,
                        data=DownloadFileChunkData(
                            source_path=str(src), local_path=str(dst), chunk_size=len(chunk_bytes), chunk_index=index,
                            is_last_chunk=False
                        )
                    )
                    index += 1

            if preserve_permissions:
                self._copy_permissions(src, dst)
        except Exception as e:
            yield self._create_error_result("download_file_stream", str(e), DownloadFileStreamResult)

    async def list_files(
            self,
            path: str,
            *,
            recursive: bool = False,
            max_depth: Optional[int] = None,
            sort_by: Literal['name', 'modified_time', 'size'] = "name",
            sort_descending: bool = False,
            file_types: Optional[List[str]] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> ListFilesResult:
        """
        Asynchronously list files under the specified path.

        Args:
            path: Target parent directory path (required).
            recursive: Whether to list files in subdirectories recursively. Defaults to False.
            max_depth: Maximum recursion depth limit, only effective when recursive=True.
            sort_by: Sorting field, supports three options:
                'name' (sort by filename, default),
                'modified_time' (sort by last modification time),
                'size' (sort by file size in bytes).
            sort_descending: Whether to sort in descending order. Defaults to False (ascending order).
            file_types: Filter files by extension (list of extensions), e.g. ['.txt', '.pdf'].
            options: Extended configuration options (dict, optional).

        Returns:
            ListFilesResult: Structured result.
        """
        try:
            spec = _ListItemsSpec(
                path=path,
                include_files=True,
                include_dirs=False,
                recursive=recursive,
                max_depth=max_depth,
                sort_by=sort_by,
                sort_descending=sort_descending,
                file_types=file_types,
            )
            items = await asyncio.to_thread(
                self._list_items_internal_sync,
                spec,
            )
            return ListFilesResult(
                code=StatusCode.SUCCESS.code,
                message=StatusCode.SUCCESS.errmsg,
                data=FileSystemData(total_count=len(items), list_items=items,
                                    root_path=str(self._resolve_path(path)), recursive=recursive,
                                    max_depth=max_depth)
            )
        except Exception as e:
            return self._create_error_result("list_files", str(e), ListFilesResult)

    async def list_directories(
            self,
            path: str,
            *,
            recursive: bool = False,
            max_depth: Optional[int] = None,
            sort_by: Literal['name', 'modified_time', 'size'] = "name",
            sort_descending: bool = False,
            options: Optional[Dict[str, Any]] = None
    ) -> ListDirsResult:
        """
        Asynchronously list directories under the specified path.

        Args:
            path: Target parent directory path (required).
            recursive: Whether to list subdirectories recursively. Defaults to False.
            max_depth: Maximum recursion depth limit, only effective when recursive=True.
            sort_by: Sorting field, supports three options:
                'name' (sort by filename, default),
                'modified_time' (sort by last modification time),
                'size' (sort by file size in bytes).
            sort_descending: Whether to sort in descending order. Defaults to False (ascending order).
            options: Extended configuration options (dict, optional).

        Returns:
            ListDirsResult: Structured result.
        """
        try:
            spec = _ListItemsSpec(
                path=path,
                include_files=False,
                include_dirs=True,
                recursive=recursive,
                max_depth=max_depth,
                sort_by=sort_by,
                sort_descending=sort_descending,
                file_types=None,
            )

            items = await asyncio.to_thread(
                self._list_items_internal_sync,
                spec,
            )

            return ListDirsResult(
                code=StatusCode.SUCCESS.code,
                message=StatusCode.SUCCESS.errmsg,
                data=FileSystemData(
                    total_count=len(items),
                    list_items=items,
                    root_path=str(self._resolve_path(path)),
                    recursive=recursive,
                    max_depth=max_depth,
                ),
            )

        except Exception as e:
            return self._create_error_result("list_directories", str(e), ListDirsResult)

    async def search_files(
            self,
            path: str,
            pattern: str,
            exclude_patterns: Optional[List[str]] = None
    ) -> SearchFilesResult:
        """
        Asynchronously search files under the specified path.

        Args:
            path: Base directory path to start the search (required).
            pattern: Search pattern to match file names.
            exclude_patterns: Optional list of patterns to exclude from results.

        Returns:
            SearchFilesResult: Structured result.
        """
        try:
            items = await asyncio.to_thread(
                self._search_files_internal_sync,
                path,
                pattern,
                exclude_patterns
            )

            return SearchFilesResult(
                code=StatusCode.SUCCESS.code,
                message=StatusCode.SUCCESS.errmsg,
                data=SearchFilesData(
                    total_matches=len(items),
                    matching_files=items,
                    search_path=str(self._resolve_path(path)),
                    search_pattern=pattern,
                    exclude_patterns=exclude_patterns
                )
            )
        except Exception as e:
            return self._create_error_result("search_files", str(e), SearchFilesResult)

    def _search_files_internal_sync(
            self,
            path: str,
            pattern: str,
            exclude_patterns: Optional[List[str]] = None
    ) -> List[FileSystemItem]:
        base = self._resolve_path(path)
        if not base.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {base}")

        matched_paths = list(base.rglob(pattern))
        if exclude_patterns:
            exclude_set = set()
            for pat in exclude_patterns:
                exclude_set.update(set(base.rglob(pat)))
            matched_paths = [p for p in matched_paths if p not in exclude_set]

        items = []
        for p in matched_paths:
            if p.is_file():
                item = self._create_fs_item(p)
                if item:
                    items.append(item)
        return items

    def _resolve_path(self, path: str, create_parent: bool = False) -> pathlib.Path:
        """Resolve path, enforce work_dir sandbox (if configured), and sanitize filenames."""
        work_dir_val = getattr(self._run_config, 'work_dir', None)

        if work_dir_val is None:
            # if work_dir is not configured
            final_path = pathlib.Path(path).expanduser().resolve()
        else:
            work_dir = pathlib.Path(work_dir_val).expanduser().resolve()
            try:
                raw_resolved = (work_dir / path).resolve()
                rel_path = raw_resolved.relative_to(work_dir)
            except ValueError as e:
                raise build_error(status=StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR,
                                  execution="resolve_path",
                                  error_msg=f"Access denied: Path {path} traverses outside {work_dir}",
                                  cause=e) from e

            sanitized_parts = [re.sub(r'[^\w.-]', '_', part) for part in rel_path.parts]
            final_path = work_dir.joinpath(*sanitized_parts)

        if create_parent:
            final_path.parent.mkdir(parents=True, exist_ok=True)

        return final_path

    @staticmethod
    def _apply_permissions(path: pathlib.Path, permissions: str | int) -> None:
        """Apply octal permissions to path on Unix-like systems (Best effort)."""
        if os.name != "nt":
            try:
                perm_int = int(str(permissions), 8) if isinstance(permissions, str) else permissions
                os.chmod(path, perm_int)
            except Exception:
                # Permission application is best-effort; failures (e.g. on non-Unix) are ignored
                pass

    @staticmethod
    def _copy_permissions(src: pathlib.Path, dst: pathlib.Path) -> None:
        """Copy permissions from src to dst on Unix-like systems (Best effort)."""
        if os.name != "nt":
            try:
                st = src.stat()
                os.chmod(dst, st.st_mode)
            except Exception:
                # Permission copy is best-effort; failures (e.g. source stat issues) are ignored
                pass

    @staticmethod
    async def _transfer_file(src: pathlib.Path, dst: pathlib.Path, chunk_size: int) -> int:
        """Asynchronously copy file contents from src to dst in chunks. Returns total size."""
        total_size = 0
        async with aiofiles.open(src, mode="rb") as src_f, aiofiles.open(dst, mode="wb") as dst_f:
            while True:
                chunk = await src_f.read(chunk_size)
                if not chunk:
                    break

                await dst_f.write(chunk)
                total_size += len(chunk)
        return total_size

    @staticmethod
    def _walk_path(base: pathlib.Path, recursive: bool = False, max_depth: Optional[int] = None) -> Iterator[
        pathlib.Path]:
        """Consolidated directory walker with recursion and depth control."""
        if not recursive:
            yield from base.iterdir()
            return

        if max_depth is None:
            yield from base.rglob("*")
            return

        root_depth = len(base.parts)
        for root, dirs, files in os.walk(base):
            current_root = pathlib.Path(root)
            current_depth = len(current_root.parts) - root_depth
            if current_depth > max_depth:
                del dirs[:]
                continue
            for d in dirs:
                yield current_root / d
            for f in files:
                yield current_root / f
            if current_depth == max_depth:
                del dirs[:]

    @staticmethod
    def _create_fs_item(p: pathlib.Path) -> Optional[FileSystemItem]:
        """Centralized FileSystemItem creation with stat error handling."""
        try:
            stat = p.stat()
            is_dir = p.is_dir()
            return FileSystemItem(
                name=p.name, path=str(p), size=stat.st_size,
                modified_time=str(datetime.datetime.fromtimestamp(stat.st_mtime)),
                is_directory=is_dir, type=p.suffix if not is_dir else None,
            )
        except Exception:
            return None

    @staticmethod
    def _sort_items(items: List[FileSystemItem], sort_by: str, reverse: bool) -> None:
        """Sort FS items by name, modified_time, or size."""
        if sort_by == "name":
            items.sort(key=lambda i: i.name, reverse=reverse)
        elif sort_by == "modified_time":
            items.sort(key=lambda i: i.modified_time, reverse=reverse)
        elif sort_by == "size":
            items.sort(key=lambda i: i.size, reverse=reverse)

    def _list_items_internal_sync(
            self,
            spec: _ListItemsSpec,
    ) -> List[FileSystemItem]:
        base = self._resolve_path(spec.path)
        if not base.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {base}")

        items: List[FileSystemItem] = []

        for p in self._walk_path(base, spec.recursive, spec.max_depth):
            is_dir = p.is_dir()

            if not spec.include_files and not is_dir:
                continue
            if not spec.include_dirs and is_dir:
                continue
            if spec.file_types and not is_dir and p.suffix not in spec.file_types:
                continue

            item = self._create_fs_item(p)
            if item:
                items.append(item)

        self._sort_items(items, spec.sort_by, spec.sort_descending)
        return items

    def _validate_read_params(
            self,
            read_params: _ReadParams,
    ):
        """
        Validate read parameters.

        Returns:
            (head, tail, line_range, error_msg)
        """
        head = read_params.head
        tail = read_params.tail
        line_range = read_params.line_range

        # Handle zero values - treat as not passed (return full content)
        if head == 0:
            head = None
        if tail == 0:
            tail = None

        # Mutually exclusive checks
        if tail is not None:
            if head is not None:
                return None, None, None, f"tail and head cannot be specified simultaneously"
            if line_range is not None:
                return None, None, None, f"tail and line_range cannot be specified simultaneously"
        elif head is not None and line_range is not None:
            return None, None, None, f"head and line_range cannot be specified simultaneously"
        return head, tail, line_range, None

    @staticmethod
    def _create_error_result(execution: str, error_msg: str, result_class: Any):
        """
        Create error result for file operations.

        Args:
            execution: The operation being executed.
            error_msg: The error message.
            result_class: The result class to instantiate.

        Returns:
            An instance of result_class with error information.
        """
        error_code = StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
        error_message = StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.errmsg.format(
            execution=execution,
            error_msg=error_msg
        )
        return result_class(
            code=error_code,
            message=error_message
        )

    async def _validate_and_resolve_path(
            self,
            read_params: _ReadParams,
            execution: str,
    ):
        """
        Validate parameters and resolve path for file operations.
        
        Returns:
            Tuple[Optional[_ReadParams], Optional[Union[ReadFileResult, ReadFileStreamResult]]]:
                - Validated ReadParams object with resolved file_path
                - Error result if validation failed
        """
        head, tail, line_range, error_msg = self._validate_read_params(read_params)
        result_class = ReadFileStreamResult if read_params.is_stream else ReadFileResult
        if error_msg:
            error_result = self._create_error_result(
                execution,
                error_msg,
                result_class,
            )
            return None, error_result

        file_path = self._resolve_path(read_params.path)
        if not file_path.is_file():
            error_result = self._create_error_result(
                execution,
                f"File not found: {file_path}",
                result_class,
            )
            return None, error_result

        # Create a new ReadParams with validated values
        validated_params = _ReadParams(
            path=read_params.path,
            head=head,
            tail=tail,
            line_range=line_range,
            is_stream=read_params.is_stream,
            encoding=getattr(read_params, 'encoding', 'utf-8'),
            mode=getattr(read_params, 'mode', 'text'),
            file_path=file_path
        )

        return validated_params, None

    @staticmethod
    async def _read_bytes(file_path: pathlib.Path, chunk_size: int):
        """
        Read file in binary mode.

        Args:
            file_path: The path to the file.
            chunk_size: Buffer size for reading.

        Returns:
            The read bytes content.
        """
        async with aiofiles.open(file_path, mode="rb") as f:
            return await f.read(chunk_size)

    @staticmethod
    async def _read_bytes_stream(file_path: pathlib.Path, chunk_size: int):
        """
        Read file in binary mode as a stream.

        Args:
            file_path: The path to the file.
            chunk_size: Buffer size for reading.

        Yields:
            ReadFileStreamResult with chunk data.
        """
        async with aiofiles.open(file_path, mode="rb") as f:
            index = 0
            while True:
                chunk_bytes = await f.read(chunk_size)
                if not chunk_bytes:
                    break
                yield ReadFileStreamResult(
                    code=StatusCode.SUCCESS.code,
                    message=StatusCode.SUCCESS.errmsg,
                    data=ReadFileChunkData(
                        path=str(file_path), chunk_content=chunk_bytes, mode="bytes",
                        chunk_size=len(chunk_bytes), chunk_index=index, is_last_chunk=False
                    )
                )
                index += 1

    @staticmethod
    async def _stream_text_file(read_params: _ReadParams):
        file_path = read_params.file_path
        encoding = read_params.encoding
        mode = read_params.mode
        head = read_params.head
        tail = read_params.tail
        line_range = read_params.line_range

        if tail is not None and tail < 0:
            # Return empty content for negative tail
            yield ReadFileStreamResult(
                code=StatusCode.SUCCESS.code,
                message=StatusCode.SUCCESS.errmsg,
                data=ReadFileChunkData(
                    path=str(file_path),
                    chunk_content="",
                    mode=mode,
                    chunk_size=0,
                    chunk_index=0,
                    is_last_chunk=True,
                ),
            )
            return

        if head is not None and head < 0:
            # Return empty content for negative head
            yield ReadFileStreamResult(
                code=StatusCode.SUCCESS.code,
                message=StatusCode.SUCCESS.errmsg,
                data=ReadFileChunkData(
                    path=str(file_path),
                    chunk_content="",
                    mode=mode,
                    chunk_size=0,
                    chunk_index=0,
                    is_last_chunk=True,
                ),
            )
            return

        if line_range is not None:
            start, end = line_range
            if start <= 0 or end <= 0 or start > end:
                # Return empty content for invalid line_range
                yield ReadFileStreamResult(
                    code=StatusCode.SUCCESS.code,
                    message=StatusCode.SUCCESS.errmsg,
                    data=ReadFileChunkData(
                        path=str(file_path),
                        chunk_content="",
                        mode=mode,
                        chunk_size=0,
                        chunk_index=0,
                        is_last_chunk=True,
                    ),
                )
                return

        # tail
        if tail is not None:
            buf = deque(maxlen=tail)
            async with aiofiles.open(file_path, mode="r", encoding=encoding) as f:
                async for line in f:
                    buf.append(line.rstrip("\n"))

            for i, content in enumerate(buf):
                yield ReadFileStreamResult(
                    code=StatusCode.SUCCESS.code,
                    message=StatusCode.SUCCESS.errmsg,
                    data=ReadFileChunkData(
                        path=str(file_path),
                        chunk_content=content,
                        mode=mode,
                        chunk_size=len(content.encode(encoding)),
                        chunk_index=i,
                        is_last_chunk=(i == len(buf) - 1),
                    ),
                )
            return

        # head / line_range / full
        async with aiofiles.open(file_path, mode="r", encoding=encoding) as f:
            index = 0
            async for line in f:
                line_no = index + 1
                content = line.rstrip("\n")

                if line_range is not None:
                    start, end = line_range
                    if not (start <= line_no <= end):
                        index += 1
                        continue

                elif head is not None and index >= head:
                    break

                yield ReadFileStreamResult(
                    code=StatusCode.SUCCESS.code,
                    message=StatusCode.SUCCESS.errmsg,
                    data=ReadFileChunkData(
                        path=str(file_path),
                        chunk_content=content,
                        mode=mode,
                        chunk_size=len(content.encode(encoding)),
                        chunk_index=index,
                        is_last_chunk=False,
                    ),
                )
                index += 1
