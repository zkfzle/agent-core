#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from typing import Optional, Tuple, Dict, Any, Literal, List, AsyncIterator, Iterator
import os
import pathlib
from datetime import datetime

import aiofiles

from openjiuwen.core.sys_operation.base import BaseOperation, OperationMode
from openjiuwen.core.sys_operation.registry import operation
from openjiuwen.core.sys_operation.result.file_operation_result import (
    ReadFileResult, WriteFileResult, \
    UploadFileResult, DownloadFileResult, ListFilesResult, ListDirsResult, SearchFilesResult, \
    ReadFileStreamResult, DownloadFileStreamResult,
    ReadFileData, ReadFileChunkData, WriteFileData, UploadFileData, DownloadFileData,
    FileSystemItem, FileSystemData, SearchFilesData, DownloadFileChunkData
)


@operation(name="file", mode=OperationMode.LOCAL, description="local file operation")
class FileOperation(BaseOperation):
    """File operation"""

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
        """Asynchronously read file, using streaming internally for text slicing consistency."""
        try:
            file_path = self._resolve_path(path)
            if not file_path.is_file():
                return ReadFileResult(code=-1, message=f"File not found: {file_path}")

            if mode == "text":
                lines = []
                async for res in self.read_file_stream(
                    path, mode=mode, head=head, tail=tail, line_range=line_range, encoding=encoding
                ):
                    if res.code != 0:
                        return ReadFileResult(code=res.code, message=res.message)
                    lines.append(res.data.chunk_content)
                final_content = "\n".join(lines)
            else:
                async with aiofiles.open(file_path, mode="rb") as f:
                    final_content = await f.read()

            data = ReadFileData(path=str(file_path), content=final_content, mode=mode)
            return ReadFileResult(code=0, message="Read file successful", data=data)
        except Exception as e:
            return ReadFileResult(code=-1, message=str(e))

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
        """Asynchronously read file streaming with specified mode and parameters."""
        try:
            file_path = self._resolve_path(path)
            if not file_path.is_file():
                yield ReadFileStreamResult(code=-1, message=f"File not found: {file_path}")
                return

            if mode == "text":
                # For tail, we need to know the total lines if we want to stream efficiently, 
                # but splitlines is simpler for local files.
                if tail is not None:
                    async with aiofiles.open(file_path, mode="r", encoding=encoding) as f:
                        lines = (await f.read()).splitlines()
                    for i, content in enumerate(lines[-tail:]):
                        yield ReadFileStreamResult(code=0, message="success", data=ReadFileChunkData(
                            path=str(file_path), chunk_content=content, mode=mode,
                            chunk_size=len(content.encode(encoding)), chunk_index=i, is_last_chunk=(i == tail - 1)
                        ))
                    return

                async with aiofiles.open(file_path, mode="r", encoding=encoding) as f:
                    index = 0
                    async for line in f:
                        content_str = line.rstrip("\n")
                        if line_range:
                            start, end = line_range
                            if not (start <= index + 1 <= end):
                                index += 1
                                continue
                        elif head is not None and index >= head:
                            break

                        yield ReadFileStreamResult(code=0, message="success", data=ReadFileChunkData(
                            path=str(file_path), chunk_content=content_str, mode=mode,
                            chunk_size=len(content_str.encode(encoding)), chunk_index=index, is_last_chunk=False
                        ))
                        index += 1
            else:
                async with aiofiles.open(file_path, mode="rb") as f:
                    index = 0
                    while True:
                        chunk_bytes = await f.read(chunk_size)
                        if not chunk_bytes: break
                        yield ReadFileStreamResult(code=0, message="success", data=ReadFileChunkData(
                            path=str(file_path), chunk_content=chunk_bytes, mode=mode,
                            chunk_size=len(chunk_bytes), chunk_index=index, is_last_chunk=False
                        ))
                        index += 1
        except Exception as e:
            yield ReadFileStreamResult(code=-1, message=str(e))

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
        """Asynchronously writes content to a file."""
        try:
            file_path = self._resolve_path(path, create_parent=True)
            if file_path.is_dir():
                return WriteFileResult(code=-1, message=f"Target path is a directory: {file_path}")
            if not create_if_not_exist and not file_path.exists():
                return WriteFileResult(code=-1, message=f"File does not exist: {file_path}")

            if mode == "text":
                txt = str(content)
                if prepend_newline: txt = "\n" + txt
                if append_newline: txt = txt + "\n"
                data_bytes = txt.encode(encoding)
            else:
                data_bytes = content if isinstance(content, (bytes, bytearray)) else bytes(content)

            async with aiofiles.open(file_path, mode="wb") as f:
                await f.write(data_bytes)

            self._apply_permissions(file_path, permissions)
            return WriteFileResult(code=0, message="Write successful", data=WriteFileData(path=str(file_path), size=len(data_bytes), mode=mode))
        except Exception as e:
            return WriteFileResult(code=-1, message=str(e))

    async def upload_file(
            self, local_path: str, target_path: str, *, overwrite: bool = False, create_parent_dirs: bool = True,
            preserve_permissions: bool = True, chunk_size: int = 1024 * 1024, options: Optional[Dict[str, Any]] = None
    ) -> UploadFileResult:
        """Asynchronous file upload."""
        try:
            src = pathlib.Path(local_path).expanduser().resolve()
            dst = self._resolve_path(target_path, create_parent=create_parent_dirs)
            if not src.is_file(): 
                return UploadFileResult(code=-1, message=f"Source not found: {src}")
            if dst.exists() and not overwrite: 
                return UploadFileResult(code=-1, message=f"Target exists: {dst}")

            size = await self._transfer_file(src, dst, chunk_size)
            if preserve_permissions: 
                self._copy_permissions(src, dst)
            return UploadFileResult(code=0, message="Upload successful", data=UploadFileData(local_path=str(src), target_path=str(dst), size=size))
        except Exception as e:
            return UploadFileResult(code=-1, message=str(e))

    async def download_file(
            self, source_path: str, local_path: str, *, overwrite: bool = False, create_parent_dirs: bool = True,
            preserve_permissions: bool = True, chunk_size: int = 1024 * 1024, options: Optional[Dict[str, Any]] = None
    ) -> DownloadFileResult:
        """Asynchronous file download."""
        try:
            src = self._resolve_path(source_path)
            dst = pathlib.Path(local_path).expanduser().resolve()
            if not src.is_file(): 
                return DownloadFileResult(code=-1, message=f"Source not found: {src}")
            if dst.exists() and not overwrite: 
                return DownloadFileResult(code=-1, message=f"Destination exists: {dst}")
            if create_parent_dirs: 
                dst.parent.mkdir(parents=True, exist_ok=True)

            size = await self._transfer_file(src, dst, chunk_size)
            if preserve_permissions: 
                self._copy_permissions(src, dst)
            return DownloadFileResult(code=0, message="Download successful", data=DownloadFileData(source_path=str(src), local_path=str(dst), size=size))
        except Exception as e:
            return DownloadFileResult(code=-1, message=str(e))

    async def download_file_stream(
            self, source_path: str, local_path: str, *, overwrite: bool = False, create_parent_dirs: bool = True,
            preserve_permissions: bool = True, chunk_size: int = 1024 * 1024, options: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[DownloadFileStreamResult]:
        """Asynchronous file download streaming."""
        try:
            src = self._resolve_path(source_path)
            dst = pathlib.Path(local_path).expanduser().resolve()
            if not src.is_file(): 
                yield DownloadFileStreamResult(code=-1, message=f"Source not found: {src}")
                return
            if dst.exists() and not overwrite: 
                yield DownloadFileStreamResult(code=-1, message=f"Destination exists: {dst}")
                return
            if create_parent_dirs: 
                dst.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(src, mode="rb") as src_f, aiofiles.open(dst, mode="wb") as dst_f:
                index = 0
                while True:
                    chunk_bytes = await src_f.read(chunk_size)
                    if not chunk_bytes: break
                    await dst_f.write(chunk_bytes)
                    yield DownloadFileStreamResult(code=0, message="success", data=DownloadFileChunkData(
                        source_path=str(src), local_path=str(dst), chunk_size=len(chunk_bytes), chunk_index=index, is_last_chunk=False
                    ))
                    index += 1

            if preserve_permissions: 
                self._copy_permissions(src, dst)
        except Exception as e:
            yield DownloadFileStreamResult(code=-1, message=str(e))

    async def list_files(
            self, path: str, *, recursive: bool = False, max_depth: Optional[int] = None,
            sort_by: Literal['name', 'modified_time', 'size'] = "name", sort_descending: bool = False,
            file_types: Optional[List[str]] = None, options: Optional[Dict[str, Any]] = None
    ) -> ListFilesResult:
        """Asynchronously list files."""
        try:
            items = await self._list_items_internal(path, include_files=True, include_dirs=False, recursive=recursive, max_depth=max_depth, sort_by=sort_by, sort_descending=sort_descending, file_types=file_types)
            return ListFilesResult(code=0, message="List files successful", data=FileSystemData(total_count=len(items), list_items=items, root_path=str(self._resolve_path(path)), recursive=recursive, max_depth=max_depth))
        except Exception as e:
            return ListFilesResult(code=-1, message=str(e))

    async def list_directories(
            self, path: str, *, recursive: bool = False, max_depth: Optional[int] = None,
            sort_by: Literal['name', 'modified_time', 'size'] = "name", sort_descending: bool = False,
            options: Optional[Dict[str, Any]] = None
    ) -> ListDirsResult:
        """Asynchronously list directories."""
        try:
            items = await self._list_items_internal(path, include_files=False, include_dirs=True, recursive=recursive, max_depth=max_depth, sort_by=sort_by, sort_descending=sort_descending)
            return ListDirsResult(code=0, message="List directories successful", data=FileSystemData(total_count=len(items), list_items=items, root_path=str(self._resolve_path(path)), recursive=recursive, max_depth=max_depth))
        except Exception as e:
            return ListDirsResult(code=-1, message=str(e))

    async def search_files(self, path: str, pattern: str, exclude_patterns: Optional[List[str]] = None) -> SearchFilesResult:
        """Asynchronously search files using consolidated walker and item creator."""
        try:
            base = self._resolve_path(path)
            if not base.is_dir(): 
                return SearchFilesResult(code=-1, message=f"Path is not a directory: {base}")

            matched_paths = list(base.rglob(pattern))
            if exclude_patterns:
                exclude_set = set()
                for pat in exclude_patterns:
                    exclude_set.update(set(base.rglob(pat)))
                matched_paths = [p for p in matched_paths if p not in exclude_set]

            items = [item for p in matched_paths if p.is_file() and (item := self._create_fs_item(p))]
            return SearchFilesResult(code=0, message="Search successful", data=SearchFilesData(total_matches=len(items), matching_files=items, search_path=str(base), search_pattern=pattern, exclude_patterns=exclude_patterns))
        except Exception as e:
            return SearchFilesResult(code=-1, message=str(e))

    def _resolve_path(self, path: str, create_parent: bool = False) -> pathlib.Path:
        """
        Resolve path relative to work_dir and check for traversal.
        """
        if not self._run_config or not hasattr(self._run_config, 'work_dir'):
            raise ValueError("Local work configuration missing or invalid")

        work_dir = pathlib.Path(self._run_config.work_dir).expanduser().resolve()
        p = pathlib.Path(path).expanduser()
        if p.is_absolute():
            target_path = p.resolve()
        else:
            target_path = (work_dir / path).resolve()

        # Check for path traversal
        try:
            target_path.relative_to(work_dir)
        except ValueError:
            raise ValueError(f"Access denied: Path {path} is outside working directory {work_dir}")

        if create_parent:
            target_path.parent.mkdir(parents=True, exist_ok=True)

        return target_path

    def _apply_permissions(self, path: pathlib.Path, permissions: str | int) -> None:
        """Apply octal permissions to path on Unix-like systems."""
        if os.name != "nt":
            try:
                perm_int = int(str(permissions), 8) if isinstance(permissions, str) else permissions
                os.chmod(path, perm_int)
            except Exception:
                pass

    def _copy_permissions(self, src: pathlib.Path, dst: pathlib.Path) -> None:
        """Copy permissions from src to dst on Unix-like systems."""
        if os.name != "nt":
            try:
                st = src.stat()
                os.chmod(dst, st.st_mode)
            except Exception:
                pass

    async def _transfer_file(self, src: pathlib.Path, dst: pathlib.Path, chunk_size: int) -> int:
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

    def _walk_path(self, base: pathlib.Path, recursive: bool = False, max_depth: Optional[int] = None) -> Iterator[pathlib.Path]:
        """Consolidated directory walker with recursion and depth control."""
        if not recursive:
            yield from base.iterdir()
            return

        if max_depth is None:
            yield from base.rglob("*")
            return

        # BFS/DFS with depth check for max_depth
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

    def _create_fs_item(self, p: pathlib.Path) -> Optional[FileSystemItem]:
        """Centralized FileSystemItem creation with stat error handling."""
        try:
            stat = p.stat()
            is_dir = p.is_dir()
            return FileSystemItem(
                name=p.name,
                path=str(p),
                size=stat.st_size,
                modified_time=str(datetime.fromtimestamp(stat.st_mtime)),
                is_directory=is_dir,
                type=p.suffix if not is_dir else None,
            )
        except Exception:
            return None

    def _sort_items(self, items: List[FileSystemItem], sort_by: str, reverse: bool) -> None:
        """Sort FS items by name, modified_time, or size."""
        if sort_by == "name":
            items.sort(key=lambda i: i.name, reverse=reverse)
        elif sort_by == "modified_time":
            items.sort(key=lambda i: i.modified_time, reverse=reverse)
        elif sort_by == "size":
            items.sort(key=lambda i: i.size, reverse=reverse)

    async def _list_items_internal(
            self, 
            path: str, 
            include_files: bool = True, 
            include_dirs: bool = True,
            recursive: bool = False,
            max_depth: Optional[int] = None,
            sort_by: str = "name",
            sort_descending: bool = False,
            file_types: Optional[List[str]] = None
    ) -> List[FileSystemItem]:
        """Core logic for listing files and directories."""
        base = self._resolve_path(path)
        if not base.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {base}")

        items = []
        for p in self._walk_path(base, recursive, max_depth):
            is_dir = p.is_dir()
            if not include_files and not is_dir: continue
            if not include_dirs and is_dir: continue
            if file_types and not is_dir and p.suffix not in file_types: continue
            
            item = self._create_fs_item(p)
            if item: items.append(item)

        self._sort_items(items, sort_by, sort_descending)
        return items