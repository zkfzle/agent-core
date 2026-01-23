# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import pytest
import shutil
import tempfile
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.sys_operation.sys_operation import SysOperationCard, SysOperation
from openjiuwen.core.sys_operation.base import OperationMode
from openjiuwen.core.sys_operation.local.config import LocalWorkConfig
from openjiuwen.core.common.exception.codes import StatusCode


@pytest.fixture
def work_dir():
    # Create a temporary directory for tests
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_fs_operation_no_work_dir(work_dir):
    await Runner.start()
    try:
        # 1. Setup
        card_id = "test_fs_op"
        card = SysOperationCard(id=card_id, mode=OperationMode.LOCAL)
        add_res = Runner.resource_mgr.add_sys_operation(card)
        assert add_res.is_ok()
        assert add_res.msg().id == card_id
        sys_op = Runner.resource_mgr.get_sys_operation(card_id)
        # 2. Basic Write/Read

        file_name = work_dir + "/basics.txt"
        lines = [f"line{i}" for i in range(1, 6)]
        content = "\n".join(lines)
        write_res = await sys_op.fs().write_file(path=file_name, content=content, prepend_newline=False)
        assert write_res.code == StatusCode.SUCCESS.code

        read_res = await sys_op.fs().read_file(path=file_name)
        assert read_res.code == StatusCode.SUCCESS.code
        assert read_res.data.content == content
    finally:
        await Runner.stop()


@pytest.mark.asyncio
async def test_fs_operation_comprehensive(work_dir):
    await Runner.start()
    try:
        # 1. Setup
        card_id = "test_fs_op"
        card = SysOperationCard(id=card_id, mode=OperationMode.LOCAL, work_config=LocalWorkConfig(work_dir=work_dir))
        add_res = Runner.resource_mgr.add_sys_operation(card)
        assert add_res.is_ok()
        assert add_res.msg().id == card_id
        sys_op = Runner.resource_mgr.get_sys_operation(card_id)

        # 2. Basic Write/Read
        file_name = "basics.txt"
        lines = [f"line{i}" for i in range(1, 6)]
        content = "\n".join(lines)
        write_res = await sys_op.fs().write_file(path=file_name, content=content, prepend_newline=False)
        assert write_res.code == StatusCode.SUCCESS.code
        assert os.path.isabs(write_res.data.path)

        read_res = await sys_op.fs().read_file(path=file_name)
        assert read_res.code == StatusCode.SUCCESS.code
        assert read_res.data.content == content

        # 3. Slicing variants
        # Head 2
        res_head = await sys_op.fs().read_file(path=file_name, head=2)
        assert res_head.code == StatusCode.SUCCESS.code
        assert res_head.data.content == "\n".join(lines[:2])
        # Tail 2
        res_tail = await sys_op.fs().read_file(path=file_name, tail=2)
        assert res_tail.code == StatusCode.SUCCESS.code
        assert res_tail.data.content == "\n".join(lines[-2:])
        # Range (2, 4)
        res_range = await sys_op.fs().read_file(path=file_name, line_range=(2, 4))
        assert res_range.code == StatusCode.SUCCESS.code
        assert res_range.data.content == "\n".join(lines[1:4])

        # 4. Binary/Chunking
        bin_file = "test.bin"
        bin_data = b"binary_data"
        await sys_op.fs().write_file(path=bin_file, content=bin_data, mode="bytes")
        res_bin = await sys_op.fs().read_file(path=bin_file, mode="bytes", chunk_size=len(bin_data))
        assert res_bin.data.content == bin_data

        # 5. Test write with prepend_newline=True
        prepend_file = "prepend_test.txt"
        prepend_content = "Hello, World!"
        await sys_op.fs().write_file(path=prepend_file, content=prepend_content, prepend_newline=True)
        prepend_read = await sys_op.fs().read_file(path=prepend_file)
        assert prepend_read.code == StatusCode.SUCCESS.code
        assert prepend_read.data.content == "\n" + prepend_content

        # 6. Test list_directories
        # Create directory structure
        os.makedirs(os.path.join(work_dir, "dir1", "subdir1"), exist_ok=True)
        os.makedirs(os.path.join(work_dir, "dir2"), exist_ok=True)

        # workdir/
        # ├── basics.txt
        # ├── test.bin
        # ├── prepend_test.txt
        # ├── dir1/
        # │   └── subdir1/
        # └── dir2/

        # Non-recursive
        dirs_res = await sys_op.fs().list_directories(path=".")
        assert dirs_res.code == StatusCode.SUCCESS.code
        assert dirs_res.data.total_count >= 2
        dir_names = [item.name for item in dirs_res.data.list_items]
        assert "dir1" in dir_names
        assert "dir2" in dir_names

        # Recursive
        recursive_dirs_res = await sys_op.fs().list_directories(path=".", recursive=True)
        assert recursive_dirs_res.code == StatusCode.SUCCESS.code
        assert recursive_dirs_res.data.total_count >= 3  # dir1, dir2, dir1/subdir1

        # 7. Test list_files with parameters
        # Create test files with different extensions
        file_extensions = [".txt", ".md", ".py", ".json"]
        for ext in file_extensions:
            await sys_op.fs().write_file(path=f"test{ext}", content=f"Test {ext}")

        # Create nested files
        os.makedirs(os.path.join(work_dir, "nested"), exist_ok=True)
        for ext in file_extensions:
            await sys_op.fs().write_file(path=f"nested/test{ext}", content=f"Nested test {ext}")

        # Test sort_by='size'
        large_file = "large.txt"
        await sys_op.fs().write_file(path=large_file, content="x" * 1000)

        # workdir/
        # ├── basics.txt
        # ├── test.bin
        # ├── prepend_test.txt
        # ├── test.txt
        # ├── test.md
        # ├── test.py
        # ├── test.json
        # ├── large.txt
        # ├── dir1/
        # │   └── subdir1/
        # ├── dir2/
        # └── nested/
        #     ├── nested.txt
        #     ├── nested.md
        #     ├── nested.py
        #     └── nested.json

        sort_size_res = await sys_op.fs().list_files(path=".", sort_by="size", sort_descending=True)
        assert sort_size_res.code == StatusCode.SUCCESS.code
        assert sort_size_res.data.list_items[0].name == large_file

        # Test file_types filter
        txt_files_res = await sys_op.fs().list_files(path=".", file_types=[".txt"])
        assert txt_files_res.code == StatusCode.SUCCESS.code
        assert all(item.type == ".txt" for item in txt_files_res.data.list_items)

        # 8. Test comprehensive search scenarios
        # Create markdown files in different directories
        os.makedirs(os.path.join(work_dir, "docs"), exist_ok=True)
        os.makedirs(os.path.join(work_dir, "notes"), exist_ok=True)

        md_files = [
            ("README.md", "Main documentation"),
            ("docs/guide.md", "User guide"),
            ("docs/api.md", "API docs"),
            ("notes/SKILL.md", "Skill"),
        ]

        for rel_path, md_content in md_files:
            await sys_op.fs().write_file(path=rel_path, content=md_content)

        # workdir/
        # ├── basics.txt
        # ├── test.bin
        # ├── prepend_test.txt
        # ├── test.txt
        # ├── test.md
        # ├── test.py
        # ├── test.json
        # ├── large.txt
        # ├── README.md
        # ├── dir1/
        # │   └── subdir1/
        # ├── dir2/
        # ├── nested/
        # │   ├── nested.txt
        # │   ├── nested.md
        # │   ├── nested.py
        # │   └── nested.json
        # ├── docs/
        # │   ├── guide.md
        # │   └── api.md
        # └── notes/
        #     └── SKILL.md

        # Search all .md files
        all_md_res = await sys_op.fs().search_files(path=".", pattern="*.md")
        assert all_md_res.code == StatusCode.SUCCESS.code
        assert all_md_res.data.total_matches >= 4

        # Search in specific directory
        docs_md_res = await sys_op.fs().search_files(path="docs", pattern="*.md")
        assert docs_md_res.code == StatusCode.SUCCESS.code
        assert docs_md_res.data.total_matches >= 2

        # Search with exclude pattern
        exclude_docs_res = await sys_op.fs().search_files(path=".", pattern="*.md", exclude_patterns=["docs/*"])
        assert exclude_docs_res.code == StatusCode.SUCCESS.code
        assert exclude_docs_res.data.total_matches >= 2

        # 9. Test streaming operations
        # Create a large file
        stream_file = "stream_test.txt"
        stream_content = "\n".join([f"Line {i}" for i in range(100)])
        await sys_op.fs().write_file(path=stream_file, content=stream_content)

        # Test read_file_stream
        chunk_count = 0
        async for chunk_result in sys_op.fs().read_file_stream(path=stream_file):
            assert chunk_result.code == StatusCode.SUCCESS.code
            assert hasattr(chunk_result.data, "chunk_content")
            assert hasattr(chunk_result.data, "chunk_index")
            chunk_count += 1
        assert chunk_count > 0

        # 10. Upload/Download Success (Success cases)
        local_temp = tempfile.mkdtemp()
        try:
            up_src = os.path.join(local_temp, "up_src.txt")
            with open(up_src, "w") as f:
                f.write("uploaded stuff")
            up_res = await sys_op.fs().upload_file(local_path=up_src, target_path="uploaded.txt")
            assert up_res.code == StatusCode.SUCCESS.code
            assert os.path.exists(os.path.join(work_dir, "uploaded.txt"))

            dl_dst = os.path.join(local_temp, "dl_dst.txt")
            dl_res = await sys_op.fs().download_file(source_path=file_name, local_path=dl_dst)
            assert dl_res.code == StatusCode.SUCCESS.code
            with open(dl_dst, "r") as f:
                assert f.read() == content
        finally:
            shutil.rmtree(local_temp)

        # 11. Security & Error branches
        # Traversal
        trap_res = await sys_op.fs().read_file(path="../outside.txt")
        assert trap_res.code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
        # FNF
        fnf_res = await sys_op.fs().read_file(path="non_existent.txt")
        assert fnf_res.code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code

        # Filename with special characters
        special_char_file = "file-with@#$%^&*()!_+-=[]{}|;:,.<>?~+with spaces中文文件名.txt"
        rst = await sys_op.fs().write_file(path=special_char_file, content="Special char test", prepend_newline=False)

        assert rst.data.path == os.path.join(work_dir,
                                             "file-with____________-_________._____with_spaces中文文件名.txt")
        special_read = await sys_op.fs().read_file(path=special_char_file)
        assert special_read.code == StatusCode.SUCCESS.code
        assert special_read.data.content == "Special char test"

        # 12. Cleanup
        rem_res = Runner.resource_mgr.remove_sys_operation(sys_operation_id=card_id)
        assert rem_res.is_ok()
        assert rem_res.msg().id == card_id
    finally:
        await Runner.stop()
