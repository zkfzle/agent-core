import pytest
import shutil
import tempfile
import os
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.sys_operation.sys_operation import SysOperationCard
from openjiuwen.core.sys_operation.base import OperationMode
from openjiuwen.core.sys_operation.local.config import LocalWorkConfig


@pytest.fixture
def work_dir():
    # Create a temporary directory for tests
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_sys_operation_file_flow(work_dir):
    print(f"workdir:{work_dir}")
    # 1. Start Runner
    await Runner.start()

    try:
        # 2. Prepare configuration
        card_id = "test_sys_op"
        config = LocalWorkConfig(work_dir=work_dir)
        card = SysOperationCard(
            id=card_id,
            mode=OperationMode.LOCAL,
            work_config=config
        )

        # 3. Add sys_operation
        add_result = Runner.resource_mgr.add_sys_operation(card)
        assert add_result is not None
        assert add_result.is_ok()
        assert add_result.msg() == card_id

        # Verify retrieval
        sys_op = Runner.resource_mgr.get_sys_operation(card_id)
        assert sys_op is not None

        # 4. Test File Operations

        # 4.1 Write File
        file_name = "test_file.txt"
        content = "Hello, SysOperation!"
        write_result = await sys_op.file.write_file(path=file_name, content=content, prepend_newline=False)
        assert write_result.code == 0
        assert write_result.data.path.endswith(file_name)
        assert write_result.data.size == len(content.encode('utf-8'))

        # Verify file exists on disk
        file_path = os.path.join(work_dir, file_name)
        assert os.path.exists(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            assert f.read() == content

        # 4.2 Read File
        read_result = await sys_op.file.read_file(path=file_name)
        assert read_result.code == 0
        assert read_result.data.content == content

        # 4.3 List Files
        list_result = await sys_op.file.list_files(path=".")
        assert list_result.code == 0
        assert list_result.data.total_count == 1
        assert list_result.data.list_items[0].name == file_name

        # 4.4 Search Files
        search_result = await sys_op.file.search_files(path=".", pattern="*.txt")
        assert search_result.code == 0
        assert search_result.data.total_matches == 1
        assert search_result.data.matching_files[0].name == file_name

        # 4.5 Subdirectory test
        sub_dir = "subdir"
        sub_file = "sub_test.txt"
        # Write file in subdir (should auto-create parent)
        await sys_op.file.write_file(
            path=os.path.join(sub_dir, sub_file),
            content="Sub content",
            create_if_not_exist=True
        )

        # List directories
        dirs_result = await sys_op.file.list_directories(path=".")
        assert dirs_result.code == 0
        assert dirs_result.data.total_count == 1
        assert dirs_result.data.list_items[0].name == sub_dir

        # 4.6 Path Traversal Security Check
        read_trap = await sys_op.file.read_file(path="../outside.txt")
        assert read_trap.code == -1
        assert "Access denied" in read_trap.message

        # 4.7 Streaming Read Test
        stream_results = []
        async for res in sys_op.file.read_file_stream(path=file_name, mode="text"):
            assert res.code == 0
            stream_results.append(res.data.chunk_content)
        assert "".join(stream_results) == content

        # 4.8 Exception Handling: FileExistsError Simulation
        # Try to upload to existing path without overwrite
        up_res = await sys_op.file.upload_file(local_path=os.path.join(work_dir, file_name), target_path=file_name,
                                               overwrite=False)
        assert up_res.code == -1
        assert "Target exists" in up_res.message

        # Try to download to existing path without overwrite
        dl_res = await sys_op.file.download_file(source_path=file_name, local_path=os.path.join(work_dir, file_name),
                                                 overwrite=False)
        assert dl_res.code == -1
        assert "Destination exists" in dl_res.message

        # 4.9 Text Slicing Tests
        multi_line_content = "line1\nline2\nline3\nline4\nline5"
        ml_file = "multi.txt"
        await sys_op.file.write_file(path=ml_file, content=multi_line_content, prepend_newline=False)

        # Head
        read_head = await sys_op.file.read_file(path=ml_file, head=2)
        assert read_head.code == 0
        assert read_head.data.content == "line1\nline2"

        # Tail
        read_tail = await sys_op.file.read_file(path=ml_file, tail=2)
        assert read_tail.code == 0
        assert read_tail.data.content == "line4\nline5"

        # Line Range
        read_range = await sys_op.file.read_file(path=ml_file, line_range=(2, 4))
        assert read_range.code == 0
        assert read_range.data.content == "line2\nline3\nline4"

        # 4.10 Binary Mode Tests
        bin_file = "test.bin"
        bin_content = b"\x00\x01\x02\x03\xff"
        await sys_op.file.write_file(path=bin_file, content=bin_content, mode="bytes")
        read_bin = await sys_op.file.read_file(path=bin_file, mode="bytes")
        assert read_bin.code == 0
        assert read_bin.data.content == bin_content

        # 4.11 Detailed Exception Tests
        # FileNotFound on read
        fnf_res = await sys_op.file.read_file(path="non_existent.txt")
        assert fnf_res.code == -1

        # IsADirectoryError on write
        test_dir = os.path.join(work_dir, "test_dir")
        os.mkdir(test_dir)
        is_dir_res = await sys_op.file.write_file(path="test_dir", content="Oops")
        assert is_dir_res.code == -1

        # NotADirectoryError on list
        not_dir_res = await sys_op.file.list_files(path=file_name)
        assert not_dir_res.code == -1

        # 5. Test remove_sys_operation
        remove_result = Runner.resource_mgr.remove_sys_operation(operation_id=card_id)
        assert remove_result.is_ok()
        assert remove_result.msg() == card_id

        # Verify removal
        removed_op = Runner.resource_mgr.get_sys_operation(card_id)
        assert removed_op is None
    finally:
        # 6. Stop Runner
        await Runner.stop()
