# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import asyncio
import os
import shutil
import tempfile

import pytest
import pytest_asyncio

from openjiuwen.core.runner import Runner
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.sys_operation import SysOperationCard, OperationMode, LocalWorkConfig


@pytest.fixture
def work_dir():
    # Create a temporary directory for tests
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest_asyncio.fixture(name="sys_op")
async def sys_op_fixture(work_dir):
    await Runner.start()
    try:
        card_id = "test_fs_op"
        card = SysOperationCard(id=card_id, mode=OperationMode.LOCAL, work_config=LocalWorkConfig(work_dir=work_dir))
        add_res = Runner.resource_mgr.add_sys_operation(card)
        assert add_res.is_ok()
        op = Runner.resource_mgr.get_sys_operation(card_id)
        yield op
    finally:
        Runner.resource_mgr.remove_sys_operation(sys_operation_id=card_id)
        await Runner.stop()


@pytest.mark.asyncio
async def test_fs_read_write(sys_op, work_dir):
    """Test combined read and write operations."""
    # 1. Basic Write & Read
    file_name = "test_basics.txt"
    content = "Hello, world!\nLine 2"

    # Write
    write_res = await sys_op.fs().write_file(path=file_name, content=content, prepend_newline=False)
    assert write_res.code == StatusCode.SUCCESS.code

    # Read
    read_res = await sys_op.fs().read_file(path=file_name)
    assert read_res.code == StatusCode.SUCCESS.code
    assert read_res.data.content == content

    # 2. Append/Prepend
    append_file = "test_append.txt"
    await sys_op.fs().write_file(path=append_file, content="Initial", prepend_newline=False)
    res = await sys_op.fs().read_file(path=append_file)
    assert res.data.content == "Initial"
    # Prepend newline
    await sys_op.fs().write_file(path=append_file, content="Appended", mode="text", prepend_newline=True,
                                 append_newline=False)

    res = await sys_op.fs().read_file(path=append_file)
    assert res.data.content == "\nAppended"

    # 3. Binary
    bin_file = "test.bin"
    bin_data = b"\x00\x01\x02"
    await sys_op.fs().write_file(path=bin_file, content=bin_data, mode="bytes")
    read_bin = await sys_op.fs().read_file(path=bin_file, mode="bytes")
    assert read_bin.data.content == bin_data


@pytest.mark.asyncio
async def test_fs_list_search_non_blocking(sys_op, work_dir):
    # Setup many files
    for i in range(200):
        subdir = os.path.join(work_dir, f"dir_{i // 20}")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, f"file_{i}.txt"), "w") as f:
            f.write("x" * 1024)

    heartbeat_count = 0
    stop = False

    async def heartbeat():
        nonlocal heartbeat_count
        while not stop:
            heartbeat_count += 1
            await asyncio.sleep(0.01)  # 10ms

    hb_task = asyncio.create_task(heartbeat())

    try:
        start = asyncio.get_running_loop().time()

        list_res = await sys_op.fs().list_files(".", recursive=True)
        assert list_res.code == StatusCode.SUCCESS.code

        search_res = await sys_op.fs().search_files(".", "*.txt")
        assert search_res.code == StatusCode.SUCCESS.code

        elapsed = asyncio.get_running_loop().time() - start

    finally:
        stop = True
        await hb_task

    expected_ticks = elapsed / 0.01

    assert heartbeat_count >= expected_ticks * 0.3, (
        f"heartbeat too slow: {heartbeat_count} vs expected {expected_ticks}"
    )


@pytest.mark.asyncio
async def test_fs_security_and_streams(sys_op, work_dir):
    """Test error handling (security) and streams."""
    # 1. Security (Path Traversal)
    res = await sys_op.fs().read_file("../outside.txt")
    assert res.code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "Access denied" in res.message or "traverses outside" in res.message

    # 2. Streams
    stream_file = "stream.txt"
    await sys_op.fs().write_file(stream_file, "line1\nline2", prepend_newline=False)

    chunks = []
    async for chunk in sys_op.fs().read_file_stream(stream_file):
        assert chunk.code == StatusCode.SUCCESS.code
        chunks.append(chunk.data.chunk_content)

    assert chunks == ["line1", "line2"]


@pytest.mark.asyncio
async def test_fs_read_file_mutually_exclusive_params(sys_op, work_dir):
    """Test that mutually exclusive parameters cannot be specified simultaneously."""
    # Create a test file with multiple lines
    test_file = "multi_line.txt"
    content = "line1\nline2\nline3\nline4\nline5"
    await sys_op.fs().write_file(test_file, content, prepend_newline=False)

    # Test 1: head and tail cannot be specified together
    res = await sys_op.fs().read_file(path=test_file, head=2, tail=2)
    assert res.code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "cannot be specified simultaneously" in res.message
    assert "head" in res.message
    assert "tail" in res.message

    # Test 2: head and line_range cannot be specified together
    res = await sys_op.fs().read_file(path=test_file, head=-1, line_range=(2, 4))
    assert res.code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "cannot be specified simultaneously" in res.message
    assert "head" in res.message
    assert "line_range" in res.message

    # Test 3: tail and line_range cannot be specified together
    res = await sys_op.fs().read_file(path=test_file, tail=2, line_range=(2, -1))
    assert res.code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "cannot be specified simultaneously" in res.message
    assert "tail" in res.message
    assert "line_range" in res.message

    # Test 4: Test mutually exclusive parameters in read_file_stream
    chunks = []
    async for chunk in sys_op.fs().read_file_stream(path=test_file, head=-1, tail=2):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "cannot be specified simultaneously" in chunks[0].message
    assert "head" in chunks[0].message
    assert "tail" in chunks[0].message

    res = await sys_op.fs().read_file(path=test_file, head=0, tail=2)
    assert res.code == StatusCode.SUCCESS.code
    assert res.data.content == "line4\nline5"


@pytest.mark.asyncio
async def test_fs_read_file_negative_zero_params(sys_op, work_dir):
    """Test handling of negative and zero values for read parameters."""
    # Create a test file with multiple lines
    test_file = "multi_line.txt"
    content = "line1\nline2\nline3\nline4\nline5"
    await sys_op.fs().write_file(test_file, content, prepend_newline=False)

    # Test 1: Negative head value should return empty content
    res = await sys_op.fs().read_file(path=test_file, head=-5)
    assert res.code == StatusCode.SUCCESS.code
    assert res.data.content == ""

    # Test 2: Negative tail value should return empty content
    res = await sys_op.fs().read_file(path=test_file, tail=-5)
    assert res.code == StatusCode.SUCCESS.code
    assert res.data.content == ""

    # Test 3: Zero head value should be treated as not passed (return full content)
    res = await sys_op.fs().read_file(path=test_file, head=0)
    assert res.code == StatusCode.SUCCESS.code
    assert res.data.content == content

    # Test 4: Zero tail value should be treated as not passed (return full content)
    res = await sys_op.fs().read_file(path=test_file, tail=0)
    assert res.code == StatusCode.SUCCESS.code
    assert res.data.content == content

    # Test 5: Zero line_range should return empty content
    res = await sys_op.fs().read_file(path=test_file, line_range=(0, 0))
    assert res.code == StatusCode.SUCCESS.code
    assert res.data.content == ""

    # Test 6: Negative head in read_file_stream should return empty content
    chunks = []
    async for chunk in sys_op.fs().read_file_stream(path=test_file, head=-5):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].code == StatusCode.SUCCESS.code
    assert chunks[0].data.chunk_content == ""

    # Test 7: Negative tail in read_file_stream should return empty content
    chunks = []
    async for chunk in sys_op.fs().read_file_stream(path=test_file, tail=-5):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].code == StatusCode.SUCCESS.code
    assert chunks[0].data.chunk_content == ""

    # Test 8: Zero parameters in read_file_stream should be treated as not passed (return full content)
    chunks = []
    async for chunk in sys_op.fs().read_file_stream(path=test_file, head=0):
        chunks.append(chunk)
    assert len(chunks) == 5  # Should return all 5 lines
    assert chunks[0].code == StatusCode.SUCCESS.code
    assert chunks[0].data.chunk_content == "line1"


@pytest.mark.asyncio
async def test_fs_read_file_binary_mode_parameters(sys_op, work_dir):
    """Test that text mode only parameters are not allowed in binary mode."""
    # Create a test file
    test_file = "binary_test.txt"
    content = "Hello, world!\nLine 2"
    await sys_op.fs().write_file(test_file, content, prepend_newline=False)

    # Test 1: read_file with head in binary mode should fail
    res = await sys_op.fs().read_file(path=test_file, mode="bytes", head=2)
    assert res.code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "only supported in text mode" in res.message

    # Test 2: read_file with tail in binary mode should fail
    res = await sys_op.fs().read_file(path=test_file, mode="bytes", tail=2)
    assert res.code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "only supported in text mode" in res.message

    # Test 3: read_file with line_range in binary mode should fail
    res = await sys_op.fs().read_file(path=test_file, mode="bytes", line_range=(1, 2))
    assert res.code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "only supported in text mode" in res.message

    # Test 4: read_file_stream with head in binary mode should fail
    chunks = []
    async for chunk in sys_op.fs().read_file_stream(path=test_file, mode="bytes", head=2):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "only supported in text mode" in chunks[0].message

    # Test 5: read_file_stream with tail in binary mode should fail
    chunks = []
    async for chunk in sys_op.fs().read_file_stream(path=test_file, mode="bytes", tail=2):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "only supported in text mode" in chunks[0].message

    # Test 6: read_file_stream with line_range in binary mode should fail
    chunks = []
    async for chunk in sys_op.fs().read_file_stream(path=test_file, mode="bytes", line_range=(1, 2)):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].code == StatusCode.SYS_OPERATION_FS_EXECUTION_ERROR.code
    assert "only supported in text mode" in chunks[0].message
