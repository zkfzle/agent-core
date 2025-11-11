#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import time
import pytest
import asyncio

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runner.drunner.remote_client.remote_agent import RemoteAgent
from openjiuwen.core.runner.drunner.server_adapter.agent_adapter import MqAgentAdapter
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.common.logging import logger
from openjiuwen.core.runner.runner_config import RunnerConfig, DistributedConfig, MessageQueueConfig, PulsarConfig, \
    DEFAULT_RUNNER_CONFIG


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires real Pulsar uv sync --extra pulsar")
class TestRunnerIntegration:
    def setup_method(self):
        # 保存原始方法
        self.original_handle_invoke = MqAgentAdapter.handle_invoke
        self.original_handle_stream = MqAgentAdapter.handle_stream

        # 替换为自定义方法
        async def mock_handle_invoke(self, inputs):
            return {"MOCK_INVOKE": "CUSTOM_RESPONSE"}

        async def mock_handle_stream(self, inputs):
            for i in range(3):
                yield {"MOCK_STREAM": f"chunk_{i}"}

        MqAgentAdapter.handle_invoke = mock_handle_invoke
        MqAgentAdapter.handle_stream = mock_handle_stream

        # 其余配置代码...
        pulsar_mq = RunnerConfig(
            distributed_mode=True,
            distributed_config=DistributedConfig(
                request_timeout=5.0,
                message_queue_config=MessageQueueConfig(
                    type="pulsar",
                    pulsar_config=PulsarConfig(
                        max_workers=8,
                        url="pulsar://localhost:6650",
                    ),
                )
            )
        )
        Runner.set_config(pulsar_mq)

    def teardown_method(self):
        # 恢复原始方法
        MqAgentAdapter.handle_invoke = self.original_handle_invoke
        MqAgentAdapter.handle_stream = self.original_handle_stream
        Runner.set_config(DEFAULT_RUNNER_CONFIG)

    async def test_agent_normal_lifecycle(self):
        """测试agent的正常生命周期：创建、调用、删除"""
        print("=== Test 0: Agent lifecycle ===")
        await Runner.start()
        weather_adapter = MqAgentAdapter(agent_id="weather-agent")
        weather_adapter.start()

        try:
            # 模拟client发请求
            client = RemoteAgent(agent_id="weather-agent")
            Runner.add_agent(agent_id="weather-agent", agent=client)

            # 1. 测试批式请求
            logger.info("=== Testing batch invoke ===")
            response = await Runner.run_agent("weather-agent", {"city": "London"})
            logger.info(f"Batch response: {response}")
            assert response is not None

            # 2. 测试流式响应
            logger.info("=== Testing stream response ===")
            chunks = []
            async for chunk in Runner.run_agent_streaming("weather-agent", {"city": "Paris"}):
                logger.info(f"Stream chunk received: {chunk}")
                chunks.append(chunk)

            assert len(chunks) > 0
            logger.info(f"Received {len(chunks)} chunks")

            # 3. 测试删除agent
            logger.info("=== Testing agent removal ===")
            Runner.remove_agent("weather-agent")

            # 4. 验证删除后调用抛出异常
            with pytest.raises(JiuWenBaseException) as e:
                await Runner.run_agent("weather-agent", {"city": "London"})
            assert e.value.error_code == StatusCode.AGENT_NOT_FOUND.code

        except Exception as e:
            logger.exception(f"Test failed with error: {e}")
            raise
        finally:
            # 确保清理资源
            await weather_adapter.stop()
            await Runner.stop()

    async def test_agent_request_cancellation(self):
        """测试请求取消（触发 CancelledError）发送消息到一个不存在的agent"""
        print("=== Test 1: Manual task cancellation ===")
        await Runner.start()

        try:
            client = RemoteAgent(agent_id="weather-agent2")
            Runner.add_agent(agent_id="weather-agent2", agent=client)

            async def long_running_request():
                """一个长时间运行的请求"""
                return await Runner.run_agent("weather-agent2", {"city": "London"})

            # 创建任务
            task = asyncio.create_task(long_running_request())

            # 等待一小段时间后取消
            await asyncio.sleep(0.1)

            task.cancel()

            # 验证任务被取消
            with pytest.raises(JiuWenBaseException) as e:
                await task
            assert e.value.error_code == StatusCode.REMOTE_AGENT_REQUEST_CANCELLED.code
        finally:
            await Runner.stop()

    async def test_agent_request_timeout(self):
        """测试请求超时（触发 TimeoutError）发送消息到一个不存在的agent"""
        print("=== Test 2: Request timeout ===")
        await Runner.start()
        try:
            client = RemoteAgent(agent_id="slow-agent")

            with pytest.raises(JiuWenBaseException) as e:
                await client.invoke({"test": "data"}, 0.1)
            assert e.value.error_code == StatusCode.REMOTE_AGENT_REQUEST_TIMEOUT.code

            logger.info("Request timed out as expected")
        finally:
            await Runner.stop()

    async def test_agent_runner_shutdown_cancels_clients(self):
        """验证 Runner 提前关闭时，未完成的 client 调用会收到 CancelledError"""
        print("=== Test 3: Runner shutdown cancels clients ===")
        await Runner.start()

        try:
            client = RemoteAgent(agent_id="slow-agent")
            Runner.add_agent(agent_id="slow-agent", agent=client)

            async def long_running_request():
                return await Runner.run_agent("slow-agent", {"city": "Berlin"})

            task = asyncio.create_task(long_running_request())

            # Runner 提前关闭
            await asyncio.sleep(0.1)
            await Runner.stop()

            # 验证：client 侧收到 CancelledError
            with pytest.raises(JiuWenBaseException) as e:
                await task
            # 如果关闭太快，请求发的时候reply已经是close则会收到cancel异常，如果collector已经创建被取消则报错runner stop
            assert e.value.error_code == StatusCode.RUNNER_STOPPED.code or e.value.error_code == StatusCode.REMOTE_AGENT_REQUEST_CANCELLED.code

            logger.info("Client received CancelledError as expected when Runner stopped")
        finally:
            pass

    async def test_agent_adapter_exception_propagation(self):
        """测试agenta dapter返回异常时错误信息正确传递给客户端"""
        print("=== Test 4: Adapter error propagation ===")
        await Runner.start()
        original_handler = MqAgentAdapter.handle_invoke

        # 模拟adapter抛出异常
        async def error_handler(self, inputs):
            raise JiuWenBaseException(
                error_code=111,
                message="ADAPTER_ERROR")

        MqAgentAdapter.handle_invoke = error_handler
        weather_adapter = MqAgentAdapter(agent_id="weather-agent")
        weather_adapter.start()

        try:
            client = RemoteAgent(agent_id="weather-agent")
            Runner.add_agent(agent_id="weather-agent", agent=client)

            # 验证客户端收到包含错误码和消息的异常
            with pytest.raises(JiuWenBaseException) as e:
                await Runner.run_agent("weather-agent", {"city": "London"})

            assert e.value.error_code == StatusCode.REMOTE_AGENT_PROCESS_ERROR.code
            assert "code: 111, message: ADAPTER_ERROR" in e.value.message
        finally:
            # 恢复原始handler
            MqAgentAdapter.handle_invoke = original_handler
            await weather_adapter.stop()
            await Runner.stop()

    async def test_agent_call_without_runner_start_should_raise_exception(self):
        """验证 Runner没有start应该报错"""
        print("=== Test 5: Runner not started ===")
        try:
            client = RemoteAgent(agent_id="slow-agent")
            Runner.add_agent(agent_id="slow-agent", agent=client)

            async def long_running_request():
                return await Runner.run_agent("slow-agent", {"city": "Berlin"})

            task = asyncio.create_task(long_running_request())
            with pytest.raises(JiuWenBaseException) as e:
                await task
            assert e.value.error_code == StatusCode.RUNNER_DISTRIBUTED_MODE_REQUIRED.code
        finally:
            await Runner.stop()

    @pytest.mark.skip(reason="Skip performance tests")
    async def test_concurrent_vs_sequential_performance_comparison(self):
        """对比并发调用和顺序调用的性能差异"""
        print("=== Test 6: Performance Comparison ===")
        await Runner.start()
        # 创建adapter和client
        weather_adapter = MqAgentAdapter(agent_id="perf-agent")
        weather_adapter.start()

        try:
            client = RemoteAgent(agent_id="perf-agent")
            Runner.add_agent(agent_id="perf-agent", agent=client)

            # 测试数据
            test_data = [{"city": f"City_{i}"} for i in range(10)]

            # 1. 顺序调用测试
            print("Testing sequential calls...")
            start_time = time.time()
            sequential_results = []
            for data in test_data:
                result = await Runner.run_agent("perf-agent", data)
                sequential_results.append(result)
            sequential_time = time.time() - start_time

            # 2. 并发调用测试 - 使用较小的并发批次
            print("Testing concurrent calls...")
            start_time = time.time()
            concurrent_results = await asyncio.gather(
                *[Runner.run_agent("perf-agent", data) for data in test_data]
            )
            concurrent_time = time.time() - start_time

            # 性能对比分析
            print(f"Sequential calls: {sequential_time:.3f}s for {len(test_data)} requests")
            print(f"Concurrent calls: {concurrent_time:.3f}s for {len(test_data)} requests")
            # 验证结果正确性
            assert len(sequential_results) == len(test_data)
            assert len(concurrent_results) == len(test_data)
            assert sequential_results == concurrent_results

            # 如果并发确实比顺序快，记录性能提升
            if concurrent_time < sequential_time:
                print(f"✓ Concurrent is {sequential_time / concurrent_time:.2f}x faster than sequential")
            else:
                print(
                    f"⚠ Concurrent is {concurrent_time / sequential_time:.2f}x slower than sequential (within acceptable range)")

        finally:
            await weather_adapter.stop()
            await Runner.stop()

    @pytest.mark.skip(reason="Skip performance tests")
    async def test_concurrent_streaming(self):
        """测试流式调用10次,每个调用返回5个chunk,并发和顺序调用的性能对比"""
        print("=== Test 9: Concurrent Streaming vs Regular Calls ===")
        await Runner.start()

        # 保存原始方法
        original_handle_stream = MqAgentAdapter.handle_stream

        # 模拟流式响应
        async def mock_handle_stream(self, inputs):
            for i in range(5):
                yield {"stream_chunk": i, "data": f"chunk_{i}_for_{inputs.get('city', 'unknown')}"}

        MqAgentAdapter.handle_stream = mock_handle_stream

        streaming_adapter = MqAgentAdapter(agent_id="streaming-agent")
        streaming_adapter.start()

        try:
            client = RemoteAgent(agent_id="streaming-agent")
            Runner.add_agent(agent_id="streaming-agent", agent=client)

            # 测试数据
            test_data = [{"city": f"StreamCity_{i}"} for i in range(10)]

            # 1. 顺序流式调用测试
            print("Testing sequential streaming calls...")
            start_time = time.time()
            sequential_chunks = []
            for data in test_data:
                chunk_count = 0
                async for chunk in Runner.run_agent_streaming("streaming-agent", data):
                    sequential_chunks.append(chunk)
                    chunk_count += 1
            sequential_time = time.time() - start_time

            # 2. 并发流式调用测试
            print("Testing concurrent streaming calls...")

            start_time = time.time()

            async def collect_streaming_chunks(data):
                chunks = []
                async for chunk in Runner.run_agent_streaming("streaming-agent", data):
                    chunks.append(chunk)
                return chunks

            concurrent_tasks = []
            for data in test_data:
                task = asyncio.create_task(collect_streaming_chunks(data))
                concurrent_tasks.append(task)

            concurrent_results = await asyncio.gather(*concurrent_tasks)
            concurrent_time = time.time() - start_time

            # 计算结果
            total_sequential_chunks = len(sequential_chunks)
            total_concurrent_chunks = sum(len(result) for result in concurrent_results)

            # 性能对比
            print(
                f"Sequential streaming: {sequential_time:.3f}s for {len(test_data)} requests, {total_sequential_chunks} chunks")
            print(
                f"Concurrent streaming: {concurrent_time:.3f}s for {len(test_data)} requests, {total_concurrent_chunks} chunks")
            print(f"Performance improvement: {sequential_time / concurrent_time:.2f}x faster")

            # 验证结果正确性
            assert total_sequential_chunks == len(test_data) * 5  # 每个请求5个chunk
            assert total_concurrent_chunks == len(test_data) * 5

            # 并发应该显著快于顺序调用
            assert sequential_time > concurrent_time, f"Sequential should be slower than concurrent"

            # 验证流内容正确性
            for result in concurrent_results:
                assert len(result) == 5, "Each streaming call should return 5 chunks"
                for chunk in result:
                    assert "stream_chunk" in chunk and "data" in chunk
        finally:
            # 恢复原始方法
            MqAgentAdapter.handle_stream = original_handle_stream
            await streaming_adapter.stop()
            await Runner.stop()
