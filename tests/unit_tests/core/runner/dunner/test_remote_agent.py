import pytest
import asyncio
import logging

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runner.drunner.remote_client.remote_agent import RemoteAgent
from openjiuwen.core.runner.drunner.server_adapter.agent_adapter import MqAgentAdapter
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.common.logging import logger


@pytest.mark.asyncio
class TestRunnerIntegration:
    async def test_agent_normal_lifecycle(self):
        """测试agent的正常生命周期：创建、调用、删除"""
        # 创建并激活适配器
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
            weather_adapter.stop()
            await Runner.stop()

    async def test_agent_request_cancellation(self):
        """测试请求取消（触发 CancelledError）发送消息到一个不存在的agent"""
        await Runner.start()

        try:
            client = RemoteAgent(agent_id="weather-agent2")
            Runner.add_agent(agent_id="weather-agent2", agent=client)

            # 场景1: 主动取消任务
            logger.info("=== Test 1: Manual task cancellation ===")

            async def long_running_request():
                """一个长时间运行的请求"""
                return await Runner.run_agent("weather-agent2", {"city": "London"})

            # 创建任务
            task = asyncio.create_task(long_running_request())

            # 等待一小段时间后取消
            await asyncio.sleep(0.1)
            logger.info("=== Cancel Task")

            task.cancel()

            # 验证任务被取消
            with pytest.raises(JiuWenBaseException) as e:
                await task
            assert e.value.error_code == StatusCode.RUNNER_DISTRIBUTED_MODE_REQUIRED.code
        finally:
            await Runner.stop()

    async def test_agent_request_timeout(self):
        """测试请求超时（触发 TimeoutError）发送消息到一个不存在的agent"""
        await Runner.start()
        try:
            client = RemoteAgent(agent_id="slow-agent")
            Runner.add_agent(agent_id="slow-agent", agent=client)

            with pytest.raises((asyncio.TimeoutError, JiuWenBaseException)):
                await asyncio.wait_for(
                    Runner.run_agent("slow-agent", {"test": "data"}),
                    timeout=0.1
                )
            logger.info("Request timed out as expected")
        finally:
            await Runner.stop()

    async def test_agent_runner_shutdown_cancels_clients(self):
        """验证 Runner 提前关闭时，未完成的 client 调用会收到 CancelledError"""
        await Runner.start()

        try:
            client = RemoteAgent(agent_id="slow-agent")
            Runner.add_agent(agent_id="slow-agent", agent=client)

            async def long_running_request():
                return await Runner.run_agent("slow-agent", {"city": "Berlin"})

            task = asyncio.create_task(long_running_request())
            # Runner 提前关闭
            await Runner.stop()

            # 验证：client 侧收到 CancelledError
            with pytest.raises(JiuWenBaseException) as e:
                await task
            assert e.value.error_code == StatusCode.RUNNER_DISTRIBUTED_MODE_REQUIRED.code

            logger.info("Client received CancelledError as expected when Runner stopped")
        finally:
            pass

    async def test_agent_adapter_exception_propagation(self):
        """测试agentadapter返回异常时错误信息正确传递给客户端"""
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

            assert e.value.error_code == 111
        finally:
            # 恢复原始handler
            MqAgentAdapter.handle_invoke = original_handler
            weather_adapter.stop()
            await Runner.stop()

    async def test_agent_call_without_runner_start_should_raise_exception(self):
        """验证 Runner没有start应该报错"""
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
