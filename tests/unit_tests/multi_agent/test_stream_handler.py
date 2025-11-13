import unittest

from openjiuwen.core.multi_agent.stream.stream_handler import StreamHandler

DEFAULT_STREAM_FRAME_TIMEOUT = 300
STREAM_END_FRAME = "end_of_stream_frame"
STREAM_FRAME_TIMEOUT_KEY = "STREAM_FRAME_TIMEOUT_KEY"

class TestStreamHandler(unittest.IsolatedAsyncioTestCase):
    """StreamHandler 单元测试类"""

    async def asyncSetUp(self):
        """每个测试前的初始化"""
        # 保存原始环境变量，避免测试间相互影响
        self.original_env = STREAM_FRAME_TIMEOUT_KEY
        self.handler = StreamHandler()

    async def test_on_stream_when_running(self):
        """测试在运行状态下添加流式数据"""
        test_message = "test data"
        await self.handler.on_stream(test_message)

        # 验证数据被放入队列
        self.assertEqual(await self.handler._queue.get(), test_message)
        self.handler._queue.task_done()

    async def test_on_stream_when_stopped(self):
        """测试在停止状态下添加流式数据"""
        await self.handler.stop()
        test_message = "test data"

        # 停止状态下添加数据应被忽略
        with self.assertLogs(level='WARNING') as log:
            await self.handler.on_stream(test_message)
            self.assertIn("StreamHandler is not running", log.output[0])

        # 队列应保持为空
        self.assertTrue(self.handler._queue.empty())

    async def test_stream_output_normal_flow(self):
        """测试正常的流式输出流程"""
        test_messages = ["msg1", "msg2", STREAM_END_FRAME]

        # 预先添加测试数据
        for msg in test_messages:
            await self.handler._queue.put(msg)

        # 收集生成器输出
        output = []
        async for msg in self.handler.stream_output():
            output.append(msg)

        # 验证输出（应不包含结束帧）
        self.assertEqual(output, ["msg1", "msg2"])
        # 验证状态已停止
        self.assertFalse(self.handler.is_running())

    async def test_stop_cleanup(self):
        """测试停止时的资源清理"""
        # 添加一些测试数据
        test_messages = ["msg1", "msg2", "msg3"]
        for msg in test_messages:
            await self.handler._queue.put(msg)

        # 执行停止
        await self.handler.stop()

        # 验证状态
        self.assertFalse(self.handler.is_running())
        # 验证队列已清空
        self.assertTrue(self.handler._queue.empty())

    async def test_stop_already_stopped(self):
        """测试停止已停止的处理器"""
        await self.handler.stop()

        # 再次停止应产生警告
        with self.assertLogs(level='WARNING') as log:
            await self.handler.stop()
            self.assertIn("StreamHandler is not running, no need to stop", log.output[0])