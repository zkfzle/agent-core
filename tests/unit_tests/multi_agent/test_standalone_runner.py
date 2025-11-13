import unittest
from unittest.mock import Mock, AsyncMock

from openjiuwen.core.multi_agent.runner.standalone_runner import StandaloneRunner
from openjiuwen.core.multi_agent.runner.agent_message_queue import AgentMessageQueue
from openjiuwen.core.multi_agent.member_instance_manager import MemberInstanceManager
from openjiuwen.core.multi_agent.runner.agent_run_space import AgentRunSpace
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode


class TestStandaloneRunner(unittest.IsolatedAsyncioTestCase):
    """StandaloneRunner 单元测试类"""

    async def asyncSetUp(self):
        """每个测试前初始化"""
        self.runner = StandaloneRunner()
        # 替换部分内部属性为mock，简化测试
        self.runner._message_queue = AsyncMock(spec=AgentMessageQueue)
        self.runner._member_instance_manager = Mock(spec=MemberInstanceManager)
        self.runner._agent_run_space = AsyncMock(spec=AgentRunSpace)

    async def test_initialization(self):
        """测试初始化状态"""
        self.assertIsNone(self.runner._run_state)
        self.assertFalse(self.runner._stopped.is_set())
        self.assertEqual(len(self.runner._running_tasks), 0)
        self.assertIsNone(self.runner._background_exception)

    async def test_stop(self):
        """测试停止runner"""
        await self.runner.start()
        # 模拟运行中的任务
        mock_task = Mock()
        mock_task.done.return_value = False
        self.runner._running_tasks.add(mock_task)

        # 执行停止
        await self.runner.stop(immediately=True)

        # 验证状态
        self.assertTrue(self.runner._stopped.is_set())
        # 验证任务被取消
        mock_task.cancel.assert_called_once()
        # 验证run_space被停止
        self.assertIsNone(self.runner._run_state)

    async def test_send_message_before_start(self):
        """测试在未启动时发送消息抛出异常"""
        with self.assertRaises(JiuWenBaseException) as ctx:
            await self.runner.send_message(
                message="test",
                recipient="agent1",
                sender="user"
            )

        self.assertEqual(
            ctx.exception.error_code,
            StatusCode.MULTI_AGENT_RUNNER_NOT_STARTED.code
        )

    async def test_register_member(self):
        """测试注册成员"""
        member_id = "planner"
        member_class = Mock()
        config = Mock()

        # 测试方法链
        result = self.runner.register_member(member_id, member_class, config)
        self.assertIs(result, self.runner)  # 验证方法链返回自身

        # 验证注册调用
        self.runner._member_instance_manager.register_member_type.assert_called_once_with(
            member_id, member_class, config
        )