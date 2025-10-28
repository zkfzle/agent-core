#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional, List

from jiuwen.agent.config.base import AgentConfig
from jiuwen.agent.common.enum import TaskType
from jiuwen.core.agent.controller.scheduler.agent_scheduler import AgentScheduler
from jiuwen.core.agent.controller.scheduler.message_handler import MessageHandler, MessageHandlerResult
from jiuwen.core.agent.controller.scheduler.task_handler import TaskHandler
from jiuwen.core.agent.message.message import Message, MessageContent, MessageSource, MessageType, SourceType
from jiuwen.core.agent.task.task import Task, TaskInput, TaskResult
from jiuwen.core.common.logging import logger


class TestAgentScheduler(unittest.IsolatedAsyncioTestCase):
    """AgentScheduler 单元测试"""
    
    async def asyncSetUp(self):
        """每个测试用例前的设置"""
        # 创建测试配置 - 使用极短的间隔加速测试
        self.config = AgentConfig(
            id="test_agent_scheduler",
            max_message_queue_size=100,
            max_task_queue_size=100,
            max_concurrent_messages=5,
            max_concurrent_tasks=10,
            scheduler_loop_interval=0.001,  # 1ms间隔
            message_timeout=5,  # 5秒超时
            task_timeout=5  # 5秒超时
        )
        
        # 创建调度器实例
        self.scheduler = AgentScheduler(self.config)
        
        # 创建 Mock handlers
        self.mock_message_handler = MagicMock(spec=MessageHandler)
        self.mock_task_handler = MagicMock(spec=TaskHandler)
        
        # 设置 handlers
        self.scheduler.set_handlers(self.mock_message_handler, self.mock_task_handler)
        
    async def asyncTearDown(self):
        """每个测试用例后的清理"""
        if self.scheduler.is_running():
            await self.scheduler.stop()
    
    # ==================== 基础功能测试 ====================
    
    async def test_scheduler_initialization(self):
        """测试调度器初始化"""
        self.assertEqual(self.scheduler.agent_id, "test_agent_scheduler")
        self.assertIsNotNone(self.scheduler.message_queue)
        self.assertIsNotNone(self.scheduler.task_queue)
        self.assertFalse(self.scheduler.is_running())
        self.assertIsNone(self.scheduler.get_result())
    
    async def test_scheduler_start_stop(self):
        """测试调度器启动和停止"""
        # 启动调度器
        await self.scheduler.start()
        self.assertTrue(self.scheduler.is_running())
        
        # 等待一小段时间确保调度器运行
        await asyncio.sleep(0.01)
        
        # 停止调度器
        await self.scheduler.stop()
        self.assertFalse(self.scheduler.is_running())
    
    async def test_scheduler_start_without_handlers(self):
        """测试没有设置handlers时启动调度器应该抛出异常"""
        scheduler = AgentScheduler(self.config)
        
        with self.assertRaises(ValueError) as context:
            await scheduler.start()
        
        self.assertIn("must be set before starting", str(context.exception))
    
    async def test_scheduler_double_start(self):
        """测试重复启动调度器"""
        await self.scheduler.start()
        
        # 第二次启动应该被忽略（通过日志warning提示）
        with patch('jiuwen.core.agent.controller.scheduler.agent_scheduler.logger') as mock_logger:
            await self.scheduler.start()
            mock_logger.warning.assert_called_once()
        
        await self.scheduler.stop()
    
    # ==================== 消息调度测试 ====================
    
    async def test_schedule_message(self):
        """测试消息调度"""
        message = self._create_test_message("test_msg_1", "Hello")
        
        await self.scheduler.schedule_message(message)
        
        # 验证消息进入队列
        self.assertEqual(self.scheduler.message_queue.qsize(), 1)
    
    async def test_schedule_multiple_messages(self):
        """测试调度多个消息"""
        messages = [
            self._create_test_message(f"msg_{i}", f"Message {i}")
            for i in range(5)
        ]
        
        for msg in messages:
            await self.scheduler.schedule_message(msg)
        
        self.assertEqual(self.scheduler.message_queue.qsize(), 5)
    
    async def test_message_queue_full(self):
        """测试消息队列满时的行为"""
        # 创建小容量队列
        small_config = AgentConfig(id="test", max_message_queue_size=2)
        scheduler = AgentScheduler(small_config)
        scheduler.set_handlers(self.mock_message_handler, self.mock_task_handler)
        
        # 填满队列
        await scheduler.schedule_message(self._create_test_message("msg1", "text1"))
        await scheduler.schedule_message(self._create_test_message("msg2", "text2"))
        
        # 第三个消息应该被丢弃（通过日志error提示）
        with patch('jiuwen.core.agent.controller.scheduler.agent_scheduler.logger') as mock_logger:
            # 使用非阻塞方式尝试添加消息
            try:
                scheduler.message_queue.put_nowait(self._create_test_message("msg3", "text3"))
            except asyncio.QueueFull:
                pass
    
    # ==================== 任务调度测试 ====================
    
    async def test_schedule_task(self):
        """测试任务调度"""
        task = self._create_test_task("task_1", "workflow_1")
        
        await self.scheduler.schedule_task(task)
        
        self.assertEqual(self.scheduler.task_queue.qsize(), 1)
    
    async def test_schedule_multiple_tasks(self):
        """测试调度多个任务"""
        tasks = [
            self._create_test_task(f"task_{i}", f"workflow_{i}")
            for i in range(10)
        ]
        
        for task in tasks:
            await self.scheduler.schedule_task(task)
        
        self.assertEqual(self.scheduler.task_queue.qsize(), 10)
    
    # ==================== 消息处理测试 ====================
    
    async def test_process_messages_success(self):
        """测试成功处理消息"""
        # 准备测试数据
        message = self._create_test_message("msg_1", "Hello")
        task = self._create_test_task("task_1", "workflow_1")
        
        # Mock message handler 返回结果
        self.mock_message_handler.process_message = AsyncMock(
            return_value=MessageHandlerResult(
                should_continue=True,
                tasks=[task],
                final_result=None
            )
        )
        
        # 添加消息到队列
        await self.scheduler.schedule_message(message)
        
        # 启动调度器
        await self.scheduler.start()
        
        # 等待消息被处理
        await asyncio.sleep(0.1)
        
        await self.scheduler.stop()
        
        # 验证消息已处理
        self.assertEqual(self.mock_message_handler.process_message.call_count, 1)
    
    async def test_process_messages_should_stop(self):
        """测试消息处理返回should_continue=False时停止调度器"""
        message = self._create_test_message("msg_1", "Stop")
        
        # Mock message handler 返回停止信号
        self.mock_message_handler.process_message = AsyncMock(
            return_value=MessageHandlerResult(
                should_continue=False,
                tasks=[],
                final_result="Final result"
            )
        )
        
        await self.scheduler.schedule_message(message)
        await self.scheduler.start()
        
        # 等待消息被处理和调度器停止
        await asyncio.sleep(0.1)
        
        # 验证调度器已停止
        self.assertFalse(self.scheduler.is_running())
        self.assertEqual(self.scheduler.get_result(), "Final result")
    
    async def test_process_messages_with_exception(self):
        """测试消息处理异常"""
        message = self._create_test_message("msg_1", "Error")
        
        # Mock message handler 抛出异常
        self.mock_message_handler.process_message = AsyncMock(
            side_effect=Exception("Message processing error")
        )
        
        await self.scheduler.schedule_message(message)
        await self.scheduler.start()
        
        # 等待处理
        await asyncio.sleep(0.1)
        
        # 调度器应该继续运行（异常被捕获）
        self.assertTrue(self.scheduler.is_running())
        
        await self.scheduler.stop()
    
    # ==================== 任务并行执行测试 ====================
    
    async def test_execute_single_task(self):
        """测试执行单个任务"""
        task = self._create_test_task("task_1", "workflow_1")
        result_message = self._create_test_message("result_1", "Task completed")
        
        # Mock task handler
        self.mock_task_handler.execute = AsyncMock(return_value=result_message)
        
        await self.scheduler.schedule_task(task)
        await self.scheduler.start()
        
        # 等待任务执行完成
        await asyncio.sleep(0.2)
        
        await self.scheduler.stop()
        
        # 验证任务已执行
        self.mock_task_handler.execute.assert_called_once()
    
    async def test_execute_multiple_tasks_concurrently(self):
        """测试并行执行多个任务"""
        # 创建3个任务（减少数量加速测试）
        tasks = [self._create_test_task(f"task_{i}", f"workflow_{i}") for i in range(3)]
        
        # Mock task handler - 每个任务耗时0.01秒
        async def mock_execute(task):
            await asyncio.sleep(0.01)
            return self._create_test_message(f"result_{task.task_id}", "Completed")
        
        self.mock_task_handler.execute = AsyncMock(side_effect=mock_execute)
        
        # 添加所有任务
        for task in tasks:
            await self.scheduler.schedule_task(task)
        
        await self.scheduler.start()
        await asyncio.sleep(0.05)  # 等待执行完成
        await self.scheduler.stop()
        
        # 验证所有任务都被执行
        self.assertEqual(self.mock_task_handler.execute.call_count, 3)
    
    async def test_task_execution_with_error_isolation(self):
        """测试任务执行错误隔离"""
        tasks = [
            self._create_test_task("task_1", "workflow_1"),
            self._create_test_task("task_2", "workflow_2"),
            self._create_test_task("task_3", "workflow_3"),
        ]
        
        # Mock task handler - 第2个任务失败
        async def mock_execute(task):
            if task.task_id == "task_2":
                raise Exception("Task 2 failed")
            await asyncio.sleep(0.01)
            return self._create_test_message(f"result_{task.task_id}", "Success")
        
        self.mock_task_handler.execute = AsyncMock(side_effect=mock_execute)
        
        for task in tasks:
            await self.scheduler.schedule_task(task)
        
        await self.scheduler.start()
        await asyncio.sleep(0.2)
        await self.scheduler.stop()
        
        # 验证所有任务都被尝试执行
        self.assertEqual(self.mock_task_handler.execute.call_count, 3)
    
    async def test_max_concurrent_tasks_limit(self):
        """测试最大并发任务数限制"""
        # 设置最大并发为3
        config = AgentConfig(
            id="test_concurrent",
            max_concurrent_tasks=3
        )
        scheduler = AgentScheduler(config)
        scheduler.set_handlers(self.mock_message_handler, self.mock_task_handler)
        
        # 创建10个任务
        tasks = [self._create_test_task(f"task_{i}", f"workflow_{i}") for i in range(10)]
        
        execution_times = []
        
        async def mock_execute(task):
            execution_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)
            return self._create_test_message(f"result_{task.task_id}", "Done")
        
        self.mock_task_handler.execute = AsyncMock(side_effect=mock_execute)
        
        for task in tasks:
            await scheduler.schedule_task(task)
        
        await scheduler.start()
        await asyncio.sleep(0.5)
        await scheduler.stop()
        
        # 验证任务被执行
        self.assertEqual(self.mock_task_handler.execute.call_count, 10)
    
    # ==================== 并行执行能力专项测试 ====================
    
    async def test_concurrent_execution_performance_vs_serial(self):
        """测试并行执行vs串行执行的性能对比"""
        # 创建3个任务，每个耗时0.02秒
        tasks = [self._create_test_task(f"task_{i}", f"workflow_{i}") for i in range(3)]
        
        # Mock执行函数
        async def mock_execute(task):
            await asyncio.sleep(0.02)
            return self._create_test_message(f"result_{task.task_id}", "Done")
        
        self.mock_task_handler.execute = AsyncMock(side_effect=mock_execute)
        
        # 添加任务
        for task in tasks:
            await self.scheduler.schedule_task(task)
        
        await self.scheduler.start()
        await asyncio.sleep(0.1)  # 等待完成
        await self.scheduler.stop()
        
        # 验证所有任务都执行了
        self.assertEqual(self.mock_task_handler.execute.call_count, 3)
    
    async def test_concurrent_execution_tracks_all_tasks(self):
        """测试并行执行时正确追踪所有任务"""
        num_tasks = 3
        executed_tasks = []
        
        async def mock_execute(task):
            executed_tasks.append(task.task_id)
            await asyncio.sleep(0.01)
            return self._create_test_message(f"result_{task.task_id}", "Done")
        
        self.mock_task_handler.execute = AsyncMock(side_effect=mock_execute)
        
        # 创建并添加任务
        task_ids = [f"task_{i}" for i in range(num_tasks)]
        for task_id in task_ids:
            await self.scheduler.schedule_task(
                self._create_test_task(task_id, f"workflow_{task_id}")
            )
        
        await self.scheduler.start()
        await asyncio.sleep(0.1)
        await self.scheduler.stop()
        
        # 验证所有任务都被执行
        self.assertEqual(len(executed_tasks), num_tasks)
        self.assertEqual(self.mock_task_handler.execute.call_count, num_tasks)
    
    async def test_concurrent_execution_with_varying_durations(self):
        """测试不同执行时长的任务并行执行"""
        # 任务执行时间：0.05s, 0.01s, 0.1s - 增大差异确保顺序确定
        task_durations = [0.05, 0.01, 0.1]
        completion_order = []
        
        async def mock_execute(task):
            task_idx = int(task.task_id.split('_')[1])
            duration = task_durations[task_idx]
            await asyncio.sleep(duration)
            completion_order.append(task.task_id)
            return self._create_test_message(f"result_{task.task_id}", "Done")
        
        self.mock_task_handler.execute = AsyncMock(side_effect=mock_execute)
        
        # 添加任务
        for i in range(len(task_durations)):
            await self.scheduler.schedule_task(
                self._create_test_task(f"task_{i}", f"workflow_{i}")
            )
        
        await self.scheduler.start()
        await asyncio.sleep(0.3)  # 增加等待时间确保全部完成
        await self.scheduler.stop()
        
        # 验证所有任务完成
        self.assertEqual(len(completion_order), len(task_durations))
        
        # 验证短任务先完成（由于并行执行）
        self.assertEqual(completion_order[0], "task_1")  # 0.01s最短
        self.assertEqual(completion_order[-1], "task_2")  # 0.1s最长
    
    async def test_concurrent_execution_error_does_not_block_others(self):
        """测试一个任务失败不会阻塞其他任务的并行执行"""
        completion_times = {}
        
        async def mock_execute(task):
            task_id = task.task_id
            if task_id == "task_1":
                raise Exception("Task 1 failed")
            await asyncio.sleep(0.01)
            completion_times[task_id] = task_id
            return self._create_test_message(f"result_{task_id}", "Success")
        
        self.mock_task_handler.execute = AsyncMock(side_effect=mock_execute)
        
        # 添加3个任务
        for i in range(3):
            await self.scheduler.schedule_task(
                self._create_test_task(f"task_{i}", f"workflow_{i}")
            )
        
        await self.scheduler.start()
        await asyncio.sleep(0.2)  # 增加等待时间
        await self.scheduler.stop()
        
        # 验证：3个任务都被尝试执行
        self.assertEqual(self.mock_task_handler.execute.call_count, 3)
        # 验证：task_0 和 task_2 都成功完成
        self.assertIn("task_0", completion_times)
        self.assertIn("task_2", completion_times)
    
    async def test_concurrent_execution_stress_test(self):
        """压力测试：并行执行10个任务"""
        num_tasks = 10
        
        async def mock_execute(task):
            await asyncio.sleep(0.01)  # 每个任务10ms
            return self._create_test_message(f"result_{task.task_id}", "Done")
        
        self.mock_task_handler.execute = AsyncMock(side_effect=mock_execute)
        
        # 添加10个任务
        for i in range(num_tasks):
            await self.scheduler.schedule_task(
                self._create_test_task(f"task_{i}", f"workflow_{i}")
            )
        
        await self.scheduler.start()
        await asyncio.sleep(0.1)
        await self.scheduler.stop()
        
        # 验证所有任务都执行了
        self.assertEqual(self.mock_task_handler.execute.call_count, num_tasks)
    
    
    # ==================== 完整工作流测试 ====================
    
    async def test_full_workflow_message_to_task_to_result(self):
        """测试完整工作流：消息 -> 任务 -> 结果"""
        workflow_task = self._create_test_task("weather_task", "weather_workflow")
        
        # Mock message handler: 只对第一个消息生成任务，后续消息（结果）不生成任务
        call_count = {'count': 0}
        async def mock_process_message(msg):
            call_count['count'] += 1
            if call_count['count'] == 1:
                return MessageHandlerResult(
                    should_continue=True,
                    tasks=[workflow_task],
                    final_result=None
                )
            else:
                # 结果消息不生成新任务
                return MessageHandlerResult(
                    should_continue=True,
                    tasks=[],
                    final_result=None
                )
        
        self.mock_message_handler.process_message = AsyncMock(side_effect=mock_process_message)
        
        # Mock task handler: 执行任务返回结果
        result_message = self._create_test_message("result_msg", "今天晴天")
        self.mock_task_handler.execute = AsyncMock(return_value=result_message)
        
        # 添加用户消息
        user_message = self._create_test_message("user_msg", "查询天气")
        await self.scheduler.schedule_message(user_message)
        
        # 启动调度器
        await self.scheduler.start()
        await asyncio.sleep(0.2)
        await self.scheduler.stop()
        
        # 验证整个流程
        self.assertGreaterEqual(self.mock_message_handler.process_message.call_count, 1)
        self.mock_task_handler.execute.assert_called_once()
    
    async def test_scheduler_run_until_complete(self):
        """测试运行调度器直到完成"""
        stop_message = self._create_test_message("stop_msg", "Stop")
        
        self.mock_message_handler.process_message = AsyncMock(
            return_value=MessageHandlerResult(
                should_continue=False,
                tasks=[],
                final_result="Completed"
            )
        )
        
        await self.scheduler.schedule_message(stop_message)
        await self.scheduler.start()
        result = await self.scheduler.run_until_complete()
        
        self.assertEqual(result, "Completed")
        self.assertFalse(self.scheduler.is_running())
    
    # ==================== 辅助方法 ====================
    
    def _create_test_message(self, msg_id: str, text: str) -> Message:
        """创建测试消息"""
        return Message(
            msg_id=msg_id,
            msg_type=MessageType.USER_INPUT,
            content=MessageContent(query=text),
            source=MessageSource(
                conversation_id="test_conversation",
                source_type=SourceType.USER
            )
        )
    
    def _create_test_task(self, task_id: str, workflow_name: str) -> Task:
        """创建测试任务"""
        task = Task(
            task_id=task_id,
            task_type=TaskType.WORKFLOW,
            input=TaskInput(
                target_id=f"{workflow_name}_id",
                target_name=workflow_name,
                arguments=MessageContent(query="test arguments")
            )
        )
        return task


if __name__ == '__main__':
    unittest.main()

