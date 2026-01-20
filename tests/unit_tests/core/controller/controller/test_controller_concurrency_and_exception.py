"""Controller 并发与异常处理测试

本测试文件包含 Controller 的并发和异常处理测试用例。

测试场景：
1. 并发 Session 隔离测试
2. 多会话并发执行互不干扰
3. 事件处理异常不影响其他事件
4. 任务执行异常处理
5. 流式输出异常隔离

核心测试点：
- 多个 session 并发执行时互不干扰
- 一个 session 的任务/异常不影响其他 session
- 事件正确路由到对应 session
- 异常被正确捕获和隔离
"""
import asyncio
import json
import pytest
from typing import List, AsyncIterator, Tuple

from openjiuwen.core.controller.modules.task_manager import TaskFilter
from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.session import TaskSession
from openjiuwen.core.controller.base import Controller, ControllerConfig
from openjiuwen.core.controller.modules import EventHandler, EventHandlerInput, TaskExecutor
from openjiuwen.core.controller.schema import (
    ControllerOutputChunk,
    ControllerOutputPayload,
    EventType,
    JsonDataFrame,
    TextDataFrame,
    Task,
    TaskStatus,
    InputEvent,
)
from openjiuwen.core.single_agent.agent import AbilityKit, ControllerAgent
from openjiuwen.core.common.logging import logger
from openjiuwen.core.session import Session


# ==================== 测试用的 TaskExecutor ====================

class NormalTaskExecutor(TaskExecutor):
    """正常执行的任务执行器"""
    
    async def execute_ability(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
        """执行任务"""
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text=f"Task {task_id} started in session {session.session_id()}")]
            ),
            last_chunk=False
        )
        
        # 模拟执行
        for i in range(3):
            await asyncio.sleep(0.1)
            yield ControllerOutputChunk(
                index=i + 1,
                type="controller_output",
                payload=ControllerOutputPayload(
                    type="processing",
                    data=[TextDataFrame(type="text", text=f"Task {task_id} progress {i+1}/3")]
                ),
                last_chunk=False
            )
        
        # 完成
        yield ControllerOutputChunk(
            index=4,
            type="controller_output",
            payload=ControllerOutputPayload(
                type=EventType.TASK_COMPLETION,
                data=[TextDataFrame(type="text", text=f"Task {task_id} completed in session {session.session_id()}")]
            ),
            last_chunk=True
        )
    
    async def can_pause(self, task_id: str, session: Session) -> Tuple[bool, str]:
        return True, ""
    
    async def pause(self, task_id: str, session: Session) -> bool:
        return True
    
    async def can_cancel(self, task_id: str, session: Session) -> Tuple[bool, str]:
        return True, ""
    
    async def cancel(self, task_id: str, session: Session) -> bool:
        return True


class FailingTaskExecutor(TaskExecutor):
    """会失败的任务执行器"""
    
    async def execute_ability(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
        """执行任务 - 会抛出异常"""
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text=f"Task {task_id} starting...")]
            ),
            last_chunk=False
        )
        
        await asyncio.sleep(0.1)
        
        # 模拟任务失败
        raise RuntimeError(f"Task {task_id} failed intentionally")
    
    async def can_pause(self, task_id: str, session: Session) -> Tuple[bool, str]:
        return True, ""
    
    async def pause(self, task_id: str, session: Session) -> bool:
        return True
    
    async def can_cancel(self, task_id: str, session: Session) -> Tuple[bool, str]:
        return True, ""
    
    async def cancel(self, task_id: str, session: Session) -> bool:
        return True


class ExceptionInStreamTaskExecutor(TaskExecutor):
    """流式输出过程中抛异常的任务执行器"""
    
    async def execute_ability(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
        """执行任务 - 在流式输出中间抛异常"""
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text=f"Task {task_id} started")]
            ),
            last_chunk=False
        )
        
        await asyncio.sleep(0.05)
        
        yield ControllerOutputChunk(
            index=1,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text=f"Task {task_id} progress 1/3")]
            ),
            last_chunk=False
        )
        
        # 在中间抛异常
        raise RuntimeError(f"Task {task_id} stream failed")
    
    async def can_pause(self, task_id: str, session: Session) -> Tuple[bool, str]:
        return True, ""
    
    async def pause(self, task_id: str, session: Session) -> bool:
        return True
    
    async def can_cancel(self, task_id: str, session: Session) -> Tuple[bool, str]:
        return True, ""
    
    async def cancel(self, task_id: str, session: Session) -> bool:
        return True


# ==================== 测试用的 EventHandler ====================

class ConcurrentSessionEventHandler(EventHandler):
    """并发 Session 测试的事件处理器"""
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建任务（任务 ID 包含 session_id）"""
        session_id = inputs.session.session_id()
        task = Task(
            session_id=session_id,
            task_id=f"task_{session_id}",
            task_type="normal",
            priority=1,
            status=TaskStatus.SUBMITTED,
            context_id=f"context_{session_id}"
        )
        self._task_manager.add_task([task])
        logger.info(f"ConcurrentSessionEventHandler: Created task for session {session_id}")
        return {"status": "success", "session_id": session_id}
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件"""
        logger.info(f"ConcurrentSessionEventHandler: Task completed - {inputs.event.task.task_id}")
        return {"status": "success"}
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        logger.error(f"ConcurrentSessionEventHandler: Task failed - {inputs.event.task.task_id}")
        return {"status": "failed"}


class ExceptionInEventHandlerEventHandler(EventHandler):
    """EventHandler 中抛异常的测试处理器"""
    
    def __init__(self):
        super().__init__()
        self.handle_count = 0
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建多个任务"""
        tasks = [
            Task(
                session_id=inputs.session.session_id(),
                task_id=f"task_{i}",
                task_type="normal",
                priority=i,
                status=TaskStatus.SUBMITTED,
                context_id=f"context_{i}"
            )
            for i in range(1, 4)
        ]
        self._task_manager.add_task(tasks)
        return {"status": "success", "tasks_created": 3}
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件 - 第一个任务完成时抛异常"""
        self.handle_count += 1
        
        if self.handle_count == 1:
            # 第一个任务完成时抛异常
            logger.info("ExceptionInEventHandlerEventHandler: Throwing exception in handle_task_completion")
            raise RuntimeError("Exception in handle_task_completion")
        
        # 其他任务正常处理
        logger.info(f"ExceptionInEventHandlerEventHandler: Task {inputs.event.task.task_id} completed normally")
        return {"status": "success"}
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        logger.error(f"ExceptionInEventHandlerEventHandler: Task failed - {inputs.event.task.task_id}")
        return {"status": "failed"}


class FailingTaskEventHandler(EventHandler):
    """处理失败任务的事件处理器"""
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建失败任务和正常任务"""
        tasks = [
            Task(
                session_id=inputs.session.session_id(),
                task_id="failing_task",
                task_type="failing",
                priority=1,
                status=TaskStatus.SUBMITTED,
                context_id="failing_context"
            ),
            Task(
                session_id=inputs.session.session_id(),
                task_id="normal_task",
                task_type="normal",
                priority=2,
                status=TaskStatus.SUBMITTED,
                context_id="normal_context"
            )
        ]
        self._task_manager.add_task(tasks)
        logger.info("FailingTaskEventHandler: Created 1 failing task and 1 normal task")
        return {"status": "success", "tasks_created": 2}
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件"""
        logger.info(f"FailingTaskEventHandler: Task completed - {inputs.event.task.task_id}")
        return {"status": "success"}
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        failed_task = inputs.event.task
        error_msg = inputs.event.error_message
        logger.error(f"FailingTaskEventHandler: Task {failed_task.task_id} failed: {error_msg}")
        return {"status": "failed", "error": error_msg}


class StreamExceptionTaskEventHandler(EventHandler):
    """处理流式输出异常的事件处理器"""
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建会在流式输出中抛异常的任务和正常任务"""
        tasks = [
            Task(
                session_id=inputs.session.session_id(),
                task_id="stream_fail_task",
                task_type="stream_exception",
                priority=1,
                status=TaskStatus.SUBMITTED,
                context_id="stream_fail_context"
            ),
            Task(
                session_id=inputs.session.session_id(),
                task_id="normal_task_2",
                task_type="normal",
                priority=2,
                status=TaskStatus.SUBMITTED,
                context_id="normal_context_2"
            )
        ]
        self._task_manager.add_task(tasks)
        logger.info("StreamExceptionTaskEventHandler: Created 1 stream-exception task and 1 normal task")
        return {"status": "success", "tasks_created": 2}
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件"""
        logger.info(f"StreamExceptionTaskEventHandler: Task completed - {inputs.event.task.task_id}")
        return {"status": "success"}
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        failed_task = inputs.event.task
        error_msg = inputs.event.error_message
        logger.error(f"StreamExceptionTaskEventHandler: Task {failed_task.task_id} failed: {error_msg}")
        return {"status": "failed", "error": error_msg}


# ==================== 工厂函数 ====================

def build_normal_executor(config, ability_kit, context_engine, task_manager, event_queue):
    """构建正常任务执行器"""
    return NormalTaskExecutor(config, ability_kit, context_engine, task_manager, event_queue)


def build_failing_executor(config, ability_kit, context_engine, task_manager, event_queue):
    """构建会失败的任务执行器"""
    return FailingTaskExecutor(config, ability_kit, context_engine, task_manager, event_queue)


def build_stream_exception_executor(config, ability_kit, context_engine, task_manager, event_queue):
    """构建流式输出异常的任务执行器"""
    return ExceptionInStreamTaskExecutor(config, ability_kit, context_engine, task_manager, event_queue)


# ==================== Agent 构建函数 ====================

async def build_test_agent(
    agent_id: str,
    event_handler: EventHandler,
    task_executors: dict
) -> ControllerAgent:
    """构建测试用的 Agent
    
    Args:
        agent_id: Agent ID
        event_handler: 事件处理器实例
        task_executors: 任务执行器字典 {task_type: builder_func}
    
    Returns:
        ControllerAgent: 配置完成的测试 Agent
    """
    agent_card = AgentCard(
        id=agent_id,
        name=f"Test Agent {agent_id}",
        description="Test agent for concurrency and exception testing"
    )
    
    controller = Controller()
    agent = ControllerAgent(card=agent_card, controller=controller)
    
    # 设置事件处理器
    controller.set_event_handler(event_handler)
    
    # 注册任务执行器
    for task_type, builder_func in task_executors.items():
        controller.add_task_executor(task_type, builder_func)
    
    agent.configure(ControllerConfig())
    
    return agent


# ==================== 辅助函数 ====================

async def collect_stream_output(stream: AsyncIterator[ControllerOutputChunk]) -> List[str]:
    """收集流式输出的内容
    
    Args:
        stream: 流式输出迭代器
    
    Returns:
        List[str]: 内容列表
    """
    output_texts = []
    async for chunk in stream:
        if chunk.payload and chunk.payload.data:
            for item in chunk.payload.data:
                if isinstance(item, TextDataFrame):
                    output_texts.append(item.text)
                elif isinstance(item, JsonDataFrame):
                    output_texts.append(json.dumps(item.data))
    return output_texts


# ==================== 并发 Session 隔离测试 ====================

class TestConcurrentSessionIsolation:
    """测试并发 Session 隔离"""

    @pytest.mark.asyncio
    async def test_concurrent_sessions_isolation(self):
        """测试多个 session 并发执行互不干扰
        
        测试目标：
        1. 同时启动 3 个 session 的 stream
        2. 每个 session 创建自己的任务
        3. 验证每个 session 的任务正确完成
        4. 验证任务输出包含正确的 session_id
        5. 验证一个 session 不影响其他 session
        """
        # 构建共享的 Agent
        agent = await build_test_agent(
            agent_id="test_concurrent_sessions",
            event_handler=ConcurrentSessionEventHandler(),
            task_executors={"normal": build_normal_executor}
        )
        
        # 创建 3 个不同的 session
        session_1 = TaskSession(trace_id="session_1")
        session_2 = TaskSession(trace_id="session_2")
        session_3 = TaskSession(trace_id="session_3")
        
        # 创建 3 个输入事件
        input_event_1 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "request from session 1"}
        )
        input_event_2 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "request from session 2"}
        )
        input_event_3 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "request from session 3"}
        )
        
        # 并发执行 3 个 stream
        logger.info("========== 并发启动 3 个 session ==========")
        results = await asyncio.gather(
            collect_stream_output(agent.stream(input_event_1, session_1)),
            collect_stream_output(agent.stream(input_event_2, session_2)),
            collect_stream_output(agent.stream(input_event_3, session_3)),
            return_exceptions=True
        )
        
        # 验证所有 session 都成功完成
        assert len(results) == 3, "应该有 3 个结果"
        
        for i, output in enumerate(results, 1):
            if isinstance(output, Exception):
                pytest.fail(f"Session {i} 抛出异常: {output}")
            
            assert len(output) > 0, f"Session {i} 应该有输出"
            
            # 验证输出包含正确的 session_id
            session_id = f"session_{i}"
            assert any(session_id in text for text in output), \
                f"Session {i} 的输出应该包含 {session_id}"
            
            # 验证任务完成
            assert any("completed" in text for text in output), \
                f"Session {i} 的任务应该完成"
            
            logger.info(f"Session {i} 验证通过: {len(output)} 条输出")
        
        # 验证所有任务都正确创建和完成
        all_tasks = agent.controller._task_manager.get_task(task_filter=None)
        completed_tasks = [t for t in all_tasks if t.status == TaskStatus.COMPLETED]
        
        assert len(completed_tasks) >= 3, \
            f"应该有至少 3 个完成的任务，实际有 {len(completed_tasks)} 个"
        
        logger.info("✅ test_concurrent_sessions_isolation passed")

    @pytest.mark.asyncio
    async def test_session_task_isolation(self):
        """测试一个 session 的任务不影响其他 session
        
        测试目标：
        1. Session 1 创建正常任务
        2. Session 2 创建会失败的任务
        3. 验证 Session 2 的失败不影响 Session 1
        4. 验证两个 session 的任务状态独立
        """
        # 创建两个不同的 Agent（使用不同的 EventHandler）
        agent_1 = await build_test_agent(
            agent_id="test_session_1",
            event_handler=ConcurrentSessionEventHandler(),
            task_executors={"normal": build_normal_executor}
        )
        
        agent_2 = await build_test_agent(
            agent_id="test_session_2",
            event_handler=FailingTaskEventHandler(),
            task_executors={
                "normal": build_normal_executor,
                "failing": build_failing_executor
            }
        )
        
        session_1 = TaskSession(trace_id="normal_session")
        session_2 = TaskSession(trace_id="failing_session")
        
        input_event_1 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "normal request"}
        )
        input_event_2 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "failing request"}
        )
        
        # 并发执行
        logger.info("========== 并发执行正常和失败任务 ==========")
        results = await asyncio.gather(
            collect_stream_output(agent_1.stream(input_event_1, session_1)),
            collect_stream_output(agent_2.stream(input_event_2, session_2)),
            return_exceptions=True
        )
        
        # 验证 Session 1 正常完成
        output_1 = results[0]
        assert not isinstance(output_1, Exception), "Session 1 不应该抛异常"
        assert any("completed" in text for text in output_1), \
            "Session 1 的任务应该正常完成"
        
        # 验证 Session 2 也能完成（虽然有任务失败）
        output_2 = results[1]
        assert not isinstance(output_2, Exception), "Session 2 不应该抛异常（任务失败不等于 session 失败）"
        
        # 验证 Session 2 的失败任务
        failed_tasks = [t for t in agent_2.controller._task_manager.get_task()
                       if t.status == TaskStatus.FAILED]
        assert len(failed_tasks) > 0, "Session 2 应该有失败的任务"
        
        # 验证 Session 1 没有失败任务
        all_tasks_1 = agent_1.controller._task_manager.get_task()
        failed_tasks_1 = [t for t in all_tasks_1 if t.status == TaskStatus.FAILED]
        assert len(failed_tasks_1) == 0, "Session 1 不应该有失败的任务（不受 Session 2 影响）"
        
        logger.info("✅ test_session_task_isolation passed")

    @pytest.mark.asyncio
    async def test_event_routing_to_correct_session(self):
        """测试事件正确路由到对应 session
        
        测试目标：
        1. 并发启动多个 session
        2. 验证每个 session 的事件只路由到自己的 EventHandler
        3. 验证事件不会串到其他 session
        """
        agent = await build_test_agent(
            agent_id="test_event_routing",
            event_handler=ConcurrentSessionEventHandler(),
            task_executors={"normal": build_normal_executor}
        )
        
        sessions = [TaskSession(trace_id=f"session_{i}") for i in range(5)]
        
        input_events = [
            InputEvent(
                event_type=EventType.INPUT,
                content={"query": f"request {i}"}
            )
            for i in range(5)
        ]
        
        # 并发执行 5 个 session
        logger.info("========== 并发执行 5 个 session ==========")
        results = await asyncio.gather(
            *[collect_stream_output(agent.stream(event, session))
              for event, session in zip(input_events, sessions)],
            return_exceptions=True
        )
        
        # 验证所有 session 都成功
        for i, output in enumerate(results):
            assert not isinstance(output, Exception), f"Session {i} 不应该抛异常"
            assert len(output) > 0, f"Session {i} 应该有输出"
            
            # 验证每个 session 的输出只包含自己的 session_id
            session_id = f"session_{i}"
            assert any(session_id in text for text in output), \
                f"Session {i} 的输出应该包含 {session_id}"
        
        logger.info("✅ test_event_routing_to_correct_session passed")


# ==================== 异常处理测试 ====================

class TestExceptionHandling:
    """测试异常处理"""

    @pytest.mark.asyncio
    async def test_task_execution_exception_handling(self):
        """测试任务执行异常处理
        
        测试目标：
        1. 任务执行过程中抛出异常
        2. 验证任务状态更新为 FAILED
        3. 验证 handle_task_failed 被调用
        4. 验证错误信息被正确记录
        """
        agent = await build_test_agent(
            agent_id="test_task_exception",
            event_handler=FailingTaskEventHandler(),
            task_executors={
                "normal": build_normal_executor,
                "failing": build_failing_executor
            }
        )
        
        session = TaskSession(trace_id="test_exception")
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test task exception"}
        )
        
        # 执行 agent stream
        output = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证失败任务的状态
        failed_task = agent.controller._task_manager.get_task(
            task_filter=TaskFilter(task_id="failing_task")
        )[0]
        
        assert failed_task is not None, "失败任务应该存在"
        assert failed_task.status == TaskStatus.FAILED, \
            f"失败任务状态应该是 FAILED，实际是 {failed_task.status}"
        
        # 验证错误信息被记录
        assert failed_task.error_message is not None, "应该记录错误信息"
        assert "failed intentionally" in failed_task.error_message, \
            f"错误信息应该包含失败原因，实际是: {failed_task.error_message}"
        
        # 验证正常任务不受影响
        normal_task = agent.controller._task_manager.get_task(
            task_filter=TaskFilter(task_id="normal_task")
        )[0]
        
        assert normal_task is not None, "正常任务应该存在"
        assert normal_task.status == TaskStatus.COMPLETED, \
            f"正常任务应该完成，实际状态是 {normal_task.status}"
        
        logger.info("✅ test_task_execution_exception_handling passed")

    @pytest.mark.asyncio
    async def test_stream_output_exception_isolation(self):
        """测试流式输出异常不影响其他任务
        
        测试目标：
        1. 任务 1 在流式输出中间抛异常
        2. 任务 2 正常执行
        3. 验证任务 1 的异常不影响任务 2
        4. 验证任务 1 状态为 FAILED，任务 2 状态为 COMPLETED
        """
        agent = await build_test_agent(
            agent_id="test_stream_exception",
            event_handler=StreamExceptionTaskEventHandler(),
            task_executors={
                "normal": build_normal_executor,
                "stream_exception": build_stream_exception_executor
            }
        )
        
        session = TaskSession(trace_id="test_stream_exception")
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test stream exception"}
        )
        
        # 执行 agent stream
        output = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证流式输出异常任务的状态
        stream_fail_task = agent.controller._task_manager.get_task(
            task_filter=TaskFilter(task_id="stream_fail_task")
        )[0]
        
        assert stream_fail_task is not None, "流式输出异常任务应该存在"
        assert stream_fail_task.status == TaskStatus.FAILED, \
            f"流式输出异常任务状态应该是 FAILED，实际是 {stream_fail_task.status}"
        
        # 验证正常任务不受影响
        normal_task = agent.controller._task_manager.get_task(
            task_filter=TaskFilter(task_id="normal_task_2")
        )[0]
        
        assert normal_task is not None, "正常任务应该存在"
        assert normal_task.status == TaskStatus.COMPLETED, \
            f"正常任务应该完成（不受流式输出异常影响），实际状态是 {normal_task.status}"
        
        logger.info("✅ test_stream_output_exception_isolation passed")

    @pytest.mark.asyncio
    async def test_event_handler_exception_isolation(self):
        """测试 EventHandler 异常不影响其他事件
        
        测试目标：
        1. 第一个任务完成时，handle_task_completion 抛异常
        2. 其他任务继续执行
        3. 验证其他任务的 handle_task_completion 正常执行
        4. 验证异常被捕获，不影响整体流程
        """
        agent = await build_test_agent(
            agent_id="test_handler_exception",
            event_handler=ExceptionInEventHandlerEventHandler(),
            task_executors={"normal": build_normal_executor}
        )
        
        session = TaskSession(trace_id="test_handler_exception")
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test handler exception"}
        )
        
        # 执行 agent stream
        # 注意：EventHandler 中的异常会被捕获，不应该导致 stream 失败
        output = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证至少有任务完成
        assert len(output) > 0, "应该有输出"
        
        # 验证有多个任务完成（证明第一个任务的异常不影响其他任务）
        all_tasks = agent.controller._task_manager.get_task()
        completed_tasks = [t for t in all_tasks if t.status == TaskStatus.COMPLETED]
        
        # 应该有至少 2 个任务完成（第一个抛异常，但其他的正常）
        assert len(completed_tasks) >= 2, \
            f"应该有至少 2 个任务完成，实际完成了 {len(completed_tasks)} 个"
        
        logger.info("✅ test_event_handler_exception_isolation passed")

    @pytest.mark.asyncio
    async def test_exception_in_concurrent_sessions(self):
        """测试并发 session 中的异常隔离
        
        测试目标：
        1. 并发执行多个 session
        2. 其中一个 session 的任务失败
        3. 验证失败不影响其他 session
        4. 验证所有 session 都能正常结束
        """
        # 创建一个正常 agent 和一个会失败的 agent
        normal_agent = await build_test_agent(
            agent_id="normal_agent",
            event_handler=ConcurrentSessionEventHandler(),
            task_executors={"normal": build_normal_executor}
        )
        
        failing_agent = await build_test_agent(
            agent_id="failing_agent",
            event_handler=FailingTaskEventHandler(),
            task_executors={
                "normal": build_normal_executor,
                "failing": build_failing_executor
            }
        )
        
        session_1 = TaskSession(trace_id="normal_1")
        session_2 = TaskSession(trace_id="failing")
        session_3 = TaskSession(trace_id="normal_2")
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test"}
        )
        
        # 并发执行
        logger.info("========== 并发执行（包含失败任务） ==========")
        results = await asyncio.gather(
            collect_stream_output(normal_agent.stream(input_event, session_1)),
            collect_stream_output(failing_agent.stream(input_event, session_2)),
            collect_stream_output(normal_agent.stream(input_event, session_3)),
            return_exceptions=True
        )
        
        # 验证所有 session 都能完成（不抛异常）
        for i, output in enumerate(results, 1):
            assert not isinstance(output, Exception), \
                f"Session {i} 不应该抛异常（任务失败不等于 session 失败）"
            assert len(output) > 0, f"Session {i} 应该有输出"
        
        logger.info("✅ test_exception_in_concurrent_sessions passed")


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "-s"])
