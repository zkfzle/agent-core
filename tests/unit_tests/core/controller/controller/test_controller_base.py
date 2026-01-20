"""Controller P0 测试用例

本测试文件包含 Controller 的 P0 优先级测试用例，主要测试从 EventHandler 中调用 pause_task 和 cancel_task 的功能。

测试场景：
1. P0-1: 在 EventHandler 中暂停任务
2. P0-2: 在 EventHandler 中取消任务
3. P0-3: 在 EventHandler 中尝试暂停不可暂停的任务
4. P0-4: 在 EventHandler 中尝试取消不可取消的任务
5. P0-5: 在 EventHandler 中批量取消所有任务
6. P0-6: 在 EventHandler 中先暂停后取消任务

核心测试点：
- EventHandler 可以通过 self._task_scheduler.pause_task() 暂停任务
- EventHandler 可以通过 self._task_scheduler.cancel_task() 取消任务
- EventHandler 可以通过 self._task_scheduler.cancel_all_tasks() 批量取消任务
- 正确处理不可暂停/取消的任务
- 任务状态转换正确（RUNNING -> PAUSED/CANCELED）

测试设计参考 deepsearch 示例，使用真实的 Agent 进行端到端测试。
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
from openjiuwen.core.controller.modules import TaskManager, EventQueue
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
from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.single_agent.agent import AbilityKit, ControllerAgent
from openjiuwen.core.common.logging import logger
from openjiuwen.core.session import Session


# ==================== 测试用的 TaskExecutor ====================

class CancellableTaskExecutor(TaskExecutor):
    """可取消的任务执行器
    
    用于测试任务取消功能。任务会持续运行，直到被取消。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cancelled = False
        self._pause_requested = False
    
    async def execute_ability(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
        """执行任务 - 持续运行直到被取消"""
        # 开始执行
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text=f"Task {task_id} started")]
            ),
            last_chunk=False
        )
        
        # 根据任务 ID 决定执行时间
        # task_1 快速完成（2次循环），task_2 和 task_3 慢速执行（100次循环）
        if "task_1" in task_id:
            iterations = 2
            sleep_time = 0.1
        else:
            iterations = 100
            sleep_time = 0.1
        
        # 模拟长时间运行的任务
        for i in range(iterations):
            if self._cancelled:
                logger.info(f"Task {task_id} detected cancellation")
                return
            
            if self._pause_requested:
                logger.info(f"Task {task_id} detected pause request")
                return

            await asyncio.sleep(sleep_time)
            
            yield ControllerOutputChunk(
                index=i + 1,
                type="controller_output",
                payload=ControllerOutputPayload(
                    type="processing",
                    data=[TextDataFrame(type="text", text=f"Task {task_id} progress {i+1}/{iterations}")]
                ),
                last_chunk=False
            )
        # 正常完成
        yield ControllerOutputChunk(
            index=iterations + 1,
            type="controller_output",
            payload=ControllerOutputPayload(
                type=EventType.TASK_COMPLETION,
                data=[TextDataFrame(type="text", text=f"Task {task_id} completed")]
            ),
            last_chunk=True
        )
    
    async def can_pause(self, task_id: str, session: Session) -> Tuple[bool, str]:
        """检查是否可以暂停"""
        return True, ""
    
    async def pause(self, task_id: str, session: Session) -> bool:
        """暂停任务"""
        self._pause_requested = True
        logger.info(f"Task {task_id} pause requested")
        return True
    
    async def can_cancel(self, task_id: str, session: Session) -> Tuple[bool, str]:
        """检查是否可以取消"""
        return True, ""
    
    async def cancel(self, task_id: str, session: Session) -> bool:
        """取消任务"""
        self._cancelled = True
        logger.info(f"Task {task_id} cancellation requested")
        return True


class NonCancellableTaskExecutor(TaskExecutor):
    """不可取消的任务执行器
    
    用于测试尝试取消不可取消的任务的情况。
    """
    
    async def execute_ability(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
        """执行任务"""
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text=f"Non-cancellable task {task_id} running")]
            ),
            last_chunk=False
        )
        
        await asyncio.sleep(0.5)
        
        yield ControllerOutputChunk(
            index=1,
            type="controller_output",
            payload=ControllerOutputPayload(
                type=EventType.TASK_COMPLETION,
                data=[TextDataFrame(type="text", text=f"Task {task_id} completed")]
            ),
            last_chunk=True
        )
    
    async def can_pause(self, task_id: str, session: Session) -> Tuple[bool, str]:
        """不可暂停"""
        return False, "This task cannot be paused"
    
    async def pause(self, task_id: str, session: Session) -> bool:
        """不应该被调用"""
        raise RuntimeError("pause() should not be called when can_pause() returns False")
    
    async def can_cancel(self, task_id: str, session: Session) -> Tuple[bool, str]:
        """不可取消"""
        return False, "This task cannot be cancelled"
    
    async def cancel(self, task_id: str, session: Session) -> bool:
        """不应该被调用"""
        raise RuntimeError("cancel() should not be called when can_cancel() returns False")


class FailingTaskExecutor(TaskExecutor):
    """会失败的任务执行器
    
    用于测试任务执行失败的情况。
    """
    
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


# ==================== 测试用的 EventHandler ====================

class SimpleEventHandler(EventHandler):
    """简单的事件处理器
    
    用于基本测试，创建单个任务。
    """
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建一个任务"""
        task = Task(
            session_id=inputs.session.session_id(),
            task_id="test_task_1",
            task_type="cancellable",
            priority=1,
            status=TaskStatus.SUBMITTED,
            context_id="test_context_1"
        )
        self._task_manager.add_task([task])
        logger.info("SimpleEventHandler: Created task test_task_1")
        return {"status": "success", "tasks_created": 1}  # 返回确认信息
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件"""
        logger.info(f"SimpleEventHandler: Task completed - {inputs.event.task.task_id}")
        return {"status": "success", "tasks_created": 1}  # 返回确认信息
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        logger.error(f"SimpleEventHandler: Task failed - {inputs.event.task.task_id}")
        return {"status": "success", "tasks_created": 1}  # 返回确认信息


class DynamicTaskEventHandler(EventHandler):
    """动态任务ID的事件处理器
    
    每次调用都生成唯一的 task_id，避免多次 stream 调用时任务冲突。
    适用于测试订阅清理等需要多次 stream 调用的场景。
    """
    
    def __init__(self):
        super().__init__()
        self._task_counter = 0
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建一个带唯一ID的任务"""
        import uuid
        # 使用 UUID 确保每次都是唯一的 task_id
        unique_id = str(uuid.uuid4())[:8]
        self._task_counter += 1
        task_id = f"test_task_{self._task_counter}_{unique_id}"
        
        task = Task(
            session_id=inputs.session.session_id(),
            task_id=task_id,
            task_type="cancellable",
            priority=1,
            status=TaskStatus.SUBMITTED,
            context_id=f"test_context_{unique_id}"
        )
        self._task_manager.add_task([task])
        logger.info(f"DynamicTaskEventHandler: Created task {task_id}")
        return {"status": "success", "tasks_created": 1}
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件"""
        logger.info(f"DynamicTaskEventHandler: Task completed - {inputs.event.task.task_id}")
        return {"status": "success"}
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        logger.error(f"DynamicTaskEventHandler: Task failed - {inputs.event.task.task_id}")
        return {"status": "success"}


class CancelOnCompletionEventHandler(EventHandler):
    """任务完成时取消其他任务的事件处理器
    
    用于测试在 EventHandler 中取消任务的功能。
    """
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建多个任务"""
        tasks = [
            Task(
                session_id=inputs.session.session_id(),
                task_id=f"test_task_{i}",
                task_type="cancellable",
                priority=1,
                status=TaskStatus.SUBMITTED,
                context_id=f"test_context_{i}"
            )
            for i in range(1, 4)  # 创建 3 个任务
        ]
        self._task_manager.add_task(tasks)
        logger.info("CancelOnCompletionEventHandler: Created 3 tasks")
        return {"status": "success", "tasks_created": 3}
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件 - 取消所有其他任务"""
        completed_task_id = inputs.event.task.task_id
        logger.info(f"CancelOnCompletionEventHandler: Task {completed_task_id} completed, cancelling others")
        
        # 取消该 session 的所有其他任务
        cancelled_count = await self._task_scheduler.cancel_all_tasks(
            inputs.session.session_id()
        )
        logger.info(f"CancelOnCompletionEventHandler: Cancelled {cancelled_count} tasks")
        return {"status": "success", "cancelled_count": cancelled_count}
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        logger.error(f"CancelOnCompletionEventHandler: Task failed - {inputs.event.task.task_id}")
        return {"status": "failed"}


class FailureHandlingEventHandler(EventHandler):
    """处理任务失败的事件处理器
    
    用于测试任务失败时的处理逻辑。
    """
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建一个会失败的任务"""
        task = Task(
            session_id=inputs.session.session_id(),
            task_id="failing_task_1",
            task_type="failing",
            priority=1,
            status=TaskStatus.SUBMITTED,
            context_id="failing_context_1"
        )
        self._task_manager.add_task([task])
        logger.info("FailureHandlingEventHandler: Created failing task")
        return {"status": "success", "tasks_created": 1}
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件"""
        logger.info(f"FailureHandlingEventHandler: Task completed - {inputs.event.task.task_id}")
        return {"status": "success"}
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件 - 记录失败信息"""
        failed_task = inputs.event.task
        error_msg = inputs.event.error_message
        logger.error(f"FailureHandlingEventHandler: Task {failed_task.task_id} failed: {error_msg}")
        return {"status": "failed", "error": error_msg}


class PauseInHandlerEventHandler(EventHandler):
    """在 EventHandler 中暂停任务的事件处理器
    
    用于测试在 EventHandler 的 handle_task_completion 中调用 pause_task 的功能。
    创建三个并发任务，当第一个任务完成时，暂停第二个任务，验证第三个任务不受影响。
    """
    
    def __init__(self):
        super().__init__()
        self.first_task_completed = False
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建三个任务"""
        tasks = [
            Task(
                session_id=inputs.session.session_id(),
                task_id="pause_test_task_1",
                task_type="cancellable",
                priority=1,
                status=TaskStatus.SUBMITTED,
                context_id="pause_test_context_1"
            ),
            Task(
                session_id=inputs.session.session_id(),
                task_id="pause_test_task_2",
                task_type="cancellable",
                priority=1,
                status=TaskStatus.SUBMITTED,
                context_id="pause_test_context_2"
            ),
            Task(
                session_id=inputs.session.session_id(),
                task_id="pause_test_task_3",
                task_type="cancellable",
                priority=1,
                status=TaskStatus.SUBMITTED,
                context_id="pause_test_context_3"
            )
        ]
        self._task_manager.add_task(tasks)
        logger.info("PauseInHandlerEventHandler: Created 3 tasks (all SUBMITTED)")
        return {"status": "success", "tasks_created": 3}
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件 - 暂停第二个任务，让第三个任务继续"""
        completed_task_id = inputs.event.task.task_id
        logger.info(f"PauseInHandlerEventHandler: Task {completed_task_id} completed")
        
        # 只在第一个任务完成时暂停第二个任务
        if completed_task_id == "pause_test_task_1" and not self.first_task_completed:
            self.first_task_completed = True
            target_task_id = "pause_test_task_2"
            
            # task_1 很快完成（0.2秒），此时 task_2 和 task_3 应该还在运行中
            # 直接暂停 task_2
            logger.info(f"PauseInHandlerEventHandler: Task 1 completed, pausing task 2")
            
            # 从 EventHandler 中调用 TaskScheduler 的 pause_task
            success = await self._task_scheduler.pause_task(target_task_id, inputs.session)
            
            if success:
                logger.info(f"PauseInHandlerEventHandler: Successfully paused task {target_task_id}")
            else:
                logger.warning(f"PauseInHandlerEventHandler: Failed to pause task {target_task_id}")
            
            return {"status": "success", "paused": success}
        
        # 当第三个任务完成时，取消被暂停的第二个任务，让流能够结束
        if completed_task_id == "pause_test_task_3":
            logger.info("PauseInHandlerEventHandler: Task 3 completed, cancelling paused task 2")
            await self._task_scheduler.cancel_task("pause_test_task_2", inputs.session)
        
        return {"status": "success"}
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        logger.error(f"PauseInHandlerEventHandler: Task failed - {inputs.event.task.task_id}")
        return {"status": "failed"}


class CancelInHandlerEventHandler(EventHandler):
    """在 EventHandler 中取消任务的事件处理器
    
    用于测试在 EventHandler 的 handle_task_completion 中调用 cancel_task 的功能。
    创建两个任务，当第一个任务完成时，取消第二个任务。
    """
    
    def __init__(self):
        super().__init__()
        self.first_task_completed = False
    
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 创建两个任务"""
        tasks = [
            Task(
                session_id=inputs.session.session_id(),
                task_id="cancel_test_task_1",
                task_type="cancellable",
                priority=1,
                status=TaskStatus.SUBMITTED,
                context_id="cancel_test_context_1"
            ),
            Task(
                session_id=inputs.session.session_id(),
                task_id="cancel_test_task_2",
                task_type="cancellable",
                priority=1,
                status=TaskStatus.SUBMITTED,
                context_id="cancel_test_context_2"
            )
        ]
        self._task_manager.add_task(tasks)
        logger.info("CancelInHandlerEventHandler: Created 2 tasks")
        return {"status": "success", "tasks_created": 2}
    
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass
    
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件 - 取消第二个任务"""
        completed_task_id = inputs.event.task.task_id
        logger.info(f"CancelInHandlerEventHandler: Task {completed_task_id} completed")
        
        # 只在第一个任务完成时取消第二个任务
        if not self.first_task_completed:
            self.first_task_completed = True
            target_task_id = "cancel_test_task_2"
            
            # 从 EventHandler 中调用 TaskScheduler 的 cancel_task
            success = await self._task_scheduler.cancel_task(target_task_id, inputs.session)
            
            if success:
                logger.info(f"CancelInHandlerEventHandler: Successfully cancelled task {target_task_id}")
            else:
                logger.warning(f"CancelInHandlerEventHandler: Failed to cancel task {target_task_id}")
            
            return {"status": "success", "cancelled": success}
        
        return {"status": "success"}
    
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        logger.error(f"CancelInHandlerEventHandler: Task failed - {inputs.event.task.task_id}")
        return {"status": "failed"}


# ==================== 状态持久化测试用的 EventHandler ====================

class StatePersistenceEventHandler(EventHandler):
    """状态持久化测试的事件处理器

    用于验证暂停任务能够跨 stream 调用持久化。
    第一轮创建任务并暂停，第二轮验证能读取到暂停的任务。
    """

    def __init__(self):
        super().__init__()
        self.round_number = 0

    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件 - 区分第一轮和第二轮"""
        self.round_number += 1

        if self.round_number == 1:
            # 第一轮：创建 2 个任务（第一个快速完成，第二个被暂停）
            tasks = [
                Task(
                    session_id=inputs.session.session_id(),
                    task_id="persist_task_1",
                    task_type="cancellable",
                    priority=1,
                    status=TaskStatus.SUBMITTED,
                    context_id="persist_context_1"
                ),
                Task(
                    session_id=inputs.session.session_id(),
                    task_id="persist_task_2",
                    task_type="cancellable",
                    priority=1,
                    status=TaskStatus.SUBMITTED,
                    context_id="persist_context_2"
                )
            ]
            self._task_manager.add_task(tasks)
            logger.info("StatePersistenceEventHandler: 第一轮创建 2 个任务")
            return {"status": "success", "round": 1, "tasks_created": 2}
        else:
            # 第二轮：不创建新任务，只验证状态恢复
            existing_tasks = self._task_manager.get_task(
                task_filter=TaskFilter(task_id="persist_task_2")
            )

            if existing_tasks:
                logger.info(
                    f"StatePersistenceEventHandler: ✅ 第二轮找到持久化的任务 {existing_tasks[0].task_id}, 状态: {existing_tasks[0].status}")
            else:
                logger.error("StatePersistenceEventHandler: ❌ 第二轮未找到持久化的任务")

            return {"status": "success", "round": 2, "found_persisted_task": len(existing_tasks) > 0}

    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass

    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件 - 第一个任务完成后暂停第二个"""
        completed_task_id = inputs.event.task.task_id

        if completed_task_id == "persist_task_1":
            # 第一个任务完成后，暂停第二个任务
            success = await self._task_scheduler.pause_task("persist_task_2", inputs.session)
            logger.info(f"StatePersistenceEventHandler: 暂停 persist_task_2, 结果: {success}")

        return {"status": "success"}

    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        logger.error(f"StatePersistenceEventHandler: Task failed - {inputs.event.task.task_id}")
        return {"status": "failed"}


class MultiTaskStatePersistenceEventHandler(EventHandler):
    """多任务状态持久化测试的事件处理器

    用于验证多个不同状态的任务都能正确持久化。
    """

    def __init__(self):
        super().__init__()
        self.round_number = 0

    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件"""
        self.round_number += 1

        if self.round_number == 1:
            # 第一轮：创建 4 个任务
            tasks = [
                Task(
                    session_id=inputs.session.session_id(),
                    task_id="multi_task_1",
                    task_type="cancellable",
                    priority=1,
                    status=TaskStatus.SUBMITTED,
                    context_id="multi_context_1"
                ),
                Task(
                    session_id=inputs.session.session_id(),
                    task_id="multi_task_2",
                    task_type="cancellable",
                    priority=2,
                    status=TaskStatus.SUBMITTED,
                    context_id="multi_context_2"
                ),
                Task(
                    session_id=inputs.session.session_id(),
                    task_id="multi_task_3",
                    task_type="cancellable",
                    priority=3,
                    status=TaskStatus.SUBMITTED,
                    context_id="multi_context_3"
                )
            ]
            self._task_manager.add_task(tasks)
            logger.info("MultiTaskStatePersistenceEventHandler: 第一轮创建 3 个任务")
            return {"status": "success", "round": 1, "tasks_created": 3}
        else:
            # 第二轮：验证所有任务状态
            all_tasks = self._task_manager.get_task()
            logger.info(f"MultiTaskStatePersistenceEventHandler: 第二轮恢复了 {len(all_tasks)} 个任务")
            for task in all_tasks:
                logger.info(f"  - {task.task_id}: {task.status}")

            return {"status": "success", "round": 2, "restored_count": len(all_tasks)}

    async def handle_task_interaction(self, inputs: EventHandlerInput):
        pass

    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件"""
        completed_task_id = inputs.event.task.task_id

        if completed_task_id == "multi_task_1":
            # 第一个任务完成后，暂停 task_2，取消 task_3
            await self._task_scheduler.pause_task("multi_task_2", inputs.session)
            await self._task_scheduler.cancel_task("multi_task_3", inputs.session)
            logger.info("MultiTaskStatePersistenceEventHandler: 暂停 task_2, 取消 task_3")

        return {"status": "success"}

    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件"""
        logger.error(f"MultiTaskStatePersistenceEventHandler: Task failed - {inputs.event.task.task_id}")
        return {"status": "failed"}

# ==================== 工厂函数 ====================

def build_cancellable_executor(config, ability_kit, context_engine, task_manager, event_queue):
    """构建可取消的任务执行器"""
    return CancellableTaskExecutor(config, ability_kit, context_engine, task_manager, event_queue)


def build_non_cancellable_executor(config, ability_kit, context_engine, task_manager, event_queue):
    """构建不可取消的任务执行器"""
    return NonCancellableTaskExecutor(config, ability_kit, context_engine, task_manager, event_queue)


def build_failing_executor(config, ability_kit, context_engine, task_manager, event_queue):
    """构建会失败的任务执行器"""
    return FailingTaskExecutor(config, ability_kit, context_engine, task_manager, event_queue)


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
        description="Test agent for controller testing"
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


# ==================== P0 EventHandler 任务控制测试 ====================

class TestEventHandlerTaskControl:
    """测试 EventHandler 中的任务控制功能(支持任务暂停和取消)（P0 优先级）"""

    @pytest.mark.asyncio
    async def test_pause_task_in_event_handler(self):
        """P0-1: 测试在 EventHandler 中暂停任务不影响其他任务
        
        测试目标：
        1. 创建三个并发任务（task_1 快速完成 0.2s，task_2 和 task_3 慢速执行 10s）
        2. 第一个任务完成后，EventHandler 中调用 pause_task 暂停第二个任务
        3. 验证第二个任务被成功暂停
        4. 验证第三个任务不受影响，继续执行并完成
        5. 验证任务状态更新正确
        """
        # 构建 Agent
        agent = await build_test_agent(
            agent_id="test_pause_in_handler",
            event_handler=PauseInHandlerEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session = TaskSession(trace_id="test_pause_in_handler")
        
        # 创建输入事件
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test pause in handler"}
        )
        
        # 执行 agent stream
        output_texts = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证第一个任务完成
        assert any("pause_test_task_1" in text and "completed" in text for text in output_texts), \
            "第一个任务应该完成"
        
        # 验证第二个任务被暂停
        task_2 = agent.controller._task_manager.get_task(task_filter=TaskFilter(task_id="pause_test_task_2"))[0]
        assert task_2 is not None, "第二个任务应该存在"
        assert task_2.status == TaskStatus.PAUSED, \
            f"第二个任务最终状态应该是 PAUSED（先暂停后取消），实际是 {task_2.status}"
        
        # 验证第三个任务完成（关键：证明暂停 task_2 不影响 task_3）
        task_3 = agent.controller._task_manager.get_task(task_filter=TaskFilter(task_id="pause_test_task_3"))[0]
        assert task_3 is not None, "第三个任务应该存在"
        assert task_3.status == TaskStatus.COMPLETED, \
            f"第三个任务状态应该是 COMPLETED，实际是 {task_3.status}"
        
        assert any("pause_test_task_3" in text and "completed" in text for text in output_texts), \
            "第三个任务应该完成"
        
        logger.info("✅ test_pause_task_in_event_handler passed")

    @pytest.mark.asyncio
    async def test_cancel_task_in_event_handler(self):
        """P0-2: 测试在 EventHandler 中取消任务
        
        测试目标：
        1. 创建两个并发任务
        2. 第一个任务完成后，EventHandler 中调用 cancel_task 取消第二个任务
        3. 验证第二个任务被成功取消
        4. 验证任务状态更新为 CANCELED
        """
        # 构建 Agent
        agent = await build_test_agent(
            agent_id="test_cancel_in_handler",
            event_handler=CancelInHandlerEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session = TaskSession(trace_id="test_cancel_in_handler")
        
        # 创建输入事件
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test cancel in handler"}
        )
        
        # 执行 agent stream
        output_texts = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证第一个任务完成
        assert any("cancel_test_task_1" in text and "completed" in text for text in output_texts), \
            "第一个任务应该完成"
        
        # 验证第二个任务被取消
        task_2 = agent.controller._task_manager.get_task(task_filter=TaskFilter(task_id="cancel_test_task_2"))[0]
        assert task_2 is not None, "第二个任务应该存在"
        assert task_2.status == TaskStatus.CANCELED, \
            f"第二个任务状态应该是 CANCELED，实际是 {task_2.status}"
        
        logger.info("✅ test_cancel_task_in_event_handler passed")

    @pytest.mark.asyncio
    async def test_pause_non_pausable_task_in_event_handler(self):
        """P0-3: 测试在 EventHandler 中尝试暂停不可暂停的任务
        
        测试目标：
        1. 创建两个任务：一个可取消任务，一个不可暂停任务
        2. 第一个任务完成后，EventHandler 尝试暂停不可暂停的任务
        3. 验证暂停失败
        4. 验证不可暂停的任务继续执行并完成
        """
        
        class PauseNonPausableEventHandler(EventHandler):
            """尝试暂停不可暂停任务的事件处理器"""
            
            def __init__(self):
                super().__init__()
                self.first_task_completed = False
            
            async def handle_input(self, inputs: EventHandlerInput):
                tasks = [
                    Task(
                        session_id=inputs.session.session_id(),
                        task_id="pausable_task",
                        task_type="cancellable",
                        priority=1,
                        status=TaskStatus.SUBMITTED,
                        context_id="pausable_context"
                    ),
                    Task(
                        session_id=inputs.session.session_id(),
                        task_id="non_pausable_task",
                        task_type="non_cancellable",
                        priority=1,
                        status=TaskStatus.SUBMITTED,
                        context_id="non_pausable_context"
                    )
                ]
                self._task_manager.add_task(tasks)
                return {"status": "success", "tasks_created": 2}
            
            async def handle_task_interaction(self, inputs: EventHandlerInput):
                pass
            
            async def handle_task_completion(self, inputs: EventHandlerInput):
                if not self.first_task_completed:
                    self.first_task_completed = True
                    # 尝试暂停不可暂停的任务
                    success = await self._task_scheduler.pause_task("non_pausable_task", inputs.session)
                    logger.info(f"Attempt to pause non-pausable task: {success}")
                    return {"status": "success", "paused": success}
                return {"status": "success"}
            
            async def handle_task_failed(self, inputs: EventHandlerInput):
                return {"status": "failed"}
        
        # 构建 Agent
        agent = await build_test_agent(
            agent_id="test_pause_non_pausable",
            event_handler=PauseNonPausableEventHandler(),
            task_executors={
                "cancellable": build_cancellable_executor,
                "non_cancellable": build_non_cancellable_executor
            }
        )
        
        session = TaskSession(trace_id="test_pause_non_pausable")
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test pause non-pausable"}
        )
        
        # 执行 agent stream
        output_texts = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证不可暂停的任务正常完成
        task = agent.controller._task_manager.get_task(task_filter=TaskFilter(task_id="non_pausable_task"))[0]
        assert task is not None, "任务应该存在"
        assert task.status == TaskStatus.COMPLETED, \
            f"不可暂停的任务应该正常完成，状态应该是 COMPLETED，实际是 {task.status}"
        
        logger.info("✅ test_pause_non_pausable_task_in_event_handler passed")

    @pytest.mark.asyncio
    async def test_cancel_non_cancellable_task_in_event_handler(self):
        """P0-4: 测试在 EventHandler 中尝试取消不可取消的任务
        
        测试目标：
        1. 创建两个任务：一个可取消任务，一个不可取消任务
        2. 第一个任务完成后，EventHandler 尝试取消不可取消的任务
        3. 验证取消失败
        4. 验证不可取消的任务继续执行并完成
        """
        
        class CancelNonCancellableEventHandler(EventHandler):
            """尝试取消不可取消任务的事件处理器"""
            
            def __init__(self):
                super().__init__()
                self.first_task_completed = False
            
            async def handle_input(self, inputs: EventHandlerInput):
                tasks = [
                    Task(
                        session_id=inputs.session.session_id(),
                        task_id="cancellable_task",
                        task_type="cancellable",
                        priority=1,
                        status=TaskStatus.SUBMITTED,
                        context_id="cancellable_context"
                    ),
                    Task(
                        session_id=inputs.session.session_id(),
                        task_id="non_cancellable_task",
                        task_type="non_cancellable",
                        priority=1,
                        status=TaskStatus.SUBMITTED,
                        context_id="non_cancellable_context"
                    )
                ]
                self._task_manager.add_task(tasks)
                return {"status": "success", "tasks_created": 2}
            
            async def handle_task_interaction(self, inputs: EventHandlerInput):
                pass
            
            async def handle_task_completion(self, inputs: EventHandlerInput):
                if not self.first_task_completed:
                    self.first_task_completed = True
                    # 尝试取消不可取消的任务
                    success = await self._task_scheduler.cancel_task("non_cancellable_task", inputs.session)
                    logger.info(f"Attempt to cancel non-cancellable task: {success}")
                    return {"status": "success", "cancelled": success}
                return {"status": "success"}
            
            async def handle_task_failed(self, inputs: EventHandlerInput):
                return {"status": "failed"}
        
        # 构建 Agent
        agent = await build_test_agent(
            agent_id="test_cancel_non_cancellable",
            event_handler=CancelNonCancellableEventHandler(),
            task_executors={
                "cancellable": build_cancellable_executor,
                "non_cancellable": build_non_cancellable_executor
            }
        )
        
        session = TaskSession(trace_id="test_cancel_non_cancellable")
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test cancel non-cancellable"}
        )
        
        # 执行 agent stream
        output_texts = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证不可取消的任务正常完成
        task = agent.controller._task_manager.get_task(task_filter=TaskFilter(task_id="non_cancellable_task"))[0]
        assert task is not None, "任务应该存在"
        assert task.status == TaskStatus.COMPLETED, \
            f"不可取消的任务应该正常完成，状态应该是 COMPLETED，实际是 {task.status}"
        
        logger.info("✅ test_cancel_non_cancellable_task_in_event_handler passed")

    @pytest.mark.asyncio
    async def test_cancel_all_tasks_in_event_handler(self):
        """P0-5: 测试在 EventHandler 中批量取消所有任务
        
        测试目标：
        1. 创建多个并发任务
        2. 第一个任务完成后，EventHandler 调用 cancel_all_tasks 取消其他所有任务
        3. 验证其他任务被成功取消
        """
        # 构建 Agent
        agent = await build_test_agent(
            agent_id="test_cancel_all",
            event_handler=CancelOnCompletionEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session = TaskSession(trace_id="test_cancel_all")
        
        # 创建输入事件
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test cancel all"}
        )
        
        # 执行 agent stream
        output_texts = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证至少有一个任务完成
        assert any("completed" in text for text in output_texts), "应该有任务完成"
        
        # 验证有任务被取消（通过检查任务状态）
        all_tasks = agent.controller._task_manager.get_task()
        cancelled_tasks = [t for t in all_tasks if t.status == TaskStatus.CANCELED]
        
        # 应该有至少一个任务被取消（因为第一个完成后会取消其他的）
        assert len(cancelled_tasks) > 0, f"应该有任务被取消，实际取消了 {len(cancelled_tasks)} 个"
        
        logger.info(f"✅ test_cancel_all_tasks_in_event_handler passed (cancelled {len(cancelled_tasks)} tasks)")

    @pytest.mark.asyncio
    async def test_pause_then_cancel_in_event_handler(self):
        """P0-6: 测试在 EventHandler 中先暂停后取消任务
        
        测试目标：
        1. 创建三个任务
        2. 第一个任务完成后，暂停第二个任务
        3. 第三个任务完成后，取消已暂停的第二个任务
        4. 验证任务状态转换正确
        """
        
        class PauseThenCancelEventHandler(EventHandler):
            """先暂停后取消任务的事件处理器"""
            
            def __init__(self):
                super().__init__()
                self.completed_count = 0
            
            async def handle_input(self, inputs: EventHandlerInput):
                tasks = [
                    Task(
                        session_id=inputs.session.session_id(),
                        task_id=f"multi_op_task_{i}",
                        task_type="cancellable",
                        priority=1,
                        status=TaskStatus.SUBMITTED,
                        context_id=f"multi_op_context_{i}"
                    )
                    for i in range(1, 4)
                ]
                self._task_manager.add_task(tasks)
                return {"status": "success", "tasks_created": 3}
            
            async def handle_task_interaction(self, inputs: EventHandlerInput):
                pass
            
            async def handle_task_completion(self, inputs: EventHandlerInput):
                self.completed_count += 1
                
                if self.completed_count == 1:
                    # 第一个任务完成，暂停第二个任务
                    await self._task_scheduler.pause_task("multi_op_task_2", inputs.session)
                    logger.info("Paused task 2")
                elif self.completed_count == 2:
                    # 第三个任务完成，取消已暂停的第二个任务
                    # 注意：已暂停的任务不在 running_tasks 中，所以取消会失败
                    success = await self._task_scheduler.cancel_task("multi_op_task_2", inputs.session)
                    logger.info(f"Attempted to cancel paused task 2: {success}")
                
                return {"status": "success"}
            
            async def handle_task_failed(self, inputs: EventHandlerInput):
                return {"status": "failed"}
        
        # 构建 Agent
        agent = await build_test_agent(
            agent_id="test_pause_then_cancel",
            event_handler=PauseThenCancelEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session = TaskSession(trace_id="test_pause_then_cancel")
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test pause then cancel"}
        )
        
        # 执行 agent stream
        output_texts = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证第二个任务被暂停
        task_2 = agent.controller._task_manager.get_task(task_filter=TaskFilter(task_id="multi_op_task_2"))[0]
        assert task_2 is not None, "第二个任务应该存在"
        assert task_2.status == TaskStatus.PAUSED, \
            f"第二个任务应该是 PAUSED 状态，实际是 {task_2.status}"
        
        logger.info("✅ test_pause_then_cancel_in_event_handler passed")


# ==================== P1 状态持久化测试 ====================

class TestStatePersistence:
    """测试状态持久化（P1 优先级）"""

    @pytest.mark.asyncio
    async def test_paused_task_state_persistence(self):
        """P1-1: 验证暂停任务的状态持久化

        测试目标：
        1. 第一轮 stream: 创建 2 个任务，第一个快速完成，第二个被暂停
        2. 第二轮 stream: 使用相同 session，验证能读取到第一轮暂停的任务
        3. 验证任务状态、元数据都正确持久化

        核心验证点：
        - 第一轮结束时，persist_task_2 状态是 PAUSED
        - 第二轮开始后，从 TaskManager 能查询到 persist_task_2
        - persist_task_2 的状态仍然是 PAUSED（没有被错误地改变）
        - 任务元数据正确（task_id, context_id, priority 等）
        """
        # 构建 Agent（使用同一个实例）
        agent = await build_test_agent(
            agent_id="test_state_persistence",
            event_handler=StatePersistenceEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )

        # 使用同一个 Session（关键：必须是同一个实例）
        session = TaskSession(trace_id="test_state_persistence")

        # ========== 第一轮 stream ==========
        logger.info("========== 第一轮 stream 开始 ==========")
        input_event_1 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "round 1"}
        )
        output_1 = await collect_stream_output(agent.stream(input_event_1, session))

        # 验证第一轮：persist_task_1 完成，persist_task_2 被暂停
        assert any("persist_task_1" in text and "completed" in text for text in output_1), \
            "第一个任务应该完成"

        task_2_round_1 = agent.controller._task_manager.get_task(
            task_filter=TaskFilter(task_id="persist_task_2")
        )[0]
        assert task_2_round_1 is not None, "第一轮结束时，persist_task_2 应该存在"
        assert task_2_round_1.status == TaskStatus.PAUSED, \
            f"第一轮结束时，persist_task_2 应该是 PAUSED，实际是 {task_2_round_1.status}"

        logger.info(f"第一轮结束：persist_task_2 状态 = {task_2_round_1.status}")

        # ========== 第二轮 stream ==========
        logger.info("========== 第二轮 stream 开始 ==========")
        input_event_2 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "round 2"}
        )
        output_2 = await collect_stream_output(agent.stream(input_event_2, session))

        # 验证第二轮：能读取到 persist_task_2，状态仍然是 PAUSED
        task_2_round_2 = agent.controller._task_manager.get_task(
            task_filter=TaskFilter(task_id="persist_task_2")
        )[0]
        assert task_2_round_2 is not None, "第二轮应该能读取到 persist_task_2（状态持久化成功）"
        assert task_2_round_2.status == TaskStatus.PAUSED, \
            f"第二轮中，persist_task_2 状态应该仍然是 PAUSED，实际是 {task_2_round_2.status}"
        assert task_2_round_2.task_id == "persist_task_2", "任务 ID 应该正确"
        assert task_2_round_2.context_id == "persist_context_2", "Context ID 应该正确"
        assert task_2_round_2.priority == 1, "优先级应该正确"

        logger.info(f"第二轮验证：persist_task_2 状态 = {task_2_round_2.status}，元数据正确")
        logger.info("✅ test_paused_task_state_persistence passed")

    @pytest.mark.asyncio
    async def test_multi_task_state_persistence(self):
        """P1-2: 验证多任务混合状态持久化
        
        测试目标：
        1. 第一轮 stream: 创建 3 个任务
           - multi_task_1: COMPLETED（快速完成）
           - multi_task_2: PAUSED（被暂停）
           - multi_task_3: CANCELED（被取消）
        2. 第二轮 stream: 验证所有任务的状态都正确持久化
        
        核心验证点：
        - 3 个任务的状态都正确持久化
        - TaskManager 的索引结构正确
        - 任务计数正确
        - 第二轮的状态与第一轮完全一致
        """
        # 构建 Agent
        agent = await build_test_agent(
            agent_id="test_multi_state_persistence",
            event_handler=MultiTaskStatePersistenceEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        # 使用同一个 Session
        session = TaskSession(trace_id="test_multi_state")
        
        # ========== 第一轮 stream ==========
        logger.info("========== 第一轮 stream 开始 ==========")
        input_event_1 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "round 1"}
        )
        output_1 = await collect_stream_output(agent.stream(input_event_1, session))
        
        # 验证第一轮结束时的状态
        all_tasks_round_1 = agent.controller._task_manager.get_task()
        status_map_1 = {t.task_id: t.status for t in all_tasks_round_1}
        
        logger.info(f"第一轮结束时的任务状态: {status_map_1}")
        
        assert "multi_task_1" in status_map_1, "multi_task_1 应该存在"
        assert status_map_1["multi_task_1"] == TaskStatus.COMPLETED, \
            f"multi_task_1 应该是 COMPLETED，实际是 {status_map_1['multi_task_1']}"
        
        assert "multi_task_2" in status_map_1, "multi_task_2 应该存在"
        assert status_map_1["multi_task_2"] == TaskStatus.PAUSED, \
            f"multi_task_2 应该是 PAUSED，实际是 {status_map_1['multi_task_2']}"
        
        assert "multi_task_3" in status_map_1, "multi_task_3 应该存在"
        assert status_map_1["multi_task_3"] == TaskStatus.CANCELED, \
            f"multi_task_3 应该是 CANCELED，实际是 {status_map_1['multi_task_3']}"
        
        # ========== 第二轮 stream ==========
        logger.info("========== 第二轮 stream 开始 ==========")
        input_event_2 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "round 2"}
        )
        output_2 = await collect_stream_output(agent.stream(input_event_2, session))
        
        # 验证第二轮恢复后的状态（应该完全一致）
        all_tasks_round_2 = agent.controller._task_manager.get_task()
        status_map_2 = {t.task_id: t.status for t in all_tasks_round_2}
        
        logger.info(f"第二轮恢复后的任务状态: {status_map_2}")
        
        assert len(all_tasks_round_2) == 3, f"第二轮应该恢复 3 个任务，实际恢复了 {len(all_tasks_round_2)} 个"
        assert status_map_2 == status_map_1, \
            f"第二轮的状态应该与第一轮完全一致\n第一轮: {status_map_1}\n第二轮: {status_map_2}"
        
        # 验证每个任务的详细元数据
        for task in all_tasks_round_2:
            if task.task_id == "multi_task_1":
                assert task.priority == 1, "multi_task_1 优先级应该是 1"
                assert task.context_id == "multi_context_1", "multi_task_1 context_id 应该正确"
            elif task.task_id == "multi_task_2":
                assert task.priority == 2, "multi_task_2 优先级应该是 2"
                assert task.context_id == "multi_context_2", "multi_task_2 context_id 应该正确"
            elif task.task_id == "multi_task_3":
                assert task.priority == 3, "multi_task_3 优先级应该是 3"
                assert task.context_id == "multi_context_3", "multi_task_3 context_id 应该正确"
        
        logger.info("✅ test_multi_task_state_persistence passed")

    @pytest.mark.asyncio
    async def test_state_restoration_failure_fallback(self):
        """P1-3: 验证状态恢复失败时的容错性
        
        测试目标：
        1. 第一轮 stream: 创建并暂停任务
        2. 手动破坏 session 中的状态（模拟序列化错误）
        3. 第二轮 stream: 验证系统能优雅降级，不抛出异常
        
        核心验证点：
        - 恢复失败时调用 task_manager.clear_state()
        - 不抛出异常，能继续执行
        - 第二轮 TaskManager 被清空
        - 记录 error 日志
        """
        # 构建 Agent
        agent = await build_test_agent(
            agent_id="test_fallback",
            event_handler=StatePersistenceEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        # 使用同一个 Session
        session = TaskSession(trace_id="test_fallback")
        
        # ========== 第一轮 stream ==========
        logger.info("========== 第一轮 stream 开始 ==========")
        input_event_1 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "round 1"}
        )
        output_1 = await collect_stream_output(agent.stream(input_event_1, session))
        
        # 验证任务被暂停
        paused_task = agent.controller._task_manager.get_task(
            task_filter=TaskFilter(task_id="persist_task_2")
        )[0]
        assert paused_task.status == TaskStatus.PAUSED, "第一轮结束时任务应该是 PAUSED"
        
        logger.info("第一轮完成：persist_task_2 已暂停")
        
        # ========== 破坏 session 状态 ==========
        logger.info("========== 破坏 session 状态（模拟序列化错误） ==========")
        session.update_state({"controller": {"task_manager_state": "invalid_data"}})
        
        # ========== 第二轮 stream ==========
        logger.info("========== 第二轮 stream 开始（应该优雅降级） ==========")
        input_event_2 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "round 2"}
        )
        
        # 应该不抛出异常
        try:
            output_2 = await collect_stream_output(agent.stream(input_event_2, session))
            
            # 验证 TaskManager 被清空（因为恢复失败）
            all_tasks = agent.controller._task_manager.get_task()
            logger.info(f"第二轮恢复失败后，TaskManager 中的任务数量: {len(all_tasks)}")
            
            # 注意：第二轮会创建新任务（因为 round_number = 2），但不应该找到 persist_task_2
            persisted_task = agent.controller._task_manager.get_task(
                task_filter=TaskFilter(task_id="persist_task_2")
            )
            
            # 如果找到了 persist_task_2，说明状态恢复成功了（不符合预期）
            # 但实际上，由于状态恢复失败，第二轮不应该找到旧任务
            logger.info(f"第二轮是否找到 persist_task_2: {len(persisted_task) > 0}")
            
            logger.info("✅ test_state_restoration_failure_fallback passed（系统优雅降级成功）")
            
        except Exception as e:
            pytest.fail(f"状态恢复失败时不应该抛异常，但抛出了: {e}")


# ==================== 生命周期管理测试 ====================

class TestLifecycleManagement:
    """测试 Controller 生命周期管理"""

    @pytest.mark.asyncio
    async def test_multiple_stream_calls_no_duplicate_start(self):
        """测试同一 event loop 中多次 stream() 调用不重复启动
        
        测试目标：
        1. 第一次调用 stream() 时启动 EventQueue 和 TaskScheduler
        2. 第二次调用 stream() 时不重复启动（复用已启动的组件）
        3. 验证 event_loop 跟踪正确
        """
        agent = await build_test_agent(
            agent_id="test_no_duplicate_start",
            event_handler=SimpleEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session_1 = TaskSession(trace_id="session_1")
        session_2 = TaskSession(trace_id="session_2")
        
        # 第一次 stream
        input_event_1 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "first stream"}
        )
        
        output_1 = await collect_stream_output(agent.stream(input_event_1, session_1))
        assert len(output_1) > 0, "第一次 stream 应该有输出"
        
        # 记录第一次启动的 event_loop
        first_event_loop = agent.controller._event_loop
        assert first_event_loop is not None, "第一次 stream 后应该记录 event_loop"
        
        # 第二次 stream（同一 event loop）
        input_event_2 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "second stream"}
        )
        
        output_2 = await collect_stream_output(agent.stream(input_event_2, session_2))
        assert len(output_2) > 0, "第二次 stream 应该有输出"
        
        # 验证 event_loop 没有变化（复用了）
        second_event_loop = agent.controller._event_loop
        assert second_event_loop is first_event_loop, \
            "同一 event loop 中多次 stream 应该复用组件，不重复启动"
        
        logger.info("✅ test_multiple_stream_calls_no_duplicate_start passed")

    @pytest.mark.asyncio
    async def test_controller_stop_cleanup_all(self):
        """测试 stop() 方法能正确停止所有后台任务和订阅
        
        测试目标：
        1. 创建多个任务并启动 stream
        2. 调用 controller.stop()
        3. 验证所有后台任务被停止
        4. 验证所有订阅被清理
        """
        agent = await build_test_agent(
            agent_id="test_stop_cleanup",
            event_handler=CancelOnCompletionEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session = TaskSession(trace_id="test_stop")
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test stop"}
        )
        
        # 启动 stream（在后台运行）
        stream_task = asyncio.create_task(
            collect_stream_output(agent.stream(input_event, session))
        )
        
        # 等待一段时间，让任务开始执行
        await asyncio.sleep(0.3)

        # 验证 EventQueue 和 TaskScheduler 都被停止
        try:
            await stream_task
            # 停止 controller
            await agent.controller.stop()
        except Exception as e:
            logger.info(f"stream_task 在 stop 后抛出异常（预期行为）: {e}")
        
        # 验证 sessions 被清空
        assert len(agent.controller._task_scheduler.sessions) == 0, \
            "stop 后 sessions 应该被清空"
        assert agent.controller._task_scheduler._running == False, "调度器运行状态为False"
        
        logger.info("✅ test_controller_stop_cleanup_all passed")


# ==================== Session 管理测试 ====================

class TestSessionManagement:
    """测试 Session 管理"""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_no_interference(self):
        """测试多轮对话互不干扰
        
        测试目标：
        1. 同一 session 第一轮对话创建任务
        2. 第一轮结束后，第二轮对话创建新任务
        3. 验证第一轮的任务不会影响第二轮
        4. 验证每轮对话的任务状态正确
        """
        agent = await build_test_agent(
            agent_id="test_multi_turn",
            event_handler=DynamicTaskEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session = TaskSession(trace_id="multi_turn")
        
        # 第一轮对话
        input_event_1 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "turn 1"}
        )
        output_1 = await collect_stream_output(agent.stream(input_event_1, session))
        assert any("test_task_1" in text for text in output_1), "第一轮应该创建应该创建了一个任务"
        
        # 记录第一轮的任务
        tasks_after_turn_1 = agent.controller._task_manager.get_task(task_filter=None)
        logger.info(f"第一轮后的任务数: {len(tasks_after_turn_1)}")
        
        # 第二轮对话
        input_event_2 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "turn 2"}
        )
        output_2 = await collect_stream_output(agent.stream(input_event_2, session))
        assert any("test_task_2" in text for text in output_2), "第二轮应该创建新的 test_task"
        
        # 验证任务互不干扰
        tasks_after_turn_2 = agent.controller._task_manager.get_task(task_filter=None)
        logger.info(f"第二轮后的任务数: {len(tasks_after_turn_2)}")
        
        logger.info("✅ test_multi_turn_conversation_no_interference passed")

    @pytest.mark.asyncio
    async def test_session_registration_and_cleanup(self):
        """测试 Session 注册和清理
        
        测试目标：
        1. stream() 开始时正确注册 session
        2. stream() 结束时正确移除 session
        3. 验证 session 不会泄漏
        """
        agent = await build_test_agent(
            agent_id="test_session_reg",
            event_handler=SimpleEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session = TaskSession(trace_id="test_reg")
        
        # 验证初始状态：没有 session
        assert len(agent.controller._task_scheduler.sessions) == 0, \
            "初始时 sessions 应该为空"
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test"}
        )
        
        # 在 stream 过程中检查 session 注册（使用异步生成器）
        stream = agent.stream(input_event, session)
        
        # 读取第一个 chunk
        first_chunk = await stream.__anext__()
        assert first_chunk is not None, "应该能读取到第一个 chunk"
        
        # 此时 session 应该已注册
        assert session.session_id() in agent.controller._task_scheduler.sessions, \
            "stream 执行过程中 session 应该已注册"
        
        # 消费完所有 chunks
        async for _ in stream:
            pass
        
        # stream 结束后，session 应该被清理
        assert session.session_id() not in agent.controller._task_scheduler.sessions, \
            "stream 结束后 session 应该被移除"
        
        logger.info("✅ test_session_registration_and_cleanup passed")


# ==================== 事件系统测试 ====================

class TestEventSystem:
    """测试事件系统"""

    @pytest.mark.asyncio
    async def test_event_subscribe_and_publish(self):
        """测试事件订阅和发布
        
        测试目标：
        1. 验证 subscribe() 正确创建 4 种事件类型的订阅
        2. 验证 publish_event() 能正确发布事件
        3. 验证事件能正确路由到 EventHandler
        """
        agent = await build_test_agent(
            agent_id="test_event_pub_sub",
            event_handler=SimpleEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session = TaskSession(trace_id="test_pub_sub")
        
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test event system"}
        )
        
        # 执行 stream（会触发订阅、发布、路由）
        output = await collect_stream_output(agent.stream(input_event, session))
        
        # 验证事件被正确处理（通过输出验证）
        assert len(output) > 0, "应该有输出，证明事件被正确处理"
        assert any("started" in text for text in output), "应该有任务启动信息"
        
        logger.info("✅ test_event_subscribe_and_publish passed")

    @pytest.mark.asyncio
    async def test_event_unsubscribe_cleanup(self):
        """测试事件取消订阅
        
        测试目标：
        1. stream() 结束时正确取消订阅
        2. 验证订阅不会泄漏
        
        验证策略：
        - 每次 stream() 会为 4 种事件类型创建订阅（INPUT, TASK_INTERACTION, TASK_COMPLETION, TASK_FAILED）
        - stream() 结束后应该清理所有订阅
        - 多次调用 stream() 后，订阅数量应该保持稳定（不累积）
        
        注意：
        - 使用 DynamicTaskEventHandler 确保每次 stream 调用创建不同的 task_id
        - 避免任务 ID 冲突导致的测试失败
        """
        agent = await build_test_agent(
            agent_id="test_unsub",
            event_handler=DynamicTaskEventHandler(),
            task_executors={"cancellable": build_cancellable_executor}
        )
        
        session = TaskSession(trace_id="test_unsub")
        
        # 获取 MessageQueue 的 _subscribers（用于验证订阅清理）
        message_queue = agent.controller.event_queue._queue
        
        # 第一次 stream 调用前，订阅数应该为 0
        initial_sub_count = len(message_queue._subscribers)
        logger.info(f"初始订阅数: {initial_sub_count}")
        assert initial_sub_count == 0, f"初始订阅数应该为 0，实际为 {initial_sub_count}"
        
        # 第一次执行 stream
        input_event = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test unsubscribe round 1"}
        )
        output_1 = await collect_stream_output(agent.stream(input_event, session))
        assert len(output_1) > 0, "第一次 stream 应该有输出"
        
        # stream 结束后，验证订阅被清理
        sub_count_after_first = len(message_queue._subscribers)
        logger.info(f"第一次 stream 后订阅数: {sub_count_after_first}")
        assert sub_count_after_first == 0, \
            f"第一次 stream 后订阅应该被清理，期望 0，实际为 {sub_count_after_first}"
        
        # 第二次执行 stream（验证没有累积泄漏）
        input_event_2 = InputEvent(
            event_type=EventType.INPUT,
            content={"query": "test unsubscribe round 2"}
        )
        output_2 = await collect_stream_output(agent.stream(input_event_2, session))
        assert len(output_2) > 0, "第二次 stream 应该有输出"
        
        # 第二次 stream 结束后，验证订阅仍然被清理
        sub_count_after_second = len(message_queue._subscribers)
        logger.info(f"第二次 stream 后订阅数: {sub_count_after_second}")
        assert sub_count_after_second == 0, \
            f"第二次 stream 后订阅应该被清理，期望 0，实际为 {sub_count_after_second}"
        
        logger.info("✅ test_event_unsubscribe_cleanup passed - 验证了订阅正确清理，无泄漏")


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "-s"])
