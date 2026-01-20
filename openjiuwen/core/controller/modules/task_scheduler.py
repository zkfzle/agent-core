"""任务调度器模块

该模块实现了任务调度和执行的核心功能，包括：
- TaskExecutor: 任务执行器抽象基类，定义任务执行的接口
- TaskExecutorRegistry: 任务执行器注册表，管理不同类型的任务执行器
- TaskScheduler: 任务调度器，负责任务的调度、执行、暂停和取消

核心工作流程：
1. 接收待执行的任务（状态为submitted）
2. 根据任务类型从注册表获取对应的TaskExecutor
3. 并发执行多个任务
4. 流式输出任务执行过程中的输出
5. 根据输出类型更新任务状态（completion/interaction/failed）
"""
import asyncio
from abc import abstractmethod, ABC
from typing import AsyncIterator, Callable, Dict, Optional, Tuple, TYPE_CHECKING

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.controller.config import ControllerConfig
from openjiuwen.core.controller.modules.event_queue import EventQueue
from openjiuwen.core.controller.modules.task_manager import TaskManager, TaskFilter
from openjiuwen.core.session import Session
from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.controller.schema import (EventType, TaskCompletionEvent, TaskInteractionEvent, TaskFailedEvent,
                                                TaskStatus, ControllerOutputChunk, ControllerOutputPayload, TextDataFrame, Task)

if TYPE_CHECKING:
    from openjiuwen.core.single_agent.agent import AbilityKit
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.common.exception.codes import StatusCode


class TaskExecutor(ABC):
    """任务执行器抽象基类
    
    定义任务执行的接口，不同类型的任务需要实现不同的TaskExecutor。
    
    主要职责：
    - 执行任务（execute_ability）
    - 检查任务是否可以暂停（can_pause）
    - 暂停任务（pause）
    - 检查任务是否可以取消（can_cancel）
    - 取消任务（cancel）
    """
    def __init__(
            self,
            config: ControllerConfig,
            ability_kit: 'AbilityKit',
            context_engine: ContextEngine,
            task_manager: TaskManager,
            event_queue: EventQueue
    ):
        """初始化任务执行器
        
        Args:
            config: 控制器配置
            ability_kit: 能力包，包含可用的工具、工作流等
            context_engine: 上下文引擎，用于管理对话上下文
            task_manager: 任务管理器，用于更新任务状态
            event_queue: 事件队列，用于发布任务相关事件
        """
        self._config = config
        self._ability_kit = ability_kit
        self._context_engine = context_engine
        self._task_manager = task_manager
        self._event_queue = event_queue

    @abstractmethod
    async def execute_ability(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
        """执行任务
        
        Args:
            task_id: 任务ID
            session: 会话对象
            
        Yields:
            ControllerOutputChunk: 任务执行过程中的输出块
        """
        ...

    @abstractmethod
    async def can_pause(self, task_id: str, session: Session) -> Tuple[bool, str]:
        """检查任务是否可以暂停
        
        Args:
            task_id: 任务ID
            session: 会话对象
            
        Returns:
            Tuple[bool, str]: (是否可以暂停, 如果不能暂停的原因)
        """
        ...

    @abstractmethod
    async def pause(self, task_id: str, session: Session) -> bool:
        """暂停任务
        
        Args:
            task_id: 任务ID
            session: 会话对象
            
        Returns:
            bool: 是否成功暂停
        """
        ...

    @abstractmethod
    async def can_cancel(self, task_id: str, session: Session) -> Tuple[bool, str]:
        """检查任务是否可以取消
        
        Args:
            task_id: 任务ID
            session: 会话对象
            
        Returns:
            Tuple[bool, str]: (是否可以取消, 如果不能取消的原因)
        """
        ...

    @abstractmethod
    async def cancel(self, task_id: str, session: Session) -> bool:
        """取消任务
        
        Args:
            task_id: 任务ID
            session: 会话对象
            
        Returns:
            bool: 是否成功取消
        """
        ...


class TaskExecutorRegistry:
    """任务执行器注册表
    
    管理不同类型的任务执行器，支持动态注册和获取。
    通过任务类型（task_type）来查找对应的TaskExecutor构建函数。
    """
    
    def __init__(self):
        """初始化任务执行器注册表"""
        self.task_executor_builders: Dict[
            str,
            Callable[[ControllerConfig, 'AbilityKit', ContextEngine, TaskManager, EventQueue], TaskExecutor]
        ] = {}

    def add_task_executor(
            self,
            task_type: str,
            task_executor_builder: Callable[
                [ControllerConfig, 'AbilityKit', ContextEngine, TaskManager, EventQueue], TaskExecutor
            ]
    ):
        """注册任务执行器
        
        Args:
            task_type: 任务类型标识符
            task_executor_builder: 任务执行器构建函数，接收配置和依赖，返回TaskExecutor实例
        """
        self.task_executor_builders[task_type] = task_executor_builder

    def remove_task_executor(self, task_type: str):
        """移除任务执行器
        
        Args:
            task_type: 任务类型标识符
        """
        if task_type in self.task_executor_builders:
            del self.task_executor_builders[task_type]

    def get_task_executor(
            self,
            task_type: str,
            config: ControllerConfig,
            ability_kit: 'AbilityKit',
            context_engine: ContextEngine,
            task_manager: TaskManager,
            event_queue: EventQueue
    ) -> TaskExecutor:
        """获取任务执行器实例
        
        Args:
            task_type: 任务类型标识符
            config: 控制器配置
            ability_kit: 能力包
            context_engine: 上下文引擎
            task_manager: 任务管理器
            event_queue: 事件队列
            
        Returns:
            TaskExecutor: 任务执行器实例
            
        Raises:
            Exception: 如果任务类型未注册
        """
        executor_builder = self.task_executor_builders.get(task_type, None)
        if executor_builder is None:
            raise build_error(StatusCode.CONTROLLER_TASK_EXECUTION_ERROR, error_msg="task executor not found")
        else:
            return executor_builder(config, ability_kit, context_engine, task_manager, event_queue)


class TaskScheduler:
    """任务调度器
    
    负责任务的调度、执行、暂停和取消。
    支持并发执行多个任务，并流式输出任务执行过程中的输出。
    
    工作流程：
    1. 定期扫描待执行任务（状态为submitted）
    2. 并发执行多个任务
    3. 流式输出任务执行过程中的输出
    4. 根据输出类型更新任务状态
    
    Attributes:
        _sessions: 会话字典，session_id -> Session
        _running_tasks: 正在执行的任务字典，task_id -> (TaskExecutor, asyncio.Task)
        _running: 调度器是否正在运行
        _scheduler_task: 调度器后台任务
        _lock: 用于同步访问的锁
    """
    
    def __init__(
            self,
            config: ControllerConfig,
            task_manager: TaskManager,
            context_engine: ContextEngine,
            ability_kit: 'AbilityKit',
            event_queue: EventQueue,
            card: AgentCard
    ):
        """初始化任务调度器
        
        Args:
            config: 控制器配置
            task_manager: 任务管理器
            context_engine: 上下文引擎
            ability_kit: 能力包
            event_queue: 事件队列
            card: Agent card（用于发布事件）
        """
        self._config = config
        self._task_manager = task_manager
        self._context_engine = context_engine
        self._ability_kit = ability_kit
        self._event_queue = event_queue
        self._task_executor_registry = TaskExecutorRegistry()
        self._sessions: Dict[str, Session] = {}
        self._card = card

        # 调度器运行状态
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None

        # 正在执行的任务：task_id -> (TaskExecutor, asyncio.Task)
        self._running_tasks: Dict[str, Tuple[Optional[TaskExecutor], Optional[asyncio.Task]]] = {}

        # 用于同步访问的锁
        self._lock = asyncio.Lock()

    @property
    def sessions(self) -> Dict[str, Session]:
        """获取会话字典"""
        return self._sessions

    @property
    def task_executor_registry(self):
        """获取任务执行器注册表"""
        return self._task_executor_registry

    async def _execute_task_wrapper(self, task_id: str, session: Session):
        """任务执行包装器
        
        包装execute_task，确保异常被捕获并正确处理。
        
        Args:
            task_id: 任务ID
            session: 会话对象
        """
        try:
            await self.execute_task(task_id, session)
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} execution cancelled")
            raise
        except Exception as e:
            logger.error(f"Task {task_id} execution failed: {e}", exc_info=True)
            self._task_manager.update_task_status(task_id, TaskStatus.FAILED)
            failed_chunk = ControllerOutputChunk(
                index=0,
                type="controller_output",
                payload=ControllerOutputPayload(
                    type=EventType.TASK_FAILED,
                    data=[TextDataFrame(type="text", text=str(e))]
                )
            )
            await self._publish_task_event(task_id, session.session_id(), failed_chunk)
        finally:
            # 清理运行中任务记录
            async with self._lock:
                if task_id in self._running_tasks:
                    del self._running_tasks[task_id]

    async def execute_task(self, task_id: str, session: Session):
        """执行任务

        执行流程：
        1. 判断当前任务是否在运行中，如果在运行中，直接返回
        2. 根据任务类型构建对应的executor，并执行execute_ability方法
        3. 在task_manager中将该任务的状态改为working
        4. 任务成功完成后，发布TaskCompletionEvent到事件队列

        Args:
            task_id: 任务ID
            session: 会话对象

        Note:
            - 任务成功完成后会自动发布TaskCompletionEvent
            - TaskCompletionEvent包含任务结果和任务对象
        """
        # 1. 获取任务对象
        tasks = self._task_manager.get_task(task_filter=TaskFilter(task_id=task_id))
        if not tasks:
            logger.error(f"Task {task_id} not found")
            return
        task = tasks[0]

        logger.info(f"Executing task {task_id} (type: {task.task_type})")
        
        # 2. 创建TaskExecutor
        executor = self._task_executor_registry.get_task_executor(
            task_type=task.task_type,
            config=self._config,
            ability_kit=self._ability_kit,
            context_engine=self._context_engine,
            task_manager=self._task_manager,
            event_queue=self._event_queue
        )

        # 更新运行中任务记录（添加executor）
        async with self._lock:
            if task_id in self._running_tasks:
                _, exec_task = self._running_tasks[task_id]
                self._running_tasks[task_id] = (executor, exec_task)
        
        # 3. 更新任务状态为WORKING
        self._task_manager.update_task_status(task_id, TaskStatus.WORKING)
        
        # 4. 流式执行任务
        try:
            async for chunk in executor.execute_ability(task_id, session):
                # 4.1 写入session流（让ControllerAgent能读取）
                await session.write_stream(chunk)
                
                # 4.2 检查输出类型，决定是否停止任务
                if chunk.payload and chunk.payload.type:
                    payload_type = chunk.payload.type
                    
                    # 任务完成
                    if payload_type == EventType.TASK_COMPLETION:
                        logger.info(f"Task {task_id} completed")
                        self._task_manager.update_task_status(task_id, TaskStatus.COMPLETED)
                        await self._publish_task_event(task_id, session.session_id(), chunk)
                        break
                    
                    # 任务需要交互
                    elif payload_type == EventType.TASK_INTERACTION:
                        logger.info(f"Task {task_id} requires interaction")
                        self._task_manager.update_task_status(task_id, TaskStatus.INPUT_REQUIRED)
                        await self._publish_task_event(task_id, session.session_id(), chunk)
                        break
                    
                    # 任务失败
                    elif payload_type == EventType.TASK_FAILED:
                        logger.error(f"Task {task_id} failed")
                        self._task_manager.update_task_status(task_id, TaskStatus.FAILED)
                        await self._publish_task_event(task_id, session.session_id(), chunk)
                        break
                    
                    # 处理中（继续执行）
                    elif payload_type == "processing":
                        continue
        
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} cancelled during execution")
            raise
        except Exception as e:
            logger.error(f"Task {task_id} execution error: {e}", exc_info=True)
            raise
        finally:
            # 检查该 session 是否所有任务都完成
            await self._check_and_notify_session_completion(session.session_id())

    async def _check_and_notify_session_completion(self, session_id: str):
        """检查 session 的所有任务是否完成，如果是则发送完成消息
        
        Args:
            session_id: 会话ID
        """
        # 获取该 session 的所有任务
        session_tasks = self._task_manager.get_task(task_filter=TaskFilter(session_id=session_id))
        if not session_tasks:
            return
        
        # 检查是否所有任务都处于终态
        terminal_states = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED,
                            TaskStatus.PAUSED, TaskStatus.INPUT_REQUIRED}
        if all(task.status in terminal_states for task in session_tasks):
            logger.info(f"All tasks have been handled for session {session_id}, sending completion message")
            
            # 获取 session 对象
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found, cannot send completion message")
                return
            
            # 发送完成消息到流
            from openjiuwen.core.controller.schema import ControllerOutputChunk, ControllerOutputPayload, TextDataFrame
            completion_chunk = ControllerOutputChunk(
                index=0,
                type="controller_output",
                payload=ControllerOutputPayload(
                    type="all_tasks_processed",
                    data=[TextDataFrame(type="text", text="All tasks have been successfully processed")],
                ),
                last_chunk=True
            )
            await session.write_stream(completion_chunk)
            logger.info(f"Sent completion message for session {session_id}")

    async def _publish_task_event(
        self,
        task_id: str,
        session_id: str,
        chunk: ControllerOutputChunk
    ):
        """发布任务事件（从 chunk 自动识别类型）
        
        根据 chunk.payload.type 自动识别事件类型并提取数据，构造对应的事件对象。
        
        Args:
            task_id: 任务ID
            session_id: 会话ID
            chunk: 输出块（包含事件类型和数据）
        """
        if not chunk.payload or not chunk.payload.type:
            logger.error(f"Invalid chunk for task {task_id}: missing payload or type")
            return
        
        tasks = self._task_manager.get_task(task_filter=TaskFilter(task_id=task_id))
        if not tasks:
            logger.error(f"Task {task_id} not found in TaskManager")
            return
        task = tasks[0]
        payload_type = chunk.payload.type
        payload_data = chunk.payload.data if chunk.payload else []
        
        # 根据 payload 类型自动构造事件
        if payload_type == EventType.TASK_COMPLETION:
            event = TaskCompletionEvent(task_result=payload_data, task=task)
        elif payload_type == EventType.TASK_INTERACTION:
            event = TaskInteractionEvent(interaction=payload_data, task=task)
        elif payload_type == EventType.TASK_FAILED:
            error_msg = payload_data[0].text if payload_data else "Unknown error"
            if task:
                task.error_message = error_msg
            event = TaskFailedEvent(error_message=error_msg, task=task)
        else:
            logger.error(f"Unsupported payload type: {payload_type}")
            return
        
        await self._event_queue.publish_event(self._card.id, session_id, event)
        logger.info(f"Published {payload_type} for task {task_id}")

    async def pause_task(self, task_id: str, session: Optional[Session] = None) -> bool:
        """暂停任务
        
        可以被EventHandler在回调中直接调用。
        执行流程：
            1. 判断当前任务是否在运行中，如果不在运行中，直接返回
            2. 获取当前任务的executor，执行can_pause方法，如果不可暂停，返回原因；否则继续
            3. 执行executor的pause方法后，取消当前任务对应的asyncio.Task
            4. 在task_manager中将该任务的状态改为paused
        
        Args:
            task_id: 任务ID
            session: 会话对象（可选，如果不提供则从task获取）
            
        Returns:
            bool: 是否成功暂停
        """
        async with self._lock:
            # 1. 检查任务是否在运行
            if task_id not in self._running_tasks:
                logger.warning(f"Task {task_id} is not running, cannot pause")
                return False
            
            # 2. 获取任务和executor
            executor, exec_task = self._running_tasks[task_id]
            tasks = self._task_manager.get_task(task_filter=TaskFilter(task_id=task_id))
            if not tasks:
                logger.error(f"Task {task_id} not found in TaskManager")
                return False
            task = tasks[0]
            
            # 3. 获取session
            session = self._sessions.get(task.session_id)
            if not session:
                logger.error(f"Session {task.session_id} not found for task {task_id}")
                return False
            
            # 4. 检查是否可以暂停
            if executor:
                try:
                    can_pause, reason = await executor.can_pause(task_id, session)
                    if not can_pause:
                        logger.warning(f"Task {task_id} cannot be paused: {reason}")
                        return False
                    
                    # 5. 执行executor的暂停逻辑
                    await executor.pause(task_id, session)
                except Exception as e:
                    logger.error(f"Error pausing task {task_id}: {e}", exc_info=True)
                    return False
            
            # 6. 取消asyncio任务
            if exec_task and not exec_task.done():
                exec_task.cancel()
            
            # 7. 更新任务状态
            self._task_manager.update_task_status(task_id, TaskStatus.PAUSED)
            
            # 8. 清理运行中任务记录
            del self._running_tasks[task_id]
            
            logger.info(f"Task {task_id} paused successfully")
            return True

    async def cancel_task(self, task_id: str, session: Optional[Session] = None) -> bool:
        """取消任务
        
        可以被EventHandler在回调中直接调用。、

        执行流程：
        1. 判断当前任务是否在运行中，如果不在运行中，直接返回
        2. 获取当前任务的executor，执行can_cancel方法，如果不可取消，返回原因；否则继续
        3. 执行executor的cancel方法后，取消当前任务对应的asyncio.Task
        4. 在task_manager中将该任务的状态改为cancelled
        
        Args:
            task_id: 任务ID
            session: 会话对象（可选，如果不提供则从task获取）
            
        Returns:
            bool: 是否成功取消
        """
        async with self._lock:
            # 1. 检查任务是否在运行
            if task_id not in self._running_tasks:
                logger.warning(f"Task {task_id} is not running, cannot cancel")
                return False
            
            # 2. 获取任务和executor
            executor, exec_task = self._running_tasks[task_id]
            tasks = self._task_manager.get_task(task_filter=TaskFilter(task_id=task_id))  
            if not tasks:
                logger.error(f"Task {task_id} not found in TaskManager")
                return False
            task = tasks[0]
            
            # 3. 获取session
            session = self._sessions.get(task.session_id)
            if not session:
                logger.error(f"Session {task.session_id} not found for task {task_id}")
                return False
            
            # 4. 检查是否可以取消
            if executor:
                try:
                    can_cancel, reason = await executor.can_cancel(task_id, session)
                    if not can_cancel:
                        logger.warning(f"Task {task_id} cannot be cancelled: {reason}")
                        return False
                    
                    # 5. 执行executor的取消逻辑
                    await executor.cancel(task_id, session)
                except Exception as e:
                    logger.error(f"Error cancelling task {task_id}: {e}", exc_info=True)
                    return False
            
            # 6. 取消asyncio任务
            if exec_task and not exec_task.done():
                exec_task.cancel()
            
            # 7. 更新任务状态
            self._task_manager.update_task_status(task_id, TaskStatus.CANCELED)
            
            # 8. 清理运行中任务记录
            del self._running_tasks[task_id]
            
            logger.info(f"Task {task_id} cancelled successfully")
            return True

    async def cancel_all_tasks(self, session_id: Optional[str] = None) -> int:
        """取消所有任务或指定session的所有任务
        
        便捷方法，用于批量取消任务。
        
        Args:
            session_id: 会话ID（可选），如果提供则只取消该session的任务
            
        Returns:
            int: 成功取消的任务数量
        """
        cancelled_count = 0
        
        # 获取要取消的任务列表（需要复制，会在循环中修改_running_tasks）
        async with self._lock:
            tasks_to_cancel = []
            for task_id, (executor, exec_task) in list(self._running_tasks.items()):
                tasks = self._task_manager.get_task(task_filter=TaskFilter(task_id=task_id))
                if tasks and (session_id is None or tasks[0].session_id == session_id):
                    tasks_to_cancel.append(task_id)
        
        # 取消任务（不在锁内，避免死锁）
        for task_id in tasks_to_cancel:
            if await self.cancel_task(task_id):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} tasks" + 
                    (f" for session {session_id}" if session_id else ""))
        return cancelled_count

    async def schedule(self):
        """后台调度循环
        定期扫描TaskManager，发现SUBMITTED任务就并发执行。
        使用 create_task 保持循环活跃，使用 gather 在停止时优雅清理。
        
        执行流程：
        1. 获取所有session_id在sessions中且状态为submitted的任务
        2. 使用 create_task 非阻塞启动所有新任务 并流式输出任务执行过程中的输出
        3. 短暂休眠后继续下一轮扫描
        4. 停止时等待所有运行中的任务完成
        """
        logger.info("TaskScheduler schedule loop started")
        
        while self._running:
            try:
                # 1. 获取待执行任务
                submitted_tasks = self._task_manager.get_task(task_filter=TaskFilter(status=TaskStatus.SUBMITTED))
                
                # 2. 并发启动所有新任务（非阻塞）
                for task in submitted_tasks:
                    # 检查是否已在运行
                    if task.task_id in self._running_tasks:
                        continue
                    
                    # 检查session是否存在
                    session = self._sessions.get(task.session_id)
                    if not session:
                        logger.warning(
                            f"Task {task.task_id} session {task.session_id} not found, skipping"
                        )
                        continue
                    
                    # 使用 create_task 非阻塞启动
                    exec_task = asyncio.create_task(
                        self._execute_task_wrapper(task.task_id, session)
                    )
                    
                    # 记录到运行中任务
                    self._running_tasks[task.task_id] = (None, exec_task)
                    
                    logger.info(f"Task {task.task_id} ({task.task_type}) started")
                
                # 3. 短暂休眠（循环不阻塞）
                await asyncio.sleep(self._config.schedule_interval)
                
            except asyncio.CancelledError:
                logger.info("TaskScheduler schedule loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in schedule loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        # 4. 停止时使用 gather 等待所有任务完成
        await self._wait_all_tasks_complete()
        
        logger.info("TaskScheduler schedule end")

    async def _wait_all_tasks_complete(self):
        """等待所有运行中的任务完成
        
        使用 gather 优雅地等待所有任务。
        """
        if not self._running_tasks:
            return
        
        logger.info(f"Waiting for {len(self._running_tasks)} running tasks to complete...")
        
        # 收集所有asyncio.Task对象
        tasks = [
            exec_task 
            for _, exec_task in self._running_tasks.values() 
            if exec_task is not None
        ]
        
        if tasks:
            # 使用 gather 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计结果
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            error_count = len(results) - success_count
            
            logger.info(
                f"All tasks completed: {success_count} succeeded, {error_count} failed"
            )
        
        # 清理
        self._running_tasks.clear()

    async def start(self):
        """启动任务调度器
        
        启动后台调度任务，开始定期扫描和执行待执行任务。
        """
        if self._running:
            logger.warning(f"TaskScheduler is already running")
            return
        self._running = True
        self._scheduler_task = asyncio.create_task(self.schedule())
        logger.info(f"TaskScheduler started")

    async def stop(self):
        """停止任务调度器
        
        停止后台调度任务，停止扫描和执行任务。
        """
        if not self._running:
            logger.warning(f"TaskScheduler is not running")
            return
            
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"TaskScheduler stopped")
