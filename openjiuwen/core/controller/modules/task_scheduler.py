"""任务调度器模块

该模块实现了任务调度和执行的核心功能，包括：
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
from typing import Callable, Dict, Optional

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.controller.base import ControllerConfig
from openjiuwen.core.controller.modules.event_queue import EventQueue
from openjiuwen.core.controller.modules.task_executor import TaskExecutor
from openjiuwen.core.controller.modules.task_manager import TaskManager
from openjiuwen.core.session import Session
from openjiuwen.core.single_agent.base import AbilityManager


class TaskExecutorRegistry:
    """任务执行器注册表
    
    管理不同类型的任务执行器，支持动态注册和获取。
    通过任务类型（task_type）来查找对应的TaskExecutor构建函数。
    """
    
    def __init__(self):
        """初始化任务执行器注册表"""
        self.task_executor_builders: Dict[
            str,
            Callable[[ControllerConfig, AbilityManager, ContextEngine, TaskManager, EventQueue], TaskExecutor]
        ] = {}

    def add_task_executor(
            self,
            task_type: str,
            task_executor_builder: Callable[
                [ControllerConfig, AbilityManager, ContextEngine, TaskManager, EventQueue], TaskExecutor
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
            ability_manager: AbilityManager,
            context_engine: ContextEngine,
            task_manager: TaskManager,
            event_queue: EventQueue
    ) -> TaskExecutor:
        """获取任务执行器实例
        
        Args:
            task_type: 任务类型标识符
            config: 控制器配置
            ability_manager: 能力包
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
            raise
        else:
            return executor_builder(config, ability_manager, context_engine, task_manager, event_queue)


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
            ability_manager: AbilityManager,
            event_queue: EventQueue
    ):
        """初始化任务调度器
        
        Args:
            config: 控制器配置
            task_manager: 任务管理器
            context_engine: 上下文引擎
            ability_manager: 能力包
            event_queue: 事件队列
        """
        self._config = config
        self._task_manager = task_manager
        self._context_engine = context_engine
        self._ability_manager = ability_manager
        self._event_queue = event_queue
        self._task_executor_registry = TaskExecutorRegistry()
        self._sessions: Dict[str, Session] = {}

        # 调度器运行状态
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None

        # 正在执行的任务：task_id -> (TaskExecutor, asyncio.Task)
        self._running_tasks: Dict[str, tuple[TaskExecutor, asyncio.Task]] = {}

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

    async def pause_task(self, task_id: str):
        """暂停任务
        
        执行流程：
        1. 判断当前任务是否在运行中，如果不在运行中，直接返回
        2. 获取当前任务的executor，执行can_pause方法，如果不可暂停，返回原因；否则继续
        3. 执行executor的pause方法后，取消当前任务对应的asyncio.Task
        4. 在task_manager中将该任务的状态改为paused
        
        Args:
            task_id: 任务ID
        """
        ...

    async def cancel_task(self, task_id: str):
        """取消任务
        
        执行流程：
        1. 判断当前任务是否在运行中，如果不在运行中，直接返回
        2. 获取当前任务的executor，执行can_cancel方法，如果不可取消，返回原因；否则继续
        3. 执行executor的cancel方法后，取消当前任务对应的asyncio.Task
        4. 在task_manager中将该任务的状态改为cancelled
        
        Args:
            task_id: 任务ID
        """
        ...

    async def schedule(self):
        """调度任务
        
        定期扫描并执行待执行任务。
        
        执行流程：
        1. 获取所有session_id在sessions中且状态为submitted的任务
        2. 如果未获取到，直接返回
        3. 并发执行所有获取到的任务，并流式输出任务执行过程中的输出
        4. 如果输出中的type为"completion"、"interaction"或"failed"时，
           停止该任务执行（取消asyncio.Task），并将task_manager中的任务状态置为相应的状态
        """
        ...

    def start(self):
        """启动任务调度器
        
        启动后台调度任务，开始定期扫描和执行待执行任务。
        """
        ...

    def stop(self):
        """停止任务调度器
        
        停止后台调度任务，停止扫描和执行任务。
        """
        ...