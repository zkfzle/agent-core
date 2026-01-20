"""控制器基类定义

包含的主要类：
- ControllerConfig: 控制器状态枚举
- BaseController: 控制器基类

控制器负责处理事件，管理任务生命周期，并进行意图识别和处理。
"""
from typing import AsyncIterator, Optional, List, Callable

from pydantic import BaseModel, Field

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.controller import InputEvent

from openjiuwen.core.controller.modules.event_queue import EventQueue, EventHandler
from openjiuwen.core.controller.schema.controller_output import ControllerOutput, ControllerOutputChunk
from openjiuwen.core.controller.modules.task_manager import TaskManager
from openjiuwen.core.controller.modules.task_scheduler import TaskScheduler
from openjiuwen.core.controller.modules.task_executor import TaskExecutor
from openjiuwen.core.session.session import Session
from openjiuwen.core.session.stream.base import StreamMode
from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.single_agent.base import AbilityManager


class ControllerConfig(BaseModel):
    """控制器配置

    定义控制器的配置参数，用于控制控制器的行为。
    配置项分为几个类别：任务调度、任务管理、事件队列和意图识别。

    Attributes:
        # ==================== 任务调度配置 ====================
        max_concurrent_tasks: 最大并发任务数，控制同时执行的任务数量上限。
                             默认为5，设置为0表示不限制。
        schedule_interval: 任务调度间隔（秒），调度器定期扫描待执行任务的间隔时间。
                           默认为1.0秒，较小的值可以提高响应速度但会增加CPU使用。
        task_timeout: 任务超时时间（秒），超过此时间的任务将被标记为失败。
                     默认为None，表示不设置超时。

        # ==================== 任务管理配置 ====================
        default_task_priority: 默认任务优先级，创建任务时如果未指定优先级则使用此值。
                               默认为1，数字越大优先级越高。
        enable_task_persistence: 是否启用任务持久化，启用后任务状态会被保存以便恢复。
                                 默认为False。

        # ==================== 事件队列配置 ====================
        event_queue_size: 事件队列大小，限制队列中可存储的事件数量。
                         默认为None，表示不限制队列大小。
        event_timeout: 事件处理超时时间（秒），超过此时间未处理的事件将被丢弃。
                      默认为None，表示不设置超时。

        # ==================== 意图识别配置 ====================
        enable_intent_recognition: 是否启用意图识别功能。
                                   默认为True，启用后会自动识别用户意图并路由到相应处理。
        intent_confidence_threshold: 意图识别置信度阈值，低于此值的意图将被视为UNKNOWN_TASK。
                                    默认为0.7，范围0.0-1.0。

    Example:
        ```python
        config = ControllerConfig(
            max_concurrent_tasks=10,
            schedule_interval=0.5,
            default_task_priority=5,
            enable_intent_recognition=True
        )
        ```
    """
    # ==================== 任务调度配置 ====================
    max_concurrent_tasks: int = Field(
        default=5,
        description="最大并发任务数，控制同时执行的任务数量上限。设置为0表示不限制。"
    )
    schedule_interval: float = Field(
        default=1.0,
        ge=0.1,
        description="任务调度间隔（秒），调度器定期扫描待执行任务的间隔时间。"
    )
    task_timeout: Optional[float] = Field(
        default=None,
        ge=600,
        description="任务超时时间（秒），超过此时间的任务将被标记为失败。None表示不设置超时。"
    )

    # ==================== 任务管理配置 ====================
    default_task_priority: int = Field(
        default=1,
        description="默认任务优先级，创建任务时如果未指定优先级则使用此值。数字越大优先级越高。"
    )

    # ==================== 事件队列配置 ====================
    event_queue_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="事件队列大小，限制队列中可存储的事件数量。None表示不限制队列大小。"
    )
    event_timeout: Optional[float] = Field(
        default=None,
        ge=600,
        description="事件处理超时时间（秒），超过此时间未处理的事件将被丢弃。None表示不设置超时。"
    )

    # ==================== 意图识别配置 ====================
    intent_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="意图识别置信度阈值，低于此值的意图将被视为UNKNOWN_TASK。范围0.0-1.0。"
    )


class Controller:
    """控制器
    
    负责处理事件，管理任务生命周期。
    是ControllerAgent的核心组件。
    """
    
    def __init__(self):
        """初始化控制器"""
        super().__init__()
        self._card: Optional[str] = None
        self._ability_manager: Optional[AbilityManager] = None
        self._config: Optional[ControllerConfig] = None
        self._context_engine: Optional[ContextEngine] = None
        self._task_manager: Optional[TaskManager] = None
        self._event_queue: Optional[EventQueue] = None
        self._task_scheduler: Optional[TaskScheduler] = None
        self._event_handler: Optional[EventHandler] = None

    def init(
            self,
            card: AgentCard,
            config: ControllerConfig,
            ability_manager: AbilityManager,
            context_engine: ContextEngine
    ):
        """初始化控制器
        
        Args:
            card: Agent名片
            config: 控制器配置
            ability_manager: 能力包
            context_engine: 上下文引擎
        """
        self._card = card
        self._config = config
        self._ability_manager = ability_manager
        self._context_engine = context_engine
        self._task_manager = TaskManager(config=self._config)
        self._event_queue = EventQueue(config=self._config)
        self._task_scheduler = TaskScheduler(
            config=self._config,
            task_manager=self._task_manager,
            context_engine=self._context_engine,
            ability_manager=self._ability_manager,
            event_queue=self._event_queue
        )

    @property
    def event_queue(self) -> EventQueue:
        """获取事件队列"""
        return self._event_queue

    @property
    def config(self) -> ControllerConfig:
        """获取控制器配置"""
        return self._config

    @property
    def context_engine(self) -> ContextEngine:
        """获取上下文引擎"""
        return self._context_engine

    @property
    def ability_manager(self) -> AbilityManager:
        """获取能力包"""
        return self._ability_manager

    @config.setter
    def config(self, config: ControllerConfig):
        """设置控制器配置"""
        self._config = config

    @context_engine.setter
    def context_engine(self, context_engine: ContextEngine):
        """设置上下文引擎"""
        self._context_engine = context_engine

    @ability_manager.setter
    def ability_manager(self, ability_manager: AbilityManager):
        """设置能力包"""
        self._ability_manager = ability_manager

    def set_event_handler(self, event_handler: EventHandler):
        """设置事件处理器
        
        Args:
            event_handler: 事件处理器实例
        """
        self._event_handler = event_handler
        self._event_handler.config = self._config
        self._event_handler.context_engine = self._context_engine
        self._event_handler.task_scheduler = self._task_scheduler
        self._event_handler.task_manager = self._task_manager
        self._event_handler.ability_manager = self._ability_manager

    def add_task_executor(
            self,
            task_type: str,
            task_executor_builder: Callable[
                [ControllerConfig, AbilityManager, ContextEngine, TaskManager, EventQueue], TaskExecutor
            ]
    ) -> "Controller":
        """添加任务执行器
        
        Args:
            task_type: 任务类型
            task_executor_builder: 任务执行器构建函数
            
        Returns:
            self（支持链式调用）
        """
        self._task_scheduler.task_executor_registry.add_task_executor(task_type, task_executor_builder)
        return self

    def remove_task_executor(self, task_type: str):
        """移除任务执行器

        Args:
            task_type: 任务类型
        """
        self._task_scheduler.task_executor_registry.remove_task_executor(task_type)

    def start(self):
        """启动控制器
        
        启动任务调度器，开始处理任务
        """
        self._task_scheduler.start()

    def stop(self):
        """停止控制器
        
        停止任务调度器，停止处理任务
        """
        self._task_scheduler.stop()

    async def invoke(
            self,
            inputs: InputEvent,
            session: Session,
            **kwargs
    ) -> ControllerOutput:
        """批执行控制器
        
        Args:
            inputs: 输入事件
            session: 会话对象
            **kwargs: 其他参数
            
        Returns:
            ControllerOutput: 控制器输出结果
            
        Note:
            1. 调用 stream 方法
            2. 将流式消息转为批消息返回
        """
        ...

    async def stream(
            self,
            inputs: InputEvent,
            session: Session,
            stream_modes: Optional[List[StreamMode]] = None,
            **kwargs
    ) -> AsyncIterator[ControllerOutputChunk]:
        """流式执行控制器
        
        Args:
            inputs: 输入事件
            session: 会话对象
            stream_modes: 流式输出模式列表（可选）
            **kwargs: 其他参数
            
        Yields:
            ControllerOutputChunk: 控制器输出块
            
        Note:
            1. 恢复 controller 状态（包括 task_manager 的状态等）
            2. 将 Session 放到 task_scheduler 的 sessions 字典中
            3. 调用 self._event_queue 的 subscribe 方法订阅
            4. 将输入的事件放到 self._event_queue 中
            5. 获取事件处理结果并流式输出
            6. deactivate 所有 subscription
            7. 保存controller 状态（包括 task_manager 的状态等）
            8. 将 Session 从 task_scheduler 的 sessions 字典中移除
        """
        ...




