"""事件处理器模块

该模块定义了事件处理器相关的类，包括：
- EventHandlerInput: 事件处理器输入数据模型
- EventHandler: 事件处理器抽象基类

事件处理器负责处理不同类型的事件：
- INPUT: 用户输入事件
- TASK_INTERACTION: 任务交互事件（任务执行过程中需要用户交互）
- TASK_COMPLETION: 任务完成事件
- TASK_FAILED: 任务失败事件
"""
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

from pydantic import BaseModel

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.controller.base import ControllerConfig
from openjiuwen.core.controller.schema.event import Event
from openjiuwen.core.session import Session
from openjiuwen.core.single_agent.base import AbilityManager

if TYPE_CHECKING:
    from openjiuwen.core.controller.modules.task_manager import TaskManager
    from openjiuwen.core.controller.modules.task_scheduler import TaskScheduler


class EventHandlerInput(BaseModel):
    """事件处理器输入数据模型

    包含事件和会话信息，用于传递给事件处理器。

    Attributes:
        event: 事件对象
        session: 会话对象
    """
    event: Event
    session: Session


class EventHandler(ABC):
    """事件处理器抽象基类

    定义事件处理的接口，不同类型的控制器需要实现不同的事件处理器。

    主要职责：
    - 处理输入事件（handle_input）
    - 处理任务交互事件（handle_task_interaction）
    - 处理任务完成事件（handle_task_completion）
    - 处理任务失败事件（handle_task_failed）
    """
    def __init__(self):
        """初始化事件处理器

        初始化时所有依赖为None，需要通过属性设置器注入依赖。
        """
        self._config = None
        self._context_engine = None
        self._ability_manager = None
        self._task_manager = None
        self._task_scheduler = None

    @property
    def config(self) -> ControllerConfig:
        return self._config

    @property
    def context_engine(self) -> ContextEngine:
        return self._context_engine

    @property
    def task_manager(self) -> "TaskManager":
        return self._task_manager

    @property
    def ability_manager(self) -> AbilityManager:
        return self._ability_manager

    @property
    def task_scheduler(self) -> "TaskScheduler":
        return self._task_scheduler

    @config.setter
    def config(self, config: ControllerConfig):
        self._config = config

    @context_engine.setter
    def context_engine(self, context_engine: ContextEngine):
        self._context_engine = context_engine

    @task_manager.setter
    def task_manager(self, task_manager: "TaskManager"):
        self._task_manager = task_manager

    @ability_manager.setter
    def ability_manager(self, ability_manager: AbilityManager):
        self._ability_manager = ability_manager

    @task_scheduler.setter
    def task_scheduler(self, task_scheduler: "TaskScheduler"):
        self._task_scheduler = task_scheduler

    @abstractmethod
    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件

        Args:
            inputs: 事件处理器输入，包含事件和会话信息
        """
        ...

    @abstractmethod
    async def handle_task_interaction(self, inputs: EventHandlerInput):
        """处理任务交互事件

        当任务执行过程中需要用户交互时触发。

        Args:
            inputs: 事件处理器输入，包含事件和会话信息
        """
        ...

    @abstractmethod
    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件

        当任务执行完成时触发。

        Args:
            inputs: 事件处理器输入，包含事件和会话信息
        """
        ...

    @abstractmethod
    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件

        当任务执行失败时触发。

        Args:
            inputs: 事件处理器输入，包含事件和会话信息
        """
        ...
