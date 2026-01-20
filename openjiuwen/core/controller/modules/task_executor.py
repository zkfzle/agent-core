"""任务执行器模块

该模块定义了任务执行器的抽象基类，用于执行不同类型的任务。

主要类：
- TaskExecutor: 任务执行器抽象基类，定义任务执行的接口

主要职责：
- 执行任务（execute）
- 检查任务是否可以暂停（can_pause）
- 暂停任务（pause）
- 检查任务是否可以取消（can_cancel）
- 取消任务（cancel）
"""
from __future__ import annotations

from abc import abstractmethod, ABC
from typing import AsyncIterator, Tuple, TYPE_CHECKING

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.controller.modules.event_queue import EventQueue
from openjiuwen.core.controller.schema.controller_output import ControllerOutputChunk
from openjiuwen.core.controller.modules.task_manager import TaskManager
from openjiuwen.core.session import Session
from openjiuwen.core.single_agent.base import AbilityManager

if TYPE_CHECKING:
    from openjiuwen.core.controller.base import ControllerConfig


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
            ability_manager: AbilityManager,
            context_engine: ContextEngine,
            task_manager: TaskManager,
            event_queue: EventQueue
    ):
        """初始化任务执行器

        Args:
            config: 控制器配置
            ability_manager: 能力包，包含可用的工具、工作流等
            context_engine: 上下文引擎，用于管理对话上下文
            task_manager: 任务管理器，用于更新任务状态
            event_queue: 事件队列，用于发布任务相关事件
        """
        self._config = config
        self._ability_manager = ability_manager
        self._context_engine = context_engine
        self._task_manager = task_manager
        self._event_queue = event_queue

    @abstractmethod
    async def execute(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
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
