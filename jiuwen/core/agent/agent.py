from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Dict, List

from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.task_manager import TaskManager
from jiuwen.core.runtime.agent_context import AgentContext
from jiuwen.core.context.controller_context.controller_context_manager import ControllerContextMgr
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.workflow.base import Workflow


class Agent(ABC):
    """
    最顶层抽象，所有 Agent 的公共基类。
    子类必须实现：
        - invoke : 同步一次性调用
        - stream : 流式调用
    """

    def __init__(self, agent_config: "AgentConfig", agent_context: "AgentContext" = None) -> None:
        self._config = agent_config
        self._controller_context_manager: Optional["ControllerContextMgr"] = \
            self._init_controller_context_manager()
        self._controller: "Controller | None" = self._init_controller()
        self._agent_handler: "AgentHandler | None" = self._init_agent_handler()
        self._task_manager: "TaskManager | None" = self._init_task_manager(agent_context)

    def _init_controller(self) -> "Controller | None":
        """
        留给子类按需实例化 Controller；默认返回 None
        """
        return None

    def _init_agent_handler(self) -> "AgentHandler | None":
        """
        留给子类按需实例化 AgentHandler；默认返回 None
        """
        return None

    def _init_task_manager(self, agent_context: AgentContext) -> "TaskManager | None":
        """
        留给子类按需实例化 TaskManager；默认返回 None
        """
        if not agent_context:
            agent_context = AgentContext()
        return TaskManager(agent_context)

    def _init_controller_context_manager(self) -> Optional["ControllerContextMgr"]:
        """
        子类返回具体的 ControllerContextMgr 实例即可。
        默认返回 None，表示无需上下文管理。
        """
        return None

    @abstractmethod
    async def invoke(self, inputs: Dict) -> Dict:
        """
        同步调用，一次性返回最终结果
        """
        pass

    @abstractmethod
    async def stream(self, inputs: Dict) -> Iterator[Any]:
        """
        流式调用，逐个 yield 中间结果
        """
        pass

    def bind_workflows(self, workflows: List[Workflow]):
        self._controller_context_manager.workflow_mgr.add_workflows(workflows)

    def bind_tools(self, tools: List[Tool]):
        self._controller_context_manager.workflow_mgr.add_tools(tools)
