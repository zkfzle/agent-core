from abc import ABC, abstractmethod
from typing import Any, Iterator, Dict, List

from jiuwen.core.agent.task.task_context import AgentRuntime
from jiuwen.core.agent.task.task_manager import TaskManager
from jiuwen.core.context.controller_context.workflow_manager import generate_workflow_key
from jiuwen.core.runtime.agent_context import AgentContext
from jiuwen.core.runtime.config import Config
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.tool.function.function import LocalFunction
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from jiuwen.core.workflow.base import Workflow


class Agent(ABC):
    """
    最顶层抽象，所有 Agent 的公共基类。
    子类必须实现：
        - invoke : 同步一次性调用
        - stream : 流式调用
    """

    def __init__(self, config: Config, agent_context: "AgentContext" = None) -> None:
        self._runtime = AgentRuntime(config=config)
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
        self._runtime.add_workflows(
            [(generate_workflow_key(workflow.config().metadata.id, workflow.config().metadata.version), workflow) for
             workflow in
             workflows])
        for workflow in workflows:
            workflow_key = generate_workflow_key(
                workflow.config().metadata.id,
                workflow.config().metadata.version
            )
            workflow_schema = workflow.config().metadata.schema
            self._runtime.add_schema(workflow_key, workflow_schema)

    def bind_tools(self, tools: List[Tool]):
        self._runtime.add_tools([(tool.name, tool) for tool in tools if (isinstance(tool, RestfulApi) or isinstance(tool, LocalFunction))])
