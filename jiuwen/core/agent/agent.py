from abc import ABC, abstractmethod
from typing import Any, Iterator, Dict, List, Union

from jiuwen.core.runtime.agent import StaticAgentRuntime
from jiuwen.core.runtime.runtime import Runtime

from jiuwen.core.runtime.workflow_manager import generate_workflow_key
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.wrapper import WrappedRuntime, StaticWrappedRuntime, TaskRuntime
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.tool.function.function import LocalFunction
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from jiuwen.core.workflow.base import Workflow


class AgentRuntime(WrappedRuntime, StaticWrappedRuntime):
    def __init__(self, config: Config = None):
        inner = StaticAgentRuntime(config)
        super().__init__(inner)
        self._runtime = inner

    async def write_stream(self, data: Union[dict, OutputSchema]):
        return await self.write_custom_stream(data)

    async def pre_run(self, **kwargs) -> Runtime:
        session_id = kwargs.get("session_id")
        if session_id is None:
            session_id = kwargs.get("trace_id")
        inputs = kwargs.get("inputs")
        inner = await self._runtime.create_agent_runtime(session_id, inputs)
        return TaskRuntime(inner=inner)

    async def release(self, session_id: str):
        await self._runtime.checkpointer().release(session_id)


class Agent(ABC):
    """
    The top-level abstract class and the common base class for all Agents.
    Subclasses must implement:
        - invoke : synchronous one-time call
        - stream : streaming call
    """

    def __init__(self, config: Config) -> None:
        self._runtime = AgentRuntime(config=config)
        self._config = config
        self._controller: "Controller | None" = self._init_controller()
        self._agent_handler: "AgentHandler | None" = self._init_agent_handler()

    def _init_controller(self) -> "Controller | None":
        return None

    def _init_agent_handler(self) -> "AgentHandler | None":
        return None

    def config(self):
        return self._config

    @abstractmethod
    async def invoke(self, inputs: Dict) -> Dict:
        pass

    @abstractmethod
    async def stream(self, inputs: Dict) -> Iterator[Any]:
        pass

    def bind_workflows(self, workflows: List[Workflow]):
        self._runtime.add_workflows(
            [(generate_workflow_key(workflow.config().metadata.id, workflow.config().metadata.version), workflow) for
             workflow in
             workflows])

    def bind_tools(self, tools: List[Tool]):
        self._runtime.add_tools(
            [(tool.name, tool) for tool in tools if (isinstance(tool, RestfulApi) or isinstance(tool, LocalFunction))])

    def get_llm_calls(self) -> Dict:
        raise NotImplementedError("")

    def copy(self) -> "Agent":
        raise NotImplementedError("")
