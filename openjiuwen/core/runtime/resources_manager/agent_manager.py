from dataclasses import dataclass
from typing import Optional, Union, Callable

from openjiuwen.core.agent.agent import Agent
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runner.drunner.remote_client.remote_agent import RemoteAgent
from openjiuwen.core.runner.runner_config import get_runner_config
from openjiuwen.core.runner.drunner.server_adapter.agent_adapter import AgentAdapter
from openjiuwen.core.runtime.agent import StaticAgentRuntime
from openjiuwen.core.runtime.resources_manager.resource_manager import ResourceMgr
from openjiuwen.core.runtime.resources_manager.abstract_manager import AbstractManager


@dataclass
class AgentWithRuntime:
    runtime: StaticAgentRuntime
    agent: Agent


AgentProvider = Callable[[], Agent]


class AgentMgr(AbstractManager[AgentWithRuntime]):
    def __init__(self, resource_manager: ResourceMgr):
        super().__init__()
        self._resource_manager: ResourceMgr = resource_manager

    from openjiuwen.core.runner.drunner.server_adapter.agent_adapter import AgentAdapter
    def add_agent(self, agent_id: str, agent: Union[Agent, AgentProvider, RemoteAgent, AgentAdapter]) -> None:
        self._validate_id(agent_id, StatusCode.RUNTIME_AGENT_ADD_FAILED, "agent")
        self._validate_resource(agent, StatusCode.RUNTIME_AGENT_ADD_FAILED, "agent cannot be None")

        # Define validation function for non-callable agents
        def validate_agent(agent_obj):
            if isinstance(agent, (RemoteAgent, AgentAdapter)):
                if get_runner_config().distributed_mode:
                    return agent
                raise JiuWenBaseException(
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.code,
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.errmsg.format(reason="RemoteAgent must be in distributed mode")
                )
            if not hasattr(agent_obj, "config"):
                raise JiuWenBaseException(
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.code,
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.errmsg.format(reason="Agent must have config method")
                )
            return AgentWithRuntime(
                runtime=StaticAgentRuntime(config=agent_obj.config(), resource_mgr=self._resource_manager),
                agent=agent_obj
            )

        self._add_resource(agent_id, agent, StatusCode.RUNTIME_AGENT_ADD_FAILED, validate_agent)

    def remove_agent(self, agent_id: str) -> Optional[Agent | RemoteAgent | AgentAdapter]:
        self._validate_id(agent_id, StatusCode.RUNTIME_AGENT_REMOVE_FAILED, "agent")

        agent_with_runtime = self._remove_resource(agent_id, StatusCode.RUNTIME_AGENT_REMOVE_FAILED)
        if isinstance(agent_with_runtime, (RemoteAgent, AgentAdapter)):
            return agent_with_runtime
        return agent_with_runtime.agent if agent_with_runtime else None

    def get_agent(self, agent_id: str) -> Optional[AgentWithRuntime | RemoteAgent]:

        self._validate_id(agent_id, StatusCode.RUNTIME_AGENT_GET_FAILED, "agent")

        # Define function to create agent from provider
        def create_agent_from_provider(provider):
            agent = provider()
            if not hasattr(agent, "config"):
                raise JiuWenBaseException(
                    StatusCode.RUNTIME_AGENT_GET_FAILED.code,
                    StatusCode.RUNTIME_AGENT_GET_FAILED.errmsg.format(
                        reason="Agent returned by provider must have config method"
                    )
                )
            return AgentWithRuntime(
                runtime=StaticAgentRuntime(config=agent.config(), resource_mgr=self._resource_manager),
                agent=agent
            )

        return self._get_resource(agent_id, StatusCode.RUNTIME_AGENT_GET_FAILED, create_agent_from_provider)
