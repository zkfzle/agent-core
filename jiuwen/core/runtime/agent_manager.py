from dataclasses import dataclass
from typing import Optional, Union

from jiuwen.core.agent.agent import AgentRuntime, Agent
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.runtime.resource_manager import ResourceMgr
from jiuwen.core.runtime.thread_safe_dict import ThreadSafeDict


@dataclass
class AgentWithRuntime:
    runtime: AgentRuntime
    agent: Agent

AgentProvider = lambda: Agent

class AgentMgr:
    def __init__(self, resource_manager: ResourceMgr):
        self._resource_manager: ResourceMgr = resource_manager
        self._agents: ThreadSafeDict[str, AgentWithRuntime] = ThreadSafeDict()
        self._agent_providers: ThreadSafeDict[str, AgentProvider] = ThreadSafeDict()

    def add_agent(self, agent_id: str, agent: Union[Agent, AgentProvider]) -> None:
        if agent_id is None or agent_id.strip() == "":
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_ADD_FAILED.code,
                StatusCode.RUNTIME_AGENT_ADD_FAILED.errmsg.format(reason="agent_id cannot be empty")
            )
            
        if agent is None:
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_ADD_FAILED.code,
                StatusCode.RUNTIME_AGENT_ADD_FAILED.errmsg.format(reason="agent cannot be None")
            )

        if callable(agent):
            self._agent_providers[agent_id] = agent
        else:
            if not hasattr(agent, "config"):
                raise JiuWenBaseException(
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.code,
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.errmsg.format(reason="Agent must have config method")
                )

            try:
                self._agents[agent_id] = AgentWithRuntime(
                    runtime=AgentRuntime(config=agent.config(), resource_mgr=self._resource_manager),
                    agent=agent
                )
            except Exception as e:
                raise JiuWenBaseException(
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.code,
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.errmsg.format(reason=f"Failed to create AgentWithRuntime: {str(e)}")
                )

    def remove_agent(self, agent_id: str) -> Optional[Agent]:
        if agent_id is None or agent_id.strip() == "":
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_REMOVE_FAILED.code,
                StatusCode.RUNTIME_AGENT_REMOVE_FAILED.errmsg.format(reason="agent_id cannot be empty")
            )
            
        try:
            agent_with_runtime = self._agents.pop(agent_id, None)
            if agent_with_runtime:
                return agent_with_runtime.agent

            self._agent_providers.pop(agent_id, None)
            return None
        except Exception as e:
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_REMOVE_FAILED.code,
                StatusCode.RUNTIME_AGENT_REMOVE_FAILED.errmsg.format(reason=str(e))
            )

    def get_agent(self, agent_id: str) -> Optional[AgentWithRuntime]:
        if agent_id is None or agent_id.strip() == "":
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_GET_FAILED.code,
                StatusCode.RUNTIME_AGENT_GET_FAILED.errmsg.format(reason="agent_id cannot be empty")
            )

        try:
            agent_with_runtime = self._agents.get(agent_id)
            if agent_with_runtime:
                return agent_with_runtime

            provider = self._agent_providers.get(agent_id)
            if provider:
                agent = provider()
                if not hasattr(agent, "config"):
                    raise JiuWenBaseException(
                        StatusCode.RUNTIME_AGENT_GET_FAILED.code,
                        StatusCode.RUNTIME_AGENT_GET_FAILED.errmsg.format(reason="Agent returned by provider must have config method")
                    )
                    
                agent_with_runtime = AgentWithRuntime(
                    runtime=AgentRuntime(config=agent.config(), resource_mgr=self._resource_manager),
                    agent=agent
                )
                self._agents[agent_id] = agent_with_runtime
                return agent_with_runtime
            return None
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_GET_FAILED.code,
                StatusCode.RUNTIME_AGENT_GET_FAILED.errmsg.format(reason=f"Failed to create agent from provider: {str(e)}")
            )