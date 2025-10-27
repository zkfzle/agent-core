from dataclasses import dataclass
from typing import Optional, Union, TypeVar

from jiuwen.core.runtime.thread_safe_dict import ThreadSafeDict

Agent = TypeVar("Agent", contravariant=True)
AgentRuntime = TypeVar("AgentRuntime", contravariant=True)

@dataclass
class AgentWithRuntime:
    runtime: AgentRuntime
    agent: Agent

AgentProvider = lambda: Agent

class AgentMgr:
    def __init__(self):
        self._agents: ThreadSafeDict[str, AgentWithRuntime] = ThreadSafeDict()
        self._agent_providers: ThreadSafeDict[str, AgentWithRuntime] = ThreadSafeDict()
        pass

    def add_agent(self, agent_id: str, agent: Union[Agent, AgentProvider]) -> None:
        pass

    def remove_agent(self, agent_id: str) -> Optional[Agent]:
        pass

    def get_agent(self, agent_id) -> Optional[AgentWithRuntime]:
        pass