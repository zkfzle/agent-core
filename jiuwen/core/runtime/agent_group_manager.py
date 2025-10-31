from typing import Optional, Union

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.runtime.thread_safe_dict import ThreadSafeDict
from jiuwen.runner.agent_group import AgentGroup

AgentGroupProvider = lambda: AgentGroup


class AgentGroupMgr:
    def __init__(self):
        self._agent_groups: ThreadSafeDict[str, AgentGroup] = ThreadSafeDict()
        self._agent_group_providers: ThreadSafeDict[str, AgentGroupProvider] = ThreadSafeDict()

    def add_agent_group(self, agent_group_id: str, agent_group: Union[AgentGroup, AgentGroupProvider]) -> None:
        if agent_group_id is None or agent_group_id.strip() == "":
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_GROUP_ADD_FAILED.code,
                "agent_group_id cannot be empty"
            )
            
        try:
            if callable(agent_group):
                self._agent_group_providers[agent_group_id] = agent_group
            elif isinstance(agent_group, AgentGroup):
                self._agent_groups[agent_group_id] = agent_group
            else:
                raise TypeError(f"agent_group must be either AgentGroup instance or callable, got {type(agent_group)}")
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_GROUP_ADD_FAILED.code,
                str(e)
            )

    def remove_agent_group(self, agent_group_id: str) -> Optional[AgentGroup]:
        if agent_group_id is None or agent_group_id.strip() == "":
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_GROUP_REMOVE_FAILED.code,
                "agent_group_id cannot be empty"
            )
            
        try:
            group = self._agent_groups.pop(agent_group_id, None)
            if group:
                return group
            self._agent_group_providers.pop(agent_group_id, None)
            return None
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_GROUP_REMOVE_FAILED.code,
                str(e)
            )

    def get_agent_group(self, agent_group_id: str) -> Optional[AgentGroup]:
        if agent_group_id is None or agent_group_id.strip() == "":
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_GROUP_GET_FAILED.code,
                "agent_group_id cannot be empty"
            )
            
        try:
            group = self._agent_groups.get(agent_group_id)
            if group:
                return group
            provider = self._agent_group_providers.get(agent_group_id)
            if provider:
                group = provider()
                if not isinstance(group, AgentGroup):
                    raise TypeError(f"Provider did not return AgentGroup instance, got {type(group)}")
                self._agent_groups[agent_group_id] = group
                return group
            return None
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(
                StatusCode.RUNTIME_AGENT_GROUP_GET_FAILED.code,
                str(e)
            )
