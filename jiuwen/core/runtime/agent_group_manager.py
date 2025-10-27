from typing import Optional, Union

from jiuwen.runner.agent_group import AgentGroup

AgentGroupProvider = lambda: AgentGroup


class AgentGroupMgr:
    def __init__(self):
        pass

    def add_agent_group(self, agent_group_id: str, agent_group: Union[AgentGroup, AgentGroupProvider]) -> None:
        pass

    def remove_agent_group(self, agent_group_id: str) -> Optional[AgentGroup]:
        pass

    def get_agent_group(self, agent_group_id) -> Optional[AgentGroup]:
        pass
