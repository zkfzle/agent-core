# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from openjiuwen.core.session import Config


class AgentGroupSession:
    """AgentGroup Session"""

    def __init__(self, config: Config = None, resource_mgr=None):
        """Initialize AgentGroupSession

        Args:
            config: Config object (optional, auto-created)
            resource_mgr: Resource manager (optional, auto-created)
        """
        # Create Config with agent_config if not provided
        if config is None:
            from openjiuwen.core.single_agent.legacy import AgentConfig
            config = Config()
            # Create virtual AgentConfig for Group Session
            agent_config = AgentConfig(id="agent_group_session")
            config.set_agent_config(agent_config)

        # Call parent constructor
        from openjiuwen.core.single_agent.legacy.agent import AgentSession
        self._inner = AgentSession(config, resource_mgr)


def create_agent_group_session(config: Config = None, resource_mgr=None) -> AgentGroupSession:
    """Factory method to create AgentGroupSession

    Args:
        config: Config object (optional)
        resource_mgr: Resource manager (optional)

    Returns:
        AgentGroupSession instance
    """
    return AgentGroupSession(config=config, resource_mgr=resource_mgr)