# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Single Agent Module

This module provides exports for single agent functionality.
Legacy implementations are in the legacy/ directory and should be
imported from openjiuwen.core.single_agent.legacy explicitly.

For migration guide, see: docs/AGENT_MIGRATION_GUIDE.md

Note: Legacy classes have been moved to the legacy submodule.
Please use 'from openjiuwen.core.single_agent.legacy import ...' for
legacy classes like LegacyReActAgent, AgentConfig, etc.
"""
from typing import Union

from openjiuwen.core.session.agent import Session, create_agent_session
from openjiuwen.core.single_agent.schema.agent_card import AgentCard

def __getattr__(name):
    if name == "BaseAgent":
        from openjiuwen.core.single_agent.agent import BaseAgent
        return BaseAgent
    elif name == "AbilityManager":
        from openjiuwen.core.single_agent.agent import AbilityManager
        return AbilityManager
    elif name == "LegacyBaseAgent":
        from openjiuwen.core.single_agent.legacy import LegacyBaseAgent
        return LegacyBaseAgent
    elif name == "ReActAgent":
        from openjiuwen.core.single_agent.agents.react_agent import ReActAgent
        return ReActAgent
    elif name == "ReActAgentConfig":
        from openjiuwen.core.single_agent.agents.react_agent import ReActAgentConfig
        return ReActAgentConfig
    elif name == "BaseAgentAlias":
        from openjiuwen.core.single_agent.agent import BaseAgent
        from openjiuwen.core.single_agent.legacy import LegacyBaseAgent
        return Union[BaseAgent, LegacyBaseAgent]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # New classes
    "AgentCard",
    "ReActAgent",
    "ReActAgentConfig",
    "Session",
    "create_agent_session",
    # For compatibility
    "BaseAgentAlias",
    "AbilityManager",
    "BaseAgent"
]
