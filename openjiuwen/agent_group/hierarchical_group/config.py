#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Hierarchical Group Configuration"""

from dataclasses import dataclass
from typing import Optional

from openjiuwen.core.agent_group.config import AgentGroupConfig


@dataclass
class HierarchicalGroupConfig(AgentGroupConfig):
    """Configuration for HierarchicalGroup (Leader-Worker pattern)
    
    Extends AgentGroupConfig with leader agent information.
    
    Attributes:
        leader_agent_id: ID of the leader agent (required)
        group_id: Group identifier (inherited)
        max_agents: Maximum number of agents (inherited)
        max_concurrent_messages: Max concurrent messages (inherited)
        message_timeout: Message timeout in seconds (inherited)
    """
    leader_agent_id: str = None
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.leader_agent_id:
            raise ValueError(
                "leader_agent_id is required for HierarchicalGroupConfig"
            )


