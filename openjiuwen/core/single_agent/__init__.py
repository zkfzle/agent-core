#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Single Agent Module

New interfaces (recommended):
    - AgentCard: Agent business card
    - BaseAgent: Agent base class
    - ReActAgent, ReActAgentConfig: ReAct Agent

Legacy interfaces (deprecated, will be removed in v1.0.0):
    - AgentConfig, ControllerAgent, AgentSession, etc.
"""

# ========== New interfaces (recommended) ==========
from openjiuwen.core.single_agent.schema.agent_card import AgentCard

# Base agent classes
from openjiuwen.core.single_agent.agent import (
    BaseAgent,
    ControllerAgent,
    AgentSession,
    WorkflowFactory,
    workflow_provider,
)

# Agent configurations
from openjiuwen.core.single_agent.config import (
    AgentConfig,
    LLMCallConfig,
    IntentDetectionConfig,
    ConstrainConfig,
    DefaultResponse,
    WorkflowAgentConfig
)

# Schema classes
from openjiuwen.core.single_agent.schema.schema import (
    WorkflowSchema,
    PluginSchema
)

# ReAct agent
from openjiuwen.core.single_agent.agents.react_agent import (
    ReActAgent,
    ReActAgentConfig,
)

# ========== Legacy interfaces (deprecated, backward compatible) ==========
from openjiuwen.core.single_agent.legacy import (
    # Mixin (for subclass use)
    LegacyMethodsMixin,
    # Deprecated factory functions
    create_react_agent_config,
)

__all__ = [
    # New interfaces
    "AgentCard",
    "BaseAgent",
    "ReActAgent",
    "ReActAgentConfig",
    # Legacy interfaces (compatible)
    "AgentConfig",
    "ControllerAgent",
    "AgentSession",
    "WorkflowFactory",
    "workflow_provider",
    "create_react_agent_config",
    "LLMCallConfig",
    "IntentDetectionConfig",
    "ConstrainConfig",
    "DefaultResponse",
    "WorkflowAgentConfig",
    "LegacyMethodsMixin",
    # Schema classes
    "WorkflowSchema",
    "PluginSchema",
]