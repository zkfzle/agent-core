#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Single Agent Module

New interfaces (recommended):
    - AgentCard: Agent business card
    - BaseAgent: New Agent base class with Ability/AbilityKit
    - Ability, AbilityKit: Ability management

Legacy interfaces (deprecated, will be removed in v1.0.0):
    - All classes imported from legacy package
    
For migration guide, see: docs/AGENT_MIGRATION_GUIDE.md

Created on: 2025-01-04
"""

# ========== New interfaces (recommended) ==========
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.core.single_agent.agent import (
    BaseAgent,
    AbilityKit,
    Ability,
)

# Schema classes
from openjiuwen.core.single_agent.schema.schema import (
    WorkflowSchema,
    PluginSchema
)

# ========== Legacy interfaces (deprecated, backward compatible) ==========
from openjiuwen.core.single_agent.legacy import (
    # Mixin
    LegacyMethodsMixin,
    # Factory functions
    create_react_agent_config,
    workflow_provider,
    # Legacy agents
    LegacyReActAgent,
    LegacyReActAgentConfig,
    # Legacy base classes
    LegacyBaseAgent,
    ControllerAgent,
    AgentSession,
    WorkflowFactory,
    # Legacy configs
    AgentConfig,
    LLMCallConfig,
    IntentDetectionConfig,
    ConstrainConfig,
    DefaultResponse,
    WorkflowAgentConfig,
)

# For backward compatibility, map old names to legacy versions
ReActAgent = LegacyReActAgent
ReActAgentConfig = LegacyReActAgentConfig

__all__ = [
    # New interfaces
    "AgentCard",
    "BaseAgent",              # New BaseAgent with Ability/AbilityKit
    "AbilityKit",
    "Ability",
    # Legacy interfaces (compatible)
    "ReActAgent",              # Points to LegacyReActAgent
    "ReActAgentConfig",        # Points to LegacyReActAgentConfig
    "LegacyBaseAgent",         # Old BaseAgent
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
