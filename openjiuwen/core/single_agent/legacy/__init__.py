#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Legacy Interface Compatibility Layer

This package contains all deprecated interfaces for backward compatibility.
All interfaces in this package will be removed in v1.0.0.

For migration guide, see: docs/AGENT_MIGRATION_GUIDE.md

Created on: 2025-01-04
"""

# Legacy compat utilities
from openjiuwen.core.single_agent.legacy.compat import (
    LegacyMethodsMixin,
    create_react_agent_config,
)

# Legacy agents
from openjiuwen.core.single_agent.legacy.react_agent import (
    LegacyReActAgent,
    LegacyReActAgentConfig,
)

# Legacy base classes (from old agent.py)
from openjiuwen.core.single_agent.legacy.agent import (
    BaseAgent as LegacyBaseAgent,
    ControllerAgent,
    AgentSession,
    WorkflowFactory,
    workflow_provider,
)

# Legacy configs (from old config.py)
from openjiuwen.core.single_agent.legacy.config import (
    AgentConfig,
    LLMCallConfig,
    IntentDetectionConfig,
    ConstrainConfig,
    DefaultResponse,
    WorkflowAgentConfig,
)

__all__ = [
    # Mixins
    "LegacyMethodsMixin",
    # Factory functions
    "create_react_agent_config",
    "workflow_provider",
    # Legacy agents
    "LegacyReActAgent",
    "LegacyReActAgentConfig",
    # Legacy base classes
    "LegacyBaseAgent",
    "ControllerAgent",
    "AgentSession",
    "WorkflowFactory",
    # Legacy configs
    "AgentConfig",
    "LLMCallConfig",
    "IntentDetectionConfig",
    "ConstrainConfig",
    "DefaultResponse",
    "WorkflowAgentConfig",
]
