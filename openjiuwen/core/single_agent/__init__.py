#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
from openjiuwen.core.single_agent.schema.agent_card import AgentCard

# Schema classes
from openjiuwen.core.single_agent.schema.schema import (
    WorkflowSchema,
    PluginSchema
)

# ReAct agent
from openjiuwen.core.single_agent.agents.react_agent import (
    ReActAgent,
    ReActAgentConfig,
    create_react_agent_config
)

_AGENT_CARD_CLASSES = [
    "AgentCard"
]

_BASE_AGENT_CLASSES = [
    "BaseAgent",
    "ControllerAgent",
]

_AGENT_RUNTIME = [
    "AgentSession",
]

_AGENT_FACTORIES = [
    "WorkflowFactory",
    "workflow_provider",
]

_REACT_AGENT_CLASSES = [
    "ReActAgent",
]

_REACT_AGENT_FUNCTIONS = [
    "create_react_agent_config"
]

_CONFIG_CLASSES = [
    "AgentConfig",
    "ReActAgentConfig",
    "LLMCallConfig",
    "IntentDetectionConfig",
    "ConstrainConfig",
    "DefaultResponse",
    "WorkflowAgentConfig"
]

_SCHEMA_CLASSES = [
    "WorkflowSchema",
    "PluginSchema"
]

__all__ = (
        _AGENT_CARD_CLASSES +
        _BASE_AGENT_CLASSES +
        _AGENT_RUNTIME +
        _AGENT_FACTORIES +
        _REACT_AGENT_CLASSES +
        _REACT_AGENT_FUNCTIONS +
        _CONFIG_CLASSES +
        _SCHEMA_CLASSES
)