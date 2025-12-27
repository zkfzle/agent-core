#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# Base agent classes
from openjiuwen.core.single_agent.agent import (
    BaseAgent,
    ControllerAgent,
    AgentRuntime,
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
    create_react_agent_config
)

_BASE_AGENT_CLASSES = [
    "BaseAgent",
    "ControllerAgent",
]

_AGENT_RUNTIME = [
    "AgentRuntime",
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
        _BASE_AGENT_CLASSES +
        _AGENT_RUNTIME +
        _AGENT_FACTORIES +
        _REACT_AGENT_CLASSES +
        _REACT_AGENT_FUNCTIONS +
        _CONFIG_CLASSES +
        _SCHEMA_CLASSES
)