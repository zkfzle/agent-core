#!/usr/bin/env python
# coding: utf-8
"""
Super Agent Agent Module
"""

from openjiuwen.core.application.agents_for_studio import (
    SuperAgentConfig,
    SuperAgentFactory,
    AgentConstraints
)
from examples.super_agent import (
    SuperReActAgent
)
from openjiuwen.core.application.agents_for_studio import (
    ContextManager
)
from openjiuwen.core.application.agents_for_studio import (
    O3Handler
)
from openjiuwen.core.application.agents_for_studio import (
    ToolCallHandler
)

__all__ = [
    # Config
    "SuperAgentConfig",
    "SuperAgentFactory",
    "AgentConstraints",
    # Agent
    "SuperReActAgent",
    # Managers and Handlers
    "ContextManager",
    "O3Handler",
    "ToolCallHandler",
    # Prompt templates
    "get_summary_prompt",
    "get_o3_hints_prompt",
    "get_o3_answer_type_prompt",
    "get_o3_final_answer_prompt",
    "get_task_instruction_prompt",
    "get_main_agent_system_prompt",
    "get_browsing_agent_system_prompt",
    "generate_mcp_system_prompt",
    "process_input",
]

