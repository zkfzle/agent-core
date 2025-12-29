#!/usr/bin/env python
# coding: utf-8
"""
Super Agent Package
Enhanced ReAct Agent with custom context management
"""

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
    # LLM
    "OpenRouterLLM",
    "OpenRouterConfig",
    "ContextLimitError",
]

