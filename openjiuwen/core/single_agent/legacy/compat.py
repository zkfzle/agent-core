#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Legacy Interface Compatibility Layer

This module provides adapters from old interfaces to new interfaces for smooth migration.
All old interfaces will be removed in v1.0.0.

Migration Guide:
    Old way:
        config = AgentConfig(id="my_agent", ...)
        agent = ReActAgent(agent_config=config, tools=[...])
        agent.add_tools([tool])
    
    New way:
        card = AgentCard(name="my_agent", ...)
        config = ReActAgentConfig(...)
        agent = ReActAgent(card=card).configure(config).add_ability([...])

Created on: 2025-01-04
"""
import warnings
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from openjiuwen.core.foundation.tool import Tool, ToolCard
    from openjiuwen.core.workflow import Workflow


def _deprecation_warning(old: str, new: str, version: str = "1.0.0"):
    """Issue deprecation warning
    
    Args:
        old: Old interface name
        new: New interface name
        version: Planned removal version
    """
    warnings.warn(
        f"{old} is deprecated and will be removed in v{version}. "
        f"Please use {new} instead. ",
        DeprecationWarning,
        stacklevel=3
    )


# ============================================================
# Legacy Interface Adapter Mixin (Add old methods to new Agent)
# ============================================================

class LegacyMethodsMixin:
    """Mixin to add legacy methods to new Agent
    
    Usage:
        class ReActAgent(LegacyMethodsMixin, BaseAgent):
            pass
    
    Note:
        - All methods will issue deprecation warnings
        - Internally delegates to new method implementations
    """
    
    def add_tools(self, tools: List) -> 'LegacyMethodsMixin':
        """[Deprecated] Add tools, please use add_ability()
        
        Args:
            tools: List of tools
        
        Returns:
            self (supports method chaining)
        """
        _deprecation_warning("add_tools()", "add_ability()")
        
        # Since current BaseAgent.add_tools requires Tool instances, keep original logic
        # Call parent's add_tools instead of add_ability
        if hasattr(super(), 'add_tools'):
            super().add_tools(tools)
        return self
    
    def add_workflows(self, workflows: List) -> 'LegacyMethodsMixin':
        """[Deprecated] Add workflows, please use add_ability()
        
        Args:
            workflows: List of workflows
        
        Returns:
            self (supports method chaining)
        """
        _deprecation_warning("add_workflows()", "add_ability()")
        
        # Since current BaseAgent.add_workflows requires Workflow instances, keep original logic
        if hasattr(super(), 'add_workflows'):
            super().add_workflows(workflows)
        return self
    
    def remove_tools(self, names: List[str]) -> 'LegacyMethodsMixin':
        """[Deprecated] Remove tools, please use remove_ability()
        
        Args:
            names: List of tool names
        
        Returns:
            self (supports method chaining)
        """
        _deprecation_warning("remove_tools()", "remove_ability()")
        
        # Currently no corresponding remove_tools implementation, return self for now
        # TODO: May need to implement remove_ability in future
        return self
    
    def remove_workflows(self, workflows: List) -> 'LegacyMethodsMixin':
        """[Deprecated] Remove workflows, please use remove_ability()
        
        Args:
            workflows: List of workflows (id, version tuples)
        
        Returns:
            self (supports method chaining)
        """
        _deprecation_warning("remove_workflows()", "remove_ability()")
        
        if hasattr(super(), 'remove_workflows'):
            super().remove_workflows(workflows)
        return self


# ============================================================
# Utility Function Compatibility
# ============================================================

def create_react_agent_config(*args, **kwargs):
    """[Deprecated] Create ReAct config
    
    Use ReActAgentConfig() constructor directly
    """
    _deprecation_warning(
        "create_react_agent_config()",
        "ReActAgentConfig()"
    )
    from openjiuwen.core.single_agent.agents import react_agent as ra_module
    # Call the original function in react_agent module
    return ra_module.create_react_agent_config(*args, **kwargs)

