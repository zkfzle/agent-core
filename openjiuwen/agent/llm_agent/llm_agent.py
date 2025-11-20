#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import Dict, List, Any, AsyncIterator

from openjiuwen.agent.common.enum import ControllerType
from openjiuwen.agent.common.schema import WorkflowSchema, PluginSchema
from openjiuwen.agent.config.react_config import ReActAgentConfig
from openjiuwen.agent.llm_agent.llm_message_handler import ReActMessageHandler
from openjiuwen.core.agent.agent import Agent
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.runtime.config import Config
from openjiuwen.core.runtime.runtime import Runtime, Workflow
from openjiuwen.core.utils.tool.base import Tool


def create_react_llm_agent_config(agent_id: str,
                              agent_version: str,
                              description: str,
                              workflows: List[WorkflowSchema],
                              plugins: List[PluginSchema],
                              model: ModelConfig,
                              prompt_template: List[Dict],
                              tools: List[str] = []):
    config = ReActAgentConfig(id=agent_id,
                              version=agent_version,
                              description=description,
                              workflows=workflows,
                              plugins=plugins,
                              model=model,
                              prompt_template=prompt_template,
                              tools=tools)
    return config


def create_react_llm_agent(agent_config: ReActAgentConfig,
                       workflows: List[Workflow] = None,
                       tools: List[Tool] = None):
    agent = ReActLLMAgent(agent_config)
    agent.bind_workflows(workflows)
    agent.bind_tools(tools or [])
    return agent


class ReActLLMAgent(Agent):
    """ReAct mode Agent - uses LLM reasoning to generate execution plans"""
    
    def __init__(self, agent_config: ReActAgentConfig):
        # Validate controller_type
        if agent_config.controller_type != ControllerType.ReActController:
            raise NotImplementedError(f"ReActAgent requires ReActController, got {agent_config.controller_type}")

        # Create configuration and initialize base class
        config = Config()
        config.set_agent_config(agent_config=agent_config)
        super().__init__(config)
        
        # Set message handler
        self.set_message_handler(ReActMessageHandler)

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """Batch invoke - use base class's generic implementation"""
        return await self.controller_invoke(inputs, runtime)

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """Stream invoke - use base class's generic implementation"""
        async for result in self.controller_stream(inputs, runtime):
            yield result
