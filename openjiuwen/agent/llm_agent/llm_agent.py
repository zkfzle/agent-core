#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""LLMAgent - ReAct style Agent based on ControllerAgent"""

from typing import Dict, List, Any, AsyncIterator
from openjiuwen.agent.common.enum import ControllerType
from openjiuwen.agent.common.schema import WorkflowSchema, PluginSchema
from openjiuwen.agent.config.react_config import ReActAgentConfig
from openjiuwen.agent.llm_agent.llm_controller import LLMController
from openjiuwen.core.agent.agent import ControllerAgent
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.workflow.base import Workflow
import asyncio


def create_llm_agent_config(agent_id: str,
                            agent_version: str,
                            description: str,
                            workflows: List[WorkflowSchema],
                            plugins: List[PluginSchema],
                            model: ModelConfig,
                            prompt_template: List[Dict],
                            tools: List[str] = []):
    """Create LLM Agent configuration - backward compatible factory function"""
    config = ReActAgentConfig(id=agent_id,
                              version=agent_version,
                              description=description,
                              workflows=workflows,
                              plugins=plugins,
                              model=model,
                              prompt_template=prompt_template,
                              tools=tools)
    return config


def create_llm_agent(agent_config: ReActAgentConfig,
                     workflows: List[Workflow] = None,
                     tools: List[Tool] = None):
    """Create LLM Agent - backward compatible factory function"""
    agent = LLMAgent(agent_config)
    agent.add_workflows(workflows)
    agent.add_tools(tools or [])
    return agent


class LLMAgent(ControllerAgent):
    """LLM Agent - ReAct style Agent based on new architecture
    
    Core features:
    1. Inherits ControllerAgent, holds LLMController
    2. Uses message queue pattern to process messages
    3. Supports LLM reasoning to generate task plans
    4. Supports multi-round conversations and task execution
    """

    def __init__(self, agent_config: ReActAgentConfig):
        """Initialize LLMControllerAgent
        
        Args:
            agent_config: ReAct Agent configuration
        """
        # Validate controller_type
        if agent_config.controller_type != ControllerType.ReActController:
            raise NotImplementedError(
                f"LLMControllerAgent requires ReActController, "
                "got {agent_config.controller_type}"
            )

        # Initialize base class (pass controller)
        super().__init__(agent_config, controller=None)

        self.controller = LLMController(
            config=agent_config,
            context_engine=self.context_engine,
            runtime=self._runtime
        )

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """Synchronous call - fully delegate to controller
        
        Args:
            inputs: Input data, contains query and conversation_id
            runtime: Runtime instance (optional)
            
        Returns:
            Execution result
        """
        # Fully delegate to ControllerAgent implementation
        return await super().invoke(inputs, runtime)

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """Streaming invocation - Fully delegate to controller

        Args:
            inputs: Input data
            runtime: Runtime instance (if None, auto create)

        Yields:
            Streaming output
        """
        if not self.controller:
            raise RuntimeError(
                f"{self.__class__.__name__} has no controller, "
                "subclass should create controller before invocation"
            )

        # If runtime not provided, create one
        session_id = inputs.get("conversation_id", "default_session")
        if runtime is None:
            agent_runtime = await self._runtime.pre_run(session_id=session_id)
            need_cleanup = True
        else:
            agent_runtime = runtime
            need_cleanup = False

        # Fully delegate to controller
        async def stream_process():
            try:
                await self.controller.invoke(inputs, agent_runtime)
            finally:
                if need_cleanup:
                    await agent_runtime.post_run()

        task = asyncio.create_task(stream_process())
        async for result in agent_runtime.stream_iterator():
            yield result
        await task
