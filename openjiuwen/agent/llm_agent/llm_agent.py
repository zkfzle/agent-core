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
from openjiuwen.core.common.logging import logger
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.stream.base import OutputSchema
from openjiuwen.core.utils.llm.messages import HumanMessage, AIMessage
from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.workflow.base import Workflow
from openjiuwen.core.memory.config.config import Config
from openjiuwen.core.memory.engine.memory_engine_factory import get_memengine_instance
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

        self._init_memory_config(agent_config.memory_config)

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
        # async write user message memory
        self._write_messages_to_memory(inputs)
        # Fully delegate to ControllerAgent implementation
        result = await super().invoke(inputs, runtime)
        await self._write_messages_to_memory(inputs, result)
        return result

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

        # async write user message memory
        user_memory_task = asyncio.create_task(self._write_messages_to_memory(inputs))

        task = asyncio.create_task(stream_process())
        result_for_memory = ""
        async for result in agent_runtime.stream_iterator():
            if (isinstance(result.payload, dict) and result.payload.get("result_type") == 'answer' and
                    isinstance(result.payload.get("output"), str)):
                result_for_memory += result.payload.get("output")
            yield result
        await task
        await user_memory_task
        if result_for_memory != "":
            agent_memory_task = asyncio.create_task(self._write_messages_to_memory(inputs, result_for_memory))
            await agent_memory_task


    def set_prompt_template(self, prompt_template: List[Dict]):
        self._agent_config.prompt_template = prompt_template
        self._config_wrapper.set_agent_config(self._agent_config)
        self._config = self._config_wrapper
        self.controller.set_llm_controller_prompt_template(prompt_template)

    def _init_memory_config(self, memory_config):
        app_id = f"{self._agent_config.id}_{self._agent_config.version}"
        logger.info(f"When init Memory Engine, app_id: {app_id}")
        if memory_config is not None:
            mem_manager_config = {}
            config = Config(**mem_manager_config)
            self._memory_engine = get_memengine_instance(config)
            if self._memory_engine:
                self._memory_engine.set_app_config(app_id, memory_config)

    async def _write_messages_to_memory(self, inputs, result = None):
        user_id = inputs.get("user_id")
        session_id = inputs.get("conversation_id", "default_session")
        app_id = inputs.get("app_id","default_app_id")

        if not user_id or not self._memory_engine:
            return

        # add ai response if exist
        if result is not None:
            if isinstance(result, OutputSchema) and result.type == "answer":
                response = result.payload.get("output")
                if response:
                    assistant_message = AIMessage(content=response)
            elif isinstance(result, str):
                assistant_message = AIMessage(content=result)
            if assistant_message is not None and assistant_message.content != "":
                await self._memory_engine.aadd_conversation_messages(user_id, app_id,[assistant_message])
            return

        #add user message
        if not isinstance(inputs, dict):
            return
        query = inputs.get("query")
        if query is not None and isinstance(query, str):
            user_message = HumanMessage(content=query)
        if user_message and user_message.content != "":
            await self._memory_engine.aadd_conversation_messages(user_id, app_id,[user_message])

