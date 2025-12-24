#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
ChatAgent - simplest llm chat single_agent
"""
from typing import Dict, Any, List, AsyncIterator

from openjiuwen.core.single_agent.config import LLMCallConfig
from openjiuwen.agent_builder.tune.chat_agent.chat_config import ChatAgentConfig
from openjiuwen.core.single_agent.agent import BaseAgent
from openjiuwen.core.common.utils.hash_util import generate_key
from openjiuwen.core.context_engine.schema.config import ContextEngineConfig
from openjiuwen.core.context_engine.context_engine import ContextEngine
from openjiuwen.core.operator.llm_call import LLMCall
from openjiuwen.core.session.runtime import Runtime
from openjiuwen.core.foundation.llm.model_utils.model_factory import ModelFactory
from openjiuwen.core.foundation.tool import Tool


def create_chat_agent_config(agent_id: str,
                             agent_version: str,
                             description: str,
                             model: LLMCallConfig
                             ):
    config = ChatAgentConfig(id=agent_id,
                             version=agent_version,
                             description=description,
                             model=model
                             )
    return config


def create_chat_agent(agent_config: ChatAgentConfig,
                      tools: List[Tool] = None):
    agent = ChatAgent(agent_config)
    agent.add_tools(tools or [])
    return agent


class ChatAgent(BaseAgent):
    def __init__(self, agent_config: ChatAgentConfig):
        # Initialize BaseAgent
        super().__init__(agent_config)
        
        # Initialize LLM Call
        llm_config = agent_config.model
        self._llm_call = LLMCall(
            llm_config.model.model_info.model_name,
            self._init_model(llm_config.model),
            llm_config.system_prompt,
            llm_config.user_prompt,
            llm_config.freeze_system_prompt,
            llm_config.freeze_user_prompt
        )

    def _init_model(self, model_config):
        """Initialize model"""
        model_id = generate_key(
            model_config.model_info.api_key,
            model_config.model_info.api_base,
            model_config.model_provider
        )

        model = self._runtime.get_model(model_id=model_id)

        if model is None:
            model = ModelFactory().get_model(
                model_provider=model_config.model_provider,
                api_base=model_config.model_info.api_base,
                api_key=model_config.model_info.api_key
            )
            self._runtime.add_model(model_id=model_id, model=model)

        return self._runtime.get_model(model_id=model_id)

    def _create_context_engine(self) -> ContextEngine:
        """ChatAgent uses default configured ContextEngine"""
        context_config = ContextEngineConfig()
        return ContextEngine(
            agent_id=self.agent_config.id,
            config=context_config,
        )

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        # 1. init ContextEngine and Runtime
        session_id = inputs.pop("conversation_id", "default_session")

        if runtime is None:
            # Compatible with old usage without runtime
            agent_runtime = await self._runtime.pre_run(session_id=session_id)
        else:
            agent_runtime = runtime

        # 2. invoke LLMCall
        agent_context = self.context_engine.get_agent_context(session_id)
        result = await self._llm_call.invoke(
            inputs=inputs,
            runtime=agent_runtime,
            history=agent_context.get_messages(),
            tools=self._runtime.get_tool_info()
        )
        if runtime is None:
            await agent_runtime.post_run()
        return dict(output=result.content, tool_calls=result.tool_calls)

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        # 1. init ContextEngine and Runtime
        session_id = inputs.pop("conversation_id", "default_session")

        if runtime is None:
            # Compatible with old usage without runtime
            agent_runtime = await self._runtime.pre_run(session_id=session_id)
        else:
            agent_runtime = runtime

        # 2. stream invoke LLMCall
        agent_context = self.context_engine.get_agent_context(session_id)
        stream_iterator = self._llm_call.stream(
            inputs=inputs,
            runtime=agent_runtime,
            history=agent_context.get_messages(),
            tools=self._runtime.get_tool_info()
        )
        if runtime is None:
            await agent_runtime.post_run()
        async for result in stream_iterator:
            yield dict(output=result.content, tool_calls=result.tool_calls)

    def get_llm_calls(self) -> Dict:
        return dict(llm_call=self._llm_call)

    def copy(self) -> "BaseAgent":
        return create_chat_agent(self.agent_config)