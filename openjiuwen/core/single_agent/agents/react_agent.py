#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
ReActAgent - Minimal ReAct Agent (no interruption, no Controller)
"""

import asyncio
import json
from typing import Dict, Any, AsyncIterator, List

from pydantic import ValidationError, Field

from openjiuwen.core.common.constants.enums import ControllerType
from openjiuwen.core.common.utils.message_utils import MessageUtils
from openjiuwen.core.memory.config.config import MemoryConfig
from openjiuwen.core.single_agent.agent import BaseAgent
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.single_agent.config import AgentConfig, ConstrainConfig
from openjiuwen.core.single_agent.schema.schema import PluginSchema
from openjiuwen.core.foundation.llm import ModelConfig
from openjiuwen.core.workflow import Workflow
from openjiuwen.core.session import Session
from openjiuwen.core.session.stream import OutputSchema
from openjiuwen.core.foundation.llm.messages import AIMessage, ToolMessage
from openjiuwen.core.foundation.llm.model_utils.model_factory import ModelFactory
from openjiuwen.core.foundation.prompt import PromptTemplate
from openjiuwen.core.foundation.tool import Tool


class ReActAgentConfig(AgentConfig):
    """ReAct Agent configuration"""
    controller_type: ControllerType = Field(default=ControllerType.ReActController)
    prompt_template_name: str = Field(default="react_system_prompt")
    prompt_template: List[Dict] = Field(default_factory=list)
    constrain: ConstrainConfig = Field(default=ConstrainConfig())
    plugins: List[PluginSchema] = Field(default_factory=list)
    memory_config: MemoryConfig = Field(default=MemoryConfig())


class ReActAgent(BaseAgent):
    """ReAct Agent - Minimal implementation (no interruption, no Controller)
    """

    def __init__(
            self,
            agent_config: ReActAgentConfig,
            workflows: List[Workflow] = None,
            tools: List[Tool] = None
    ):
        """Initialize ReActAgent
        
        Args:
            agent_config: ReAct config
            workflows: Workflow list
            tools: Tool list
        """
        # Call parent init (BaseAgent creates session, context_engine, etc.)
        super().__init__(agent_config)

        # LLM instance (lazy creation)
        self._llm = None

        # 通过 BaseAgent 的接口添加 tools 和 workflows（自动同步）
        if tools:
            self.add_tools(tools)
        if workflows:
            self.add_workflows(workflows)

    def _get_llm(self):
        """Get LLM instance"""
        if self._llm is None:
            self._llm = ModelFactory().get_model(
                model_provider=self.agent_config.model.model_provider,
                **self.agent_config.model.model_info.model_dump(exclude=['model_name', 'streaming'])
            )
        return self._llm

    async def call_model(self, user_input: str, session: Session, is_first_call: bool = False):
        """Call LLM for reasoning
        
        Args:
            user_input: User input or tool result
            session: Session instance
            is_first_call: Whether first call (first call needs to add user message)
        
        Returns:
            llm_output: LLM output (contains content and tool_calls)
        """
        # 1. If first call, add user message
        if is_first_call:
            MessageUtils.add_user_message(user_input, self.context_engine, session)

        # 2. Get chat history
        chat_history = MessageUtils.get_chat_history(
            self.context_engine, session, self.agent_config
        )

        # 3. Format prompt
        messages = []
        # Add system prompt
        try:
            system_prompt = PromptTemplate(content=self.agent_config.prompt_template).to_messages()
            for prompt in system_prompt:
                prompt_dict = prompt.model_dump(exclude_none=True)
                messages.append(prompt_dict)
        except ValidationError as e:
            raise JiuWenBaseException(
                error_code=StatusCode.PROMPT_PARAMS_CHECK_ERROR.code,
                message=StatusCode.PROMPT_PARAMS_CHECK_ERROR.errmsg.format(msg=str(e))
            ) from e

        # Add chat history (need complete BaseMessage conversion)
        for msg in chat_history:
            # Use model_dump to export message completely, exclude None values
            msg_dict = msg.model_dump(exclude_none=True)
            messages.append(msg_dict)

        # 4. Get available tool info
        tools = session.get_tool_info()

        # 5. Call LLM
        llm = self._get_llm()
        llm_output = await llm.ainvoke(
            self.agent_config.model.model_info.model_name,
            messages,
            tools
        )

        # 6. Save AI response to chat history
        ai_message = AIMessage(
            content=llm_output.content,
            tool_calls=llm_output.tool_calls
        )
        MessageUtils.add_ai_message(ai_message, self.context_engine, session)

        return llm_output

    async def _execute_tool_call(self, tool_call, session: Session) -> Any:
        """Execute single tool call
        
        Args:
            tool_call: Tool call object returned by LLM
            session: Session instance
        
        Returns:
            Tool execution result
        """
        # Parse tool name and parameters
        tool_name = tool_call.name
        try:
            tool_args = json.loads(tool_call.arguments) if isinstance(tool_call.arguments, str) else tool_call.arguments
        except (json.JSONDecodeError, AttributeError):
            tool_args = {}

        # Get and execute tool
        tool = session.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        result = await tool.invoke(tool_args)

        # Add tool result to chat history
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call.id
        )
        MessageUtils.add_tool_message(tool_message, self.context_engine, session)

        return result

    async def invoke(self, inputs: Dict, session: Session = None) -> Dict:
        """Sync call - Complete ReAct loop
        
        Args:
            inputs: Input data, must contain 'query' field
            session: Optional Session (if not provided, use BaseAgent's _session)
        
        Returns:
            Execution result
        """
        # 1. Prepare Session
        session_id = inputs.get("conversation_id", "default_session")
        session_created = False
        if session is None:
            # Use BaseAgent's _session, need to create task session
            session = await self._session.pre_run(session_id=session_id, inputs=inputs)
            session_created = True

        try:
            user_input = inputs.get("query", "")
            if not user_input:
                return {"output": "No query provided", "result_type": "error"}

            # 2. ReAct loop
            iteration = 0
            max_iteration = self.agent_config.constrain.max_iteration
            is_first_call = True

            while iteration < max_iteration:
                iteration += 1
                logger.info(f"ReAct iteration {iteration}")

                # 2.1 Call model for reasoning
                llm_output = await self.call_model(
                    user_input,
                    session,
                    is_first_call=is_first_call
                )
                is_first_call = False  # Set to False after first call

                # 2.2 If no tool calls, LLM thinks problem is solved
                if not llm_output.tool_calls:
                    logger.info("No tool calls, task completed")
                    return {
                        "output": llm_output.content,
                        "result_type": "answer"
                    }

                # 2.3 Execute tool calls (tool results already added to history in _execute_tool_call)
                for tool_call in llm_output.tool_calls:
                    tool_name = tool_call.name
                    logger.info(f"Executing tool: {tool_name}")
                    result = await self._execute_tool_call(tool_call, session)
                    logger.info(f"Tool {tool_name} completed with result: {result}")

            # 3. Exceeded max iteration count
            logger.warning(f"Exceeded max iteration {max_iteration}")
            return {
                "output": "Exceeded max iteration",
                "result_type": "error"
            }
        finally:
            # 4. Cleanup session (if we created it)
            if session_created:
                await session.post_run()

    async def stream(self, inputs: Dict, session: Session = None) -> AsyncIterator[Any]:
        """Stream call - minimal version
        
        Note:
            When external session is provided, data is written to it but not read
            from stream_iterator (to avoid nested read deadlock). External caller
            reads stream data from session.
        """
        # Prepare session
        session_id = inputs.get("conversation_id", "default_session")
        if session is None:
            # Use BaseAgent's _session, need to create task session
            agent_session = await self._session.pre_run(
                session_id=session_id, inputs=inputs
            )
            need_cleanup = True
            own_stream = True  # Owns stream lifecycle
        else:
            agent_session = session
            need_cleanup = False
            own_stream = False  # External owns stream lifecycle

            # Sync single_agent's tools to external session
            # When external session is provided, single_agent's tools need to be registered
            if hasattr(self, '_tools') and self._tools:
                tools_to_add = [(tool.name, tool) for tool in self._tools]
                agent_session.add_tools(tools_to_add)

        # Store final result for send_to_agent
        final_result_holder = {"result": None}

        async def stream_process():
            try:
                final_result = await self.invoke(inputs, agent_session)
                final_result_holder["result"] = final_result
                await agent_session.write_stream(OutputSchema(
                    type="answer",
                    index=0,
                    payload={"output": final_result, "result_type": "answer"}
                ))
            except Exception as e:
                logger.error(f"ReActAgent stream error: {e}")
            finally:
                # Cleanup session (if we created it)
                if need_cleanup:
                    await agent_session.post_run()

        task = asyncio.create_task(stream_process())

        if own_stream:
            # Read from stream_iterator only when owning stream
            # External caller reads if external session provided
            async for result in agent_session.stream_iterator():
                yield result

        await task

        # When own_stream=False, yield final result to send_to_agent
        # so send_to_agent can get single_agent's actual return value
        if not own_stream and final_result_holder["result"] is not None:
            yield final_result_holder["result"]


# ===== Factory Functions =====
def create_react_agent_config(
        agent_id: str,
        agent_version: str,
        description: str,
        model: ModelConfig,
        prompt_template: List[Dict]
) -> ReActAgentConfig:
    """Create ReAct Agent config
    
    Args:
        agent_id: Agent ID
        agent_version: Agent version
        description: Agent description
        model: Model config
        prompt_template: Prompt template
    
    Returns:
        ReActAgentConfig instance
    """
    return ReActAgentConfig(
        id=agent_id,
        version=agent_version,
        description=description,
        model=model,
        prompt_template=prompt_template
    )
