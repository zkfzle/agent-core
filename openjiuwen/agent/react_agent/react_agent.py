# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
ReActAgent - Minimal ReAct Agent (no interruption, no Controller)

Modified by openjiuwen-code project to support:
- Streaming LLM output (content_chunk events)
- Tool call events (tool_call, tool_result)
"""

import json
import asyncio
from typing import Dict, Any, AsyncIterator, List, Optional, Tuple

from pydantic import ValidationError

from openjiuwen.core.agent.agent import BaseAgent
from openjiuwen.agent.config.react_config import ReActAgentConfig
from openjiuwen.agent.common.schema import WorkflowSchema, PluginSchema
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runtime.runtime import Runtime, Workflow
from openjiuwen.core.stream.base import OutputSchema
from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.messages import AIMessage, ToolMessage
from openjiuwen.core.utils.llm.messages_chunk import AIMessageChunk
from openjiuwen.core.utils.prompt.template.template import Template
from openjiuwen.agent.utils import MessageUtils


class ReActAgent(BaseAgent):
    """ReAct Agent - Minimal implementation (no interruption, no Controller)
    
    Enhanced with streaming support for content and tool events.
    """

    def __init__(
            self,
            agent_config: ReActAgentConfig
    ):
        """Initialize ReActAgent
        
        Args:
            agent_config: ReAct config
            workflows: Workflow list
            tools: Tool list
        """
        # Call parent init (BaseAgent creates runtime, context_engine, etc.)
        super().__init__(agent_config)

        # LLM instance (lazy creation)
        self._llm = None

    def _get_llm(self):
        """Get LLM instance"""
        if self._llm is None:
            self._llm = ModelFactory().get_model(
                model_provider=self.agent_config.model.model_provider,
                **self.agent_config.model.model_info.model_dump(exclude=['model_name', 'streaming'])
            )
        return self._llm

    def _prepare_messages(
        self, 
        user_input: str, 
        runtime: Runtime, 
        is_first_call: bool = False
    ) -> Tuple[List[Dict], List[Dict]]:
        """Prepare messages for LLM call
        
        Returns:
            Tuple of (messages, tools)
        """
        # 1. If first call, add user message
        if is_first_call:
            MessageUtils.add_user_message(user_input, self.context_engine, runtime)

        # 2. Get chat history
        chat_history = MessageUtils.get_chat_history(
            self.context_engine, runtime, self.agent_config
        )

        # 3. Format prompt
        messages = []
        # Add system prompt
        try:
            system_prompt = Template(
                content=self.agent_config.prompt_template
            ).to_messages()
            for prompt in system_prompt:
                prompt_dict = prompt.model_dump(exclude_none=True)
                messages.append(prompt_dict)
        except ValidationError as e:
            raise JiuWenBaseException(
                error_code=StatusCode.PROMPT_PARAMS_CHECK_ERROR.code,
                message=StatusCode.PROMPT_PARAMS_CHECK_ERROR.errmsg.format(msg=str(e))
            ) from e

        # Add chat history
        for msg in chat_history:
            msg_dict = msg.model_dump(exclude_none=True)
            messages.append(msg_dict)

        # 4. Get available tool info
        tools = runtime.get_tool_info()
        
        return messages, tools

    async def call_model(
        self, 
        user_input: str, 
        runtime: Runtime, 
        is_first_call: bool = False
    ):
        """Call LLM for reasoning (non-streaming)
        
        Args:
            user_input: User input or tool result
            runtime: Runtime instance
            is_first_call: Whether first call
        
        Returns:
            llm_output: LLM output (contains content and tool_calls)
        """
        messages, tools = self._prepare_messages(user_input, runtime, is_first_call)

        # Call LLM
        llm = self._get_llm()
        llm_output = await llm.ainvoke(
            self.agent_config.model.model_info.model_name,
            messages,
            tools
        )

        # Save AI response to chat history
        ai_message = AIMessage(
            content=llm_output.content,
            tool_calls=llm_output.tool_calls
        )
        MessageUtils.add_ai_message(ai_message, self.context_engine, runtime)

        return llm_output

    async def call_model_stream(
        self, 
        user_input: str, 
        runtime: Runtime, 
        is_first_call: bool = False
    ) -> AsyncIterator[AIMessageChunk]:
        """Call LLM for reasoning with streaming output
        
        Args:
            user_input: User input or tool result
            runtime: Runtime instance
            is_first_call: Whether first call
        
        Yields:
            AIMessageChunk: Streaming chunks from LLM
        """
        messages, tools = self._prepare_messages(user_input, runtime, is_first_call)

        # Call LLM with streaming
        llm = self._get_llm()
        
        # Accumulate full message for chat history
        full_content = ""
        full_tool_calls = []
        
        async for chunk in llm.astream(
            self.agent_config.model.model_info.model_name,
            messages,
            tools
        ):
            # Accumulate content
            if chunk.content:
                full_content += chunk.content
            
            # Accumulate tool calls
            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    # Check if this is continuation of existing tool call
                    found = False
                    for existing in full_tool_calls:
                        if (existing.id and tc.id and existing.id == tc.id) or \
                           (not existing.id or not tc.id):
                            existing.id = existing.id or tc.id
                            existing.name = (existing.name or "") + (tc.name or "")
                            existing.arguments = (
                                (existing.arguments or "") + (tc.arguments or "")
                            )
                            found = True
                            break
                    if not found:
                        full_tool_calls.append(tc)
            
            yield chunk
        
        # Save full AI response to chat history
        ai_message = AIMessage(
            content=full_content,
            tool_calls=full_tool_calls if full_tool_calls else None
        )
        MessageUtils.add_ai_message(ai_message, self.context_engine, runtime)

    async def _execute_tool_call(self, tool_call, runtime: Runtime) -> Any:
        """Execute single tool call
        
        Args:
            tool_call: Tool call object returned by LLM
            runtime: Runtime instance
        
        Returns:
            Tool execution result
        """
        # Parse tool name and parameters
        tool_name = tool_call.name
        try:
            tool_args = (
                json.loads(tool_call.arguments) 
                if isinstance(tool_call.arguments, str) 
                else tool_call.arguments
            )
        except (json.JSONDecodeError, AttributeError):
            tool_args = {}

        # Get and execute tool
        tool = runtime.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        result = await tool.ainvoke(tool_args)

        # Add tool result to chat history
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call.id
        )
        MessageUtils.add_tool_message(tool_message, self.context_engine, runtime)

        return result

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """Sync call - Complete ReAct loop (non-streaming)
        
        Args:
            inputs: Input data, must contain 'query' field
            runtime: Optional Runtime
        
        Returns:
            Execution result
        """
        # 1. Prepare Runtime
        session_id = inputs.get("conversation_id", "default_session")
        runtime_created = False
        if runtime is None:
            runtime = await self._runtime.pre_run(
                session_id=session_id, inputs=inputs
            )
            runtime_created = True

        # Register tools to runtime
        if hasattr(self, '_tools') and self._tools:
            tools_to_add = [(tool.name, tool) for tool in self._tools]
            runtime.add_tools(tools_to_add)

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
                    runtime,
                    is_first_call=is_first_call
                )
                is_first_call = False

                # 2.2 If no tool calls, task completed
                if not llm_output.tool_calls:
                    logger.info("No tool calls, task completed")
                    return {
                        "output": llm_output.content,
                        "result_type": "answer"
                    }

                # 2.3 Execute tool calls
                for tool_call in llm_output.tool_calls:
                    tool_name = tool_call.name
                    logger.info(f"Executing tool: {tool_name}")
                    result = await self._execute_tool_call(tool_call, runtime)
                    logger.info(f"Tool {tool_name} completed")

            # 3. Exceeded max iteration
            logger.warning(f"Exceeded max iteration {max_iteration}")
            return {
                "output": "Exceeded max iteration",
                "result_type": "error"
            }
        finally:
            if runtime_created:
                await runtime.post_run()

    async def invoke_stream(
        self, 
        inputs: Dict, 
        runtime: Runtime
    ) -> AsyncIterator[OutputSchema]:
        """Streaming ReAct loop - yields events as they occur
        
        Args:
            inputs: Input data with 'query' field
            runtime: Runtime instance
        
        Yields:
            OutputSchema: Events (content_chunk, tool_call, tool_result, answer)
        """
        # Register tools to runtime
        if hasattr(self, '_tools') and self._tools:
            tools_to_add = [(tool.name, tool) for tool in self._tools]
            runtime.add_tools(tools_to_add)

        user_input = inputs.get("query", "")
        if not user_input:
            yield OutputSchema(
                type="error",
                index=0,
                payload={"error": "No query provided", "result_type": "error"}
            )
            return

        # Send thinking event
        yield OutputSchema(
            type="thinking",
            index=0,
            payload={"status": "thinking"}
        )

        # ReAct loop
        iteration = 0
        max_iteration = self.agent_config.constrain.max_iteration
        is_first_call = True
        event_index = 0

        while iteration < max_iteration:
            iteration += 1

            # Call model with streaming
            full_content = ""
            full_tool_calls = []
            
            async for chunk in self.call_model_stream(
                user_input, runtime, is_first_call=is_first_call
            ):
                # Yield content chunks for streaming display
                if chunk.content:
                    full_content += chunk.content
                    event_index += 1
                    yield OutputSchema(
                        type="content_chunk",
                        index=event_index,
                        payload={"content": chunk.content}
                    )
                
                # Accumulate tool calls
                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        found = False
                        for existing in full_tool_calls:
                            if existing.id == tc.id or (not existing.id or not tc.id):
                                existing.id = existing.id or tc.id
                                existing.name = (existing.name or "") + (tc.name or "")
                                existing.arguments = (
                                    (existing.arguments or "") + (tc.arguments or "")
                                )
                                found = True
                                break
                        if not found:
                            full_tool_calls.append(tc)

            is_first_call = False

            # If no tool calls, task completed
            if not full_tool_calls:
                event_index += 1
                yield OutputSchema(
                    type="answer",
                    index=event_index,
                    payload={"output": full_content, "result_type": "answer"}
                )
                return

            # Execute tool calls with events
            for tool_call in full_tool_calls:
                tool_name = tool_call.name
                try:
                    tool_args = (
                        json.loads(tool_call.arguments) 
                        if isinstance(tool_call.arguments, str) 
                        else tool_call.arguments
                    )
                except (json.JSONDecodeError, AttributeError):
                    tool_args = {}

                # Yield tool_call event (tool starting)
                event_index += 1
                yield OutputSchema(
                    type="tool_call",
                    index=event_index,
                    payload={
                        "tool_call": {
                            "id": tool_call.id,
                            "name": tool_name,
                            "arguments": tool_args
                        }
                    }
                )

                # Execute tool
                try:
                    result = await self._execute_tool_call(tool_call, runtime)
                    
                    # Yield tool_result event (tool completed)
                    event_index += 1
                    yield OutputSchema(
                        type="tool_result",
                        index=event_index,
                        payload={
                            "tool_result": {
                                "name": tool_name,
                                "result": str(result),
                                "success": True
                            }
                        }
                    )
                except Exception as e:
                    # Tool execution failed
                    event_index += 1
                    yield OutputSchema(
                        type="tool_result",
                        index=event_index,
                        payload={
                            "tool_result": {
                                "name": tool_name,
                                "result": str(e),
                                "success": False
                            }
                        }
                    )

        # Exceeded max iteration
        yield OutputSchema(
            type="error",
            index=event_index + 1,
            payload={"error": "Exceeded max iteration", "result_type": "error"}
        )

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """Stream call - with streaming LLM output and tool events
        
        Yields OutputSchema events:
        - thinking: Agent is thinking
        - content_chunk: Streaming text content from LLM
        - tool_call: Tool is being called (with name and arguments)
        - tool_result: Tool execution completed (with result)
        - answer: Final answer
        - error: Error occurred
        """
        # Prepare runtime
        session_id = inputs.get("conversation_id", "default_session")
        if runtime is None:
            agent_runtime = await self._runtime.pre_run(
                session_id=session_id, inputs=inputs
            )
            need_cleanup = True
        else:
            agent_runtime = runtime
            need_cleanup = False

        try:
            # Use streaming invoke method
            async for event in self.invoke_stream(inputs, agent_runtime):
                yield event
                
        except Exception as e:
            logger.error(f"ReActAgent stream error: {e}")
            yield OutputSchema(
                type="error",
                index=0,
                payload={"error": str(e), "result_type": "error"}
            )
        finally:
            if need_cleanup:
                await agent_runtime.post_run()


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
