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
from pathlib import Path

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

    def _compress_message_history(
        self,
        messages: List[Dict],
        max_chars: int = 30000,  # 30KB 限制
        max_messages: int = 30   # 最多保留 30 条消息（支持 ~15 轮工具调用）
    ) -> List[Dict]:
        """压缩消息历史，确保保持正确的消息序列

        关键设计：
        1. 一条 user 消息可能触发多个 assistant+tool 对
        2. 必须保留当前轮次的第一条 user 消息
        3. 按"对话轮次"截取，不破坏 assistant+tool 配对

        API 要求:
        1. system 消息后必须是 user 消息
        2. 每个 tool 消息必须跟在有 tool_calls 的 assistant 消息后

        Args:
            messages: List of message dicts
            max_chars: Maximum total characters allowed (default 30KB)
            max_messages: Maximum number of messages to keep (default 30)

        Returns:
            Compressed message list
        """
        # 1. 分离 system 消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        if not other_msgs:
            return system_msgs

        # 2. 找到所有 user 消息的位置（每个 user 消息开始一个新轮次）
        user_indices = [i for i, m in enumerate(other_msgs) if m.get("role") == "user"]

        if not user_indices:
            logger.error("No user message found in history! Adding placeholder.")
            other_msgs.insert(0, {"role": "user", "content": "[对话历史]"})
            user_indices = [0]

        # 3. 如果消息数量超过限制，按轮次截取
        if len(other_msgs) > max_messages:
            logger.warning(f"Message count ({len(other_msgs)}) exceeds limit ({max_messages}), truncating by rounds...")

            # 找到最后一个 user 消息的位置（当前轮次的开始）
            last_user_idx = user_indices[-1]

            # 计算当前轮次的消息数量
            current_round_msgs = len(other_msgs) - last_user_idx

            if current_round_msgs > max_messages:
                # 当前轮次消息太多，只保留当前轮次（从最后一个 user 开始）
                other_msgs = other_msgs[last_user_idx:]
                logger.warning(f"Current round has {current_round_msgs} messages, keeping only current round")
            else:
                # 可以保留一些历史轮次
                # 从后往前找，保留尽可能多的完整轮次
                remaining_slots = max_messages - current_round_msgs

                # 找到可以保留的历史轮次起点
                cut_idx = last_user_idx
                for i in range(len(user_indices) - 2, -1, -1):  # 从倒数第二个 user 开始
                    round_start = user_indices[i]
                    round_end = user_indices[i + 1] if i + 1 < len(user_indices) else last_user_idx
                    round_size = round_end - round_start

                    if round_size <= remaining_slots:
                        cut_idx = round_start
                        remaining_slots -= round_size
                    else:
                        break

                other_msgs = other_msgs[cut_idx:]
                logger.info(f"Keeping messages from index {cut_idx}, total {len(other_msgs)} messages")

        # 4. 压缩内容（不删除消息，只压缩内容）
        total_chars = sum(len(str(m.get("content", ""))) for m in other_msgs)

        if total_chars > max_chars:
            logger.warning(f"Message history too large ({total_chars} chars), compressing content...")

            # 找到当前轮次的起点（最后一个 user 消息）
            current_round_start = 0
            for i, m in enumerate(other_msgs):
                if m.get("role") == "user":
                    current_round_start = i

            # 保留当前轮次最近的消息完整，压缩历史
            keep_recent = min(5, len(other_msgs) - current_round_start)  # 当前轮次最近 5 条
            compressed = []

            for i, msg in enumerate(other_msgs):
                # 历史轮次的消息（当前轮次之前）
                if i < current_round_start:
                    if msg.get("role") == "tool":
                        content = msg.get("content", "")
                        if len(content) > 100:
                            msg = msg.copy()
                            msg["content"] = content[:100] + "\n...[历史工具结果已压缩]"
                    elif msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if len(content) > 150:
                            msg = msg.copy()
                            msg["content"] = content[:150] + "\n...[历史响应已压缩]"
                # 当前轮次的早期消息
                elif i < len(other_msgs) - keep_recent:
                    if msg.get("role") == "tool":
                        content = msg.get("content", "")
                        if len(content) > 200:
                            msg = msg.copy()
                            msg["content"] = content[:200] + "\n...[工具结果已压缩]"

                compressed.append(msg)

            other_msgs = compressed
            new_total = sum(len(str(m.get("content", ""))) for m in other_msgs)
            logger.info(f"Compressed: {total_chars} -> {new_total} chars")

        # 5. 最终验证：确保第一条消息是 user
        if other_msgs and other_msgs[0].get("role") != "user":
            logger.error(f"First message is {other_msgs[0].get('role')}, not 'user'! Fixing...")
            other_msgs.insert(0, {"role": "user", "content": "[对话历史已压缩]"})

        return system_msgs + other_msgs

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

        # 4. Compress message history if too large
        messages = self._compress_message_history(messages)

        # 5. Get available tool info
        tools = runtime.get_tool_info()

        # 6. Log message size for debugging
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        logger.info(f"Sending {len(messages)} messages with {total_chars} chars to LLM")

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

    def _has_pending_todos(self, session_id: str) -> tuple:
        """Check if there are pending todos that need to be completed

        Args:
            session_id: Session ID to find the todo file

        Returns:
            Tuple of (has_pending: bool, pending_todos: list)
        """
        try:
            todos_dir = Path.home() / ".jiuwen" / "todos"
            if not todos_dir.exists():
                logger.info(f"Todos dir does not exist: {todos_dir}")
                return False, []

            # Try exact match first
            todo_file = todos_dir / f"{session_id}.json"
            if todo_file.exists():
                with open(todo_file, 'r', encoding='utf-8') as file:
                    todos = json.load(file)

                if not todos:
                    logger.info("Todo file is empty")
                    return False, []

                pending_todos = []
                for todo in todos:
                    status = todo.get("status", "")
                    if status in ["pending", "in_progress"]:
                        pending_todos.append(todo.get('content', 'Unknown task'))

                if pending_todos:
                    logger.info(f"Found {len(pending_todos)} pending todos")
                    return True, pending_todos

                logger.info("All todos are completed")
                return False, []

            # Fallback: Find todo file containing session_id
            logger.info(f"Looking for todo file with session_id: {session_id}")
            for f in todos_dir.glob("*.json"):
                logger.info(f"Checking file: {f.name}")
                if session_id in f.name:
                    logger.info(f"Found matching file: {f}")
                    with open(f, 'r', encoding='utf-8') as file:
                        todos = json.load(file)

                    if not todos:
                        logger.info("Todo file is empty")
                        return False, []

                    pending_todos = []
                    for todo in todos:
                        status = todo.get("status", "")
                        if status in ["pending", "in_progress"]:
                            pending_todos.append(todo.get('content', 'Unknown task'))

                    if pending_todos:
                        logger.info(f"Found {len(pending_todos)} pending todos")
                        return True, pending_todos

                    logger.info("All todos are completed")
                    return False, []

            logger.info(f"No matching todo file found for session_id: {session_id}")
            return False, []
        except Exception as e:
            logger.error(f"Error checking pending todos: {e}")
            return False, []

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
        consecutive_no_tool_calls = 0  # Track consecutive iterations without tool calls
        max_consecutive_no_tool = 5    # Max allowed consecutive no-tool iterations (从 3 改为 5)

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

            # If no tool calls, check if we should continue or finish
            if not full_tool_calls:
                consecutive_no_tool_calls += 1

                # Get session_id from inputs
                session_id = inputs.get("conversation_id", "default_session")

                # Check if there are pending todos
                has_pending, pending_todos = self._has_pending_todos(session_id)

                if has_pending:
                    # Check if we've exceeded max consecutive no-tool iterations
                    if consecutive_no_tool_calls >= max_consecutive_no_tool:
                        logger.warning(f"Exceeded max consecutive no-tool iterations ({max_consecutive_no_tool}), forcing completion")
                        event_index += 1
                        yield OutputSchema(
                            type="warning",
                            index=event_index,
                            payload={
                                "message": f"警告：连续 {consecutive_no_tool_calls} 次未调用工具，但仍有 {len(pending_todos)} 个未完成任务。",
                                "pending_todos": pending_todos
                            }
                        )
                        # Continue anyway to give LLM another chance

                    # There are pending tasks, continue the loop
                    # Add a system message to remind LLM to continue
                    logger.info("No tool calls but pending todos exist, continuing loop")

                    # Build a stronger reminder message with specific pending tasks
                    pending_list = "\n".join([f"  - {t}" for t in pending_todos[:5]])  # Show up to 5 tasks
                    reminder = (
                        f"⚠️ 【重要系统提醒】你还有 {len(pending_todos)} 个未完成的任务！\n\n"
                        f"待完成任务：\n{pending_list}\n\n"
                        "【强制要求】你必须立即调用工具来完成这些任务。\n"
                        "- 不要只是回复文字说明\n"
                        "- 不要总结或解释\n"
                        "- 必须调用工具（如 write_file、edit_file、bash 等）\n"
                        "- 现在就开始执行下一个任务！"
                    )
                    MessageUtils.add_user_message(reminder, self.context_engine, runtime)

                    continue

                # No pending todos, task completed
                event_index += 1
                yield OutputSchema(
                    type="answer",
                    index=event_index,
                    payload={"output": full_content, "result_type": "answer"}
                )
                return
            else:
                # Reset counter when tool calls are made
                consecutive_no_tool_calls = 0

            # Execute tool calls with events
            # Check if we have multiple task tool calls for parallel execution
            task_calls = [tc for tc in full_tool_calls if tc.name == "task"]
            other_calls = [tc for tc in full_tool_calls if tc.name != "task"]

            # Execute non-task tools sequentially first
            for tool_call in other_calls:
                tool_name = tool_call.name
                try:
                    tool_args = (
                        json.loads(tool_call.arguments)
                        if isinstance(tool_call.arguments, str)
                        else tool_call.arguments
                    )
                except (json.JSONDecodeError, AttributeError):
                    tool_args = {}

                event_index += 1
                yield OutputSchema(
                    type="tool_call",
                    index=event_index,
                    payload={"tool_call": {"id": tool_call.id, "name": tool_name, "arguments": tool_args}}
                )

                try:
                    result = await self._execute_tool_call(tool_call, runtime)
                    event_index += 1
                    yield OutputSchema(
                        type="tool_result",
                        index=event_index,
                        payload={"tool_result": {"name": tool_name, "result": str(result), "success": True}}
                    )
                except Exception as e:
                    event_index += 1
                    yield OutputSchema(
                        type="tool_result",
                        index=event_index,
                        payload={"tool_result": {"name": tool_name, "result": str(e), "success": False}}
                    )

            # Execute task tools in parallel if multiple
            if len(task_calls) > 1:
                logger.info(f"Executing {len(task_calls)} task tools in parallel")
                # Yield all tool_call events first
                for tool_call in task_calls:
                    try:
                        tool_args = json.loads(tool_call.arguments) if isinstance(tool_call.arguments, str) else tool_call.arguments
                    except:
                        tool_args = {}
                    event_index += 1
                    yield OutputSchema(
                        type="tool_call",
                        index=event_index,
                        payload={"tool_call": {"id": tool_call.id, "name": "task", "arguments": tool_args}}
                    )

                # Execute all tasks in parallel
                async def execute_task(tc):
                    try:
                        return (tc, await self._execute_tool_call(tc, runtime), None)
                    except Exception as e:
                        return (tc, None, e)

                results = await asyncio.gather(*[execute_task(tc) for tc in task_calls])

                # Yield all results
                for tc, result, error in results:
                    event_index += 1
                    if error:
                        yield OutputSchema(
                            type="tool_result",
                            index=event_index,
                            payload={"tool_result": {"name": "task", "result": str(error), "success": False}}
                        )
                    else:
                        yield OutputSchema(
                            type="tool_result",
                            index=event_index,
                            payload={"tool_result": {"name": "task", "result": str(result), "success": True}}
                        )
            elif len(task_calls) == 1:
                # Single task tool, execute normally
                tool_call = task_calls[0]
                try:
                    tool_args = json.loads(tool_call.arguments) if isinstance(tool_call.arguments, str) else tool_call.arguments
                except:
                    tool_args = {}
                event_index += 1
                yield OutputSchema(
                    type="tool_call",
                    index=event_index,
                    payload={"tool_call": {"id": tool_call.id, "name": "task", "arguments": tool_args}}
                )
                try:
                    result = await self._execute_tool_call(tool_call, runtime)
                    event_index += 1
                    yield OutputSchema(
                        type="tool_result",
                        index=event_index,
                        payload={"tool_result": {"name": "task", "result": str(result), "success": True}}
                    )
                except Exception as e:
                    event_index += 1
                    yield OutputSchema(
                        type="tool_result",
                        index=event_index,
                        payload={"tool_result": {"name": "task", "result": str(e), "success": False}}
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
