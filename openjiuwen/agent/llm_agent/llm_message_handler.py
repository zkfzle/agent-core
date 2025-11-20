#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.agent.controller.scheduler.message_handler import MessageHandler, MessageHandlerResult
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.agent.message.message import MessageType
from openjiuwen.core.agent.controller.utils import MessageHandlerUtils
from openjiuwen.agent.utils import MessageUtils
from openjiuwen.core.common.security.user_config import UserConfig


class ReActMessageHandler(MessageHandler):
    """ReActMessageHandler - ReAct mode message handler with ReAct state management"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        super().__init__(config, context_engine, runtime)
        self.load_state()  # Load state from runtime (base class implementation)
        self.iteration = 0  # Reasoning iteration count

    def get_state_key(self) -> str:
        """Return react handler state storage key"""
        return "react_controller_state"

    async def handle_message(self, message: Message) -> MessageHandlerResult:
        """Handle react mode messages and return processing results
        
        Message distribution logic:
        - TASK_COMPLETED: Call LLM for reasoning again to decide whether to continue
        - TASK_INTERRUPTED: Use base class general interrupt handling
        - USER_INPUT: Call LLM reasoning to generate plans
        - ERROR: Use base class general error handling
        """
        try:
            # Distribute processing according to message type
            if message.msg_type == MessageType.TASK_COMPLETED:
                return await self._handle_task_completed(message)

            elif message.msg_type == MessageType.TASK_INTERRUPTED:
                # Use base class general interrupt handling
                return await self._handle_task_interrupted(message)

            elif message.msg_type == MessageType.USER_INPUT:
                return await self._handle_user_input(message)

            elif message.msg_type == MessageType.ERROR:
                # Use base class general error handling
                return await self._handle_error(message)

            else:
                logger.warning(f"Unsupported message type: {message.msg_type}")
                return MessageHandlerResult(tasks=[], stop=False)

        except Exception as e:
            logger.error(f"Error in ReActMessageHandler: {e}")
            error_result = await self._send_error_stream(str(e))
            return MessageHandlerResult(
                tasks=[],
                stop=True,
                final_result=error_result
            )

    async def _handle_user_input(self, message: Message) -> MessageHandlerResult:
        """Handle user input messages - React core: LLM reasoning to generate plans
        
        Process:
        1. Add user message to conversation history
        2. Call LLM reasoning to generate task plans
        3. If it's a workflow task, check if it needs to resume interrupted task (using base class methods)
        4. Return task list
        """
        if not self.config.model:
            logger.warning("Model config missing, cannot generate plan")
            return MessageHandlerResult(tasks=[], stop=True)

        # Add user_message to conversation history
        MessageUtils.add_user_message(message.get_display_content(), self.context_engine, self.runtime)

        # Call LLM reasoning to generate plans
        tasks, llm_output = await self._generate_plan_from_llm(message)
        if not tasks:
            logger.info("No task is generated")
            final_result = await self._send_final_stream(llm_output.content)
            return MessageHandlerResult(tasks=[], stop=True, final_result=final_result)

        # Check if planned task is a workflow task
        workflow_task = self._resolve_workflow_from_tasks(tasks)
        if not workflow_task:
            # Plugin task - return directly
            logger.info("Created plugin task: %s", tasks)
            return MessageHandlerResult(tasks=tasks, stop=False)

        # Workflow task - check if it needs to resume interrupted task (using base class methods)
        interrupted_task = self._find_interrupted_task(workflow_task)
        if interrupted_task:
            # Workflow returned by reasoning is in interrupted state, resume it (using base class methods)
            logger.info(f"Resuming interrupted workflow: {workflow_task.name}")
            return await self._create_resume_task(message, workflow_task, interrupted_task)

        logger.info(f"Created new workflow task: {workflow_task.name}")
        return MessageHandlerResult(tasks=tasks, stop=False)

    async def _handle_task_completed(self, message: Message) -> MessageHandlerResult:
        """Handle task completion messages - React specific: continue LLM reasoning until task completion
        
        Difference from Workflow:
        - Workflow: Stop when task is completed
        - React: Call LLM reasoning again after task completion to determine if the problem is solved
        """
        # Write stream data for completed tasks
        await self._write_message_stream_data(message)

        # Add tool call results to history
        if message.content.stream_data[0].type in ("plugin_final", "workflow_final"):
            MessageHandlerUtils.add_tool_result(message, self.context_engine, self.runtime)

        # Clear workflow interruption state (if any) - use base class state management
        if message.context.workflow_id:
            self._state.clear_interrupted_task(message.context.workflow_id)
            self.save_state()
            logger.info(f"Cleared interrupt state for workflow: {message.context.workflow_id}")

        # Call LLM reasoning again to generate plans and determine if the problem is answered
        if self.iteration < self.config.constrain.max_iteration:
            tasks, llm_output = await self._generate_plan_from_llm(message)
            # If no new tasks, return stop signal, write stream data with final result
            if not tasks:
                logger.info("No new tasks generated, task completed, returning stop signal with final result")
                final_result = await self._send_final_stream(llm_output.content)
                return MessageHandlerResult(tasks=[], stop=True, final_result=final_result)
            return MessageHandlerResult(tasks=tasks, stop=False)

        # Exceed maximum iteration count, directly return stop signal with final result
        result = message.content.stream_data
        logger.info(f"Exceed max iteration {self.config.constrain.max_iteration}, "
                    "task completed, returning stop signal with final result")

        return MessageHandlerResult(
            tasks=[],
            stop=True,
            final_result=result
        )

    async def _generate_plan_from_llm(self, message: Message):
        """Call LLM to generate plans - React core method
        
        This is the only differentiated logic for ReAct mode:
        - Use LLM reasoning to generate task execution plans
        - Parse LLM output into task list
        - Increment iteration count by 1
        
        Returns:
            Tuple[List[Task], BaseMessage]: (tasks, llm_output)
        """
        logger.info(f"ReAct iteration {self.iteration + 1}")
        inputs = message.get_display_content()
        tools = self.runtime.get_tool_info()
        chat_history = MessageUtils.get_chat_history(self.context_engine, self.runtime, self.config)
        llm_inputs = MessageHandlerUtils.format_llm_inputs(inputs, chat_history, self.config)
        if UserConfig.is_sensitive():
            logger.info(f"React llm inputs")
        else:
            logger.info(f"React llm inputs: {llm_inputs}")

        try:
            model = self._get_model()
            llm_output = await model.ainvoke(
                self.config.model.model_info.model_name,
                llm_inputs,
                tools
            )

            tasks = MessageHandlerUtils.parse_llm_output(llm_output, self.config)
            # Add LLM output information to CE conversation history
            MessageUtils.add_ai_message(llm_output, self.context_engine, self.runtime)
            if UserConfig.is_sensitive():
                logger.info(f"React llm output")
            else:
                logger.info(f"React llm output: {llm_output}")
        except Exception as e:
            self.iteration += 1
            logger.error(f"Failed to invoke model, {e}")
            raise JiuWenBaseException(-1, "Failed to invoke model")

        self.iteration += 1
        return tasks, llm_output
