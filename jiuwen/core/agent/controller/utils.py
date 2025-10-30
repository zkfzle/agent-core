#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import copy
from typing import List, Dict, Any, Optional

from jiuwen.agent.common.enum import TaskType
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task import Task, TaskInput
from jiuwen.core.agent.controller.types import ControllerOutput, ControllerInput
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.utils.format.format_utils import FormatUtils
from jiuwen.core.utils.llm.messages import BaseMessage, ToolCall, AIMessage, HumanMessage, ToolMessage
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.common.logging import logger
from pydantic import Field, ConfigDict
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.utils.llm.hash_util import generate_key
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.component.common.configs.model_config import ModelConfig


class WorkflowControllerOutput(ControllerOutput):
    tasks: List[Task] = Field(default_factory=list)
    messages: Any = Field(default_factory=list)


class WorkflowControllerInput(ControllerInput):
    model_config = ConfigDict(extra='allow')


class ReActControllerInput(ControllerInput):
    model_config = ConfigDict(extra='allow')


class ReActControllerOutput(ControllerOutput):
    should_continue: bool = Field(default=False)
    llm_output: Optional[AIMessage] = Field(default=None)
    tasks: List[Task] = Field(default_factory=list)


class ReActControllerUtils:

    @staticmethod
    def format_llm_inputs(
            inputs,
            chat_history: List[BaseMessage],
            config: AgentConfig
    ) -> List[BaseMessage]:
        if isinstance(inputs.query, InteractiveInput):
            user_fields = copy.deepcopy(inputs.model_dump())
            user_fields.pop("query")
        else:
            user_fields = inputs.model_dump()

        system_prompt = (Template(
            name=config.prompt_template_name,
            content=config.prompt_template
        ).format(user_fields).to_messages())

        return FormatUtils.create_llm_inputs(system_prompt, chat_history)

    @staticmethod
    def parse_llm_output(response: BaseMessage, config: AgentConfig) -> "ReActControllerOutput":
        tasks = ReActControllerUtils.create_tasks_from_tool_calls(
            response.tool_calls, config
        )
        should_continue = len(tasks) > 0
        return ReActControllerOutput(
            should_continue=should_continue,
            llm_output=response,
            tasks=tasks
        )

    @staticmethod
    def create_tasks_from_tool_calls(
            tool_calls: List[ToolCall],
            config: AgentConfig
    ) -> List[Task]:
        if not tool_calls:
            return []

        result = []
        for tool_call in tool_calls:
            task_type = ReActControllerUtils.determine_task_type(
                tool_call.function.name, config
            )
            result.append(Task(
                id=tool_call.id,
                input=TaskInput(
                    target_name=tool_call.function.name,
                    arguments=FormatUtils.json_loads(tool_call.function.arguments)
                ),
                task_type=task_type
            ))
        return result

    @staticmethod
    def determine_task_type(tool_name: str, config: AgentConfig) -> TaskType:
        for workflow in config.workflows:
            if tool_name == workflow.name:
                return TaskType.WORKFLOW

        for plugin in config.plugins:
            if tool_name == plugin.name:
                return TaskType.PLUGIN

        raise JiuWenBaseException(5000, f"not find tool call type: {tool_name}")

    @staticmethod
    def is_interaction_result(exec_result: Any) -> bool:
        return (isinstance(exec_result, dict) and
                exec_result.get("error") and
                isinstance(exec_result.get("value"), list))

    @staticmethod
    def create_interrupt_result(e, tool_name: str) -> Dict[str, Any]:
        return {
            "error": True,
            "value": e.message,
            "tool_name": tool_name
        }

    @staticmethod
    def validate_execution_inputs(exec_result: Any, sub_task_result: Any) -> bool:
        return exec_result is not None

    @staticmethod
    def should_add_user_message(query: str, context_engine: ContextEngine, runtime: Runtime) -> bool:
        agent_context = context_engine.get_agent_context(runtime.session_id())
        last_message = agent_context.get_latest_message()

        if not last_message:
            return True

        if last_message.role == 'tool':
            logger.info("Skipping user message - post-tool-call request")
            return False

        if last_message.role == 'user' and last_message.content == query:
            logger.info("Skipping duplicate user message")
            return False

        return True

    @staticmethod
    def add_user_message(query: Any, context_engine: ContextEngine, runtime: Runtime):
        if ReActControllerUtils.should_add_user_message(query, context_engine, runtime):
            agent_context = context_engine.get_agent_context(runtime.session_id())
            user_message = HumanMessage(content=query)
            agent_context.add_message(user_message)
            if UserConfig.is_sensitive():
                logger.info(f"Added user message")
            else:
                logger.info(f"Added user message: {query}")

    @staticmethod
    def add_ai_message(ai_message: AIMessage, context_engine: ContextEngine, runtime: Runtime):
        if ai_message:
            agent_context = context_engine.get_agent_context(runtime.session_id())
            agent_context.add_message(ai_message)

    @staticmethod
    def add_tool_results(completed_tasks: List[Task], context_engine: ContextEngine, runtime: Runtime):
        if not completed_tasks:
            logger.warning("No completed sub tasks to add to chat history")
            return

        agent_context = context_engine.get_agent_context(runtime.session_id())
        logger.info(f"Adding {len(completed_tasks)} tool results to chat history")

        for task in completed_tasks:
            if task.result:
                tool_message = ToolMessage(content=task.result, tool_call_id=task.id)
                agent_context.add_message(tool_message)
                if UserConfig.is_sensitive():
                    logger.info(f"Added tool result")
                else:
                    logger.info(f"Added tool result: {task.input.target_name}")
            else:
                if UserConfig.is_sensitive():
                    logger.warning(f"Task has no result")
                else:
                    logger.warning(f"Task {task.input.target_name} has no result")

    @staticmethod
    def get_chat_history(context_engine: ContextEngine, runtime: Runtime, config: AgentConfig) -> List[BaseMessage]:
        agent_context = context_engine.get_agent_context(runtime.session_id())
        chat_history = agent_context.get_messages()
        max_rounds = config.constrain.reserved_max_chat_rounds
        return chat_history[-2 * max_rounds:]

class ReasonerUtils:
    @staticmethod
    def get_chat_history(context_engine: ContextEngine, runtime: Runtime, chat_history_max_turn: int) -> List[BaseMessage]:
        """根据最大对话轮数获取历史记录"""
        agent_context = context_engine.get_agent_context(runtime.session_id())
        chat_history = agent_context.get_messages()
        return chat_history[-2 * chat_history_max_turn:]
    
    @staticmethod
    def get_model(model_config: ModelConfig, runtime: Runtime):
        """根据模型配置获取模型实例"""
        model_id = generate_key(
            model_config.model_info.api_key,
            model_config.model_info.api_base,
            model_config.model_provider
        )

        model = runtime.get_model(model_id=model_id)

        if model is None:
            model = ModelFactory().get_model(
                model_provider=model_config.model_provider,
                api_base=model_config.model_info.api_base,
                api_key=model_config.model_info.api_key
            )
            runtime.add_model(model_id=model_id, model=model)

        return model
