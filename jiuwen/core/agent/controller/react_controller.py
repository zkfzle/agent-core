#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""Controller of ReActAgent"""
import ast
import json
from typing import List, Dict, Any, Optional

from pydantic import Field

from jiuwen.agent.common.enum import SubTaskType
from jiuwen.agent.common.schema import WorkflowSchema, PluginSchema
from jiuwen.core.agent.controller.base import ControllerOutput, ControllerInput, Controller
from jiuwen.core.agent.handler.base import AgentHandler
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.agent.task.task_context import TaskContext
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.context.controller_context.controller_context_manager import ControllerContextMgr
from jiuwen.core.utils.llm.messages import BaseMessage, ToolInfo, Function, Parameters, HumanMessage, AIMessage, \
    ToolCall
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.utils.output_parser.base import BaseOutputParser
from jiuwen.core.utils.output_parser.null_output_parser import NullOutputParser
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.prompt.template.template_manager import TemplateManager


DIALOGUE_HISTORY_KEY = "mock_dialogue_history"


class ReActControllerOutput(ControllerOutput):
    should_continue: bool = Field(default=False)
    llm_output: Optional[AIMessage] = Field(default=None)
    sub_tasks: List[SubTask] = Field(default_factory=list)


class ReActControllerInput(ControllerInput):
    user_fields: Dict[str, Any] = Field(default_factory=dict, alias="userFields")


class ReActControllerUtils:
    @classmethod
    def format_input_parameters(cls, inputs: dict) -> Parameters:
        properties, required_parameters = dict(), list()
        if inputs.get("properties"):
            for key, value in inputs.get("properties").items():
                if value.get("required", False):
                    required_parameters.append(key)
                param_type = value.get("type", "").lower()
                if param_type in ["array", "object"]:
                    nested_result = dict()
                    cls._recursive_format_nested_params(param_type, value.get("properties", dict()), nested_result)
                    properties[key] = nested_result
                else:
                    properties[key] = dict(description=value.get("description", ""), type=param_type)
        parameters = Parameters(properties=properties, required=required_parameters)
        return parameters

    @classmethod
    def _recursive_format_nested_params(cls, param_type, properties, output):
        pass

    @staticmethod
    def get_dialogue_history_from_context(context: TaskContext):
        if hasattr(context, "context_manager"):
            chat_history = context.context_manager.message_mgr.get_chat_history()
        else:
            # TODO: 临时获取对话历史
            chat_history = context.state().get(DIALOGUE_HISTORY_KEY) or list()
        return chat_history

    @staticmethod
    def set_dialogue_history_to_context(current_messages, context):
        if hasattr(context, "context_manager"):
            context.context_manager.message_mgr.set_chat_history(current_messages)
        else:
            # TODO: 临时存储对话历史
            context.state().update({DIALOGUE_HISTORY_KEY: current_messages})

    @staticmethod
    def json_loads(arguments: str):
        result = dict()
        try:
            result = json.loads(arguments, strict=False)
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(arguments)
            except (SyntaxError, AttributeError, ValueError):
                pass
        return result


class ReActController(Controller):
    def __init__(self, config: AgentConfig, context_mgr: ControllerContextMgr):
        super().__init__(config, context_mgr)
        self._model = self._init_model()
        self._output_parser = self._init_output_parser()

    def invoke(self, inputs: ReActControllerInput, context: TaskContext) -> ReActControllerOutput:
        query = inputs.query
        user_fields = inputs.user_fields
        chat_history = self._get_latest_chat_history(context)
        tools = self._format_tools_info()
        system_prompt = self._format_system_prompt_template(user_fields, context)
        llm_inputs = self._create_llm_inputs(query, system_prompt, chat_history)
        result = self._invoke_llm_and_parse_output(llm_inputs, tools)
        self._update_llm_response_to_context(result.llm_output, chat_history, context)
        return result

    def set_agent_handler(self, agent_handler: AgentHandler):
        self._agent_handler = agent_handler

    def _get_latest_chat_history(self, context: TaskContext) -> List[BaseMessage]:
        chat_history = ReActControllerUtils.get_dialogue_history_from_context(context)
        chat_history = chat_history[-2 * self._config.constrain.reserved_max_chat_rounds:]
        return chat_history

    def _format_tools_info(self) -> List[ToolInfo]:
        tool_info_list: List[ToolInfo] = list()
        workflows_metadata = self._config.workflows
        plugins_metadata = self._config.plugins
        tool_info_list.extend(self._format_workflows_metadata(workflows_metadata))
        tool_info_list.extend(self._format_plugins_metadata(plugins_metadata))
        return tool_info_list

    @staticmethod
    def _format_workflows_metadata(workflows_metadata: List[WorkflowSchema]) -> List[ToolInfo]:
        result = []
        for workflow in workflows_metadata:
            parameters = ReActControllerUtils.format_input_parameters(workflow.inputs)
            function = Function(name=workflow.name, description=workflow.description, parameters=parameters)
            result.append(ToolInfo(function=function))
        return result

    @staticmethod
    def _format_plugins_metadata(plugins_metadata: List[PluginSchema]) -> List[ToolInfo]:
        result = []
        for plugin in plugins_metadata:
            parameters = ReActControllerUtils.format_input_parameters(plugin.inputs)
            function = Function(name=plugin.name, description=plugin.description, parameters=parameters)
            result.append(ToolInfo(function=function))
        return result

    @staticmethod
    def _create_llm_inputs(query: str, system_prompt: List[BaseMessage], chat_history: List[BaseMessage]) -> List[BaseMessage]:
        if not chat_history or chat_history[0].role != "system":
            chat_history[:0] = system_prompt
        if chat_history and chat_history[-1].role in ["system", "assistant"]:
            chat_history.append(HumanMessage(content=query))
        return chat_history

    def _invoke_llm_and_parse_output(self, llm_inputs: List[BaseMessage], tools: List[ToolInfo]) -> ReActControllerOutput:
        try:
            response = self._model.invoke(llm_inputs, tools)
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.INVOKE_LLM_FAILED.code,
                message=StatusCode.INVOKE_LLM_FAILED.errmsg
            ) from e

        if isinstance(self._output_parser, NullOutputParser):
            llm_output = response
        else:
            llm_output = self._output_parser.parse(response)
        sub_tasks = self._format_sub_tasks(llm_output.tool_calls)
        should_continue = isinstance(sub_tasks, list) and len(sub_tasks) > 0
        return ReActControllerOutput(should_continue=should_continue, llm_output=llm_output, sub_tasks=sub_tasks)

    def _format_system_prompt_template(self, user_fields, context):
        if not self._config.prompt_template and hasattr(context, "context_manager"):
                template_manager: TemplateManager = context.context_manager.template_mgr
                if template_manager:
                    try:
                        template = template_manager.get(self._config.prompt_template_name)
                    except JiuWenBaseException:
                        template = Template(name=self._config.prompt_template_name, content=self._config.prompt_template)
                        template_manager.register(template)
                    return template.format(user_fields).to_messages()

        return (Template(name=self._config.prompt_template_name, content=self._config.prompt_template)
                .format(user_fields)
                .to_messages())

    def _init_model(self):
        return ModelFactory().get_model(model_provider=self._config.model.model_provider,
                                        model_info=self._config.model.model_info)

    @staticmethod
    def _update_llm_response_to_context(llm_output: AIMessage, chat_history: List[BaseMessage], context: TaskContext):
        if llm_output:
            chat_history.append(llm_output)
        ReActControllerUtils.set_dialogue_history_to_context(chat_history, context)

    def _format_sub_tasks(self, tool_calls: List[ToolCall]) -> List[SubTask]:
        if not tool_calls:
            return []
        result = []
        for tool_call in tool_calls:
            tool_call_id = tool_call.id
            tool_call_info = tool_call.function
            tool_call_name = tool_call_info.name
            tool_call_args = ReActControllerUtils.json_loads(tool_call_info.arguments)
            tool_call_type = self._check_sub_task_type(tool_call_name)
            result.append(
                SubTask(id=tool_call_id, func_name=tool_call_name, func_args=tool_call_args, sub_task_type=tool_call_type))
        return result

    def _check_sub_task_type(self, tool_call_name: str) -> SubTaskType:
        result = SubTaskType.UNDEFINED
        workflows_metadata = self._config.workflows
        for workflow in workflows_metadata:
            if tool_call_name == workflow.name:
                result = SubTaskType.WORKFLOW
                break
        if result == SubTaskType.UNDEFINED:
            plugins_metadata = self._config.plugins
            for plugin in plugins_metadata:
                if tool_call_name == plugin.name:
                    result = SubTaskType.PLUGIN
                    break
        if result == SubTaskType.UNDEFINED:
            raise JiuWenBaseException()
        return result

    def _init_output_parser(self) -> BaseOutputParser:
        model_provider = self._model.model_provider()
        if model_provider in ["siliconflow"]:
            result = BaseOutputParser.from_config(model_provider)
        else:
            result = BaseOutputParser.from_config("novel_tool")
        return result
