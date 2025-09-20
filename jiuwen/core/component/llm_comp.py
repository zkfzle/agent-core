#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import json
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, AsyncIterator

from jiuwen.core.common.constants.constant import QUERY
from jiuwen.core.common.enum.enum import WorkflowLLMResponseType, MessageRole
from jiuwen.core.common.exception.exception import JiuWenBaseException, InterruptException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.common.utils.utils import WorkflowLLMUtils, OutputFormatter, ValidationUtils, SchemaGenerator
from jiuwen.core.component.base import ComponentConfig, WorkflowComponent
from jiuwen.core.context_engine.base import Context
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.stream.writer import OutputSchema
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import AIMessage
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.prompt.template.template_manager import TemplateManager

WORKFLOW_CHAT_HISTORY = "workflow_chat_history"
CHAT_HISTORY_MAX_TURN = 3
_ROLE = "role"
_CONTENT = "content"
ROLE_MAP = {"user": "用户", "assistant": "助手", "system": "系统"}
_SPAN = "span"
_WORKFLOW_DATA = "workflow_data"
_ID = "id"
_TYPE = "type"
_INSTRUCTION_NAME = "instruction_name"
_TEMPLATE_NAME = "template_name"

RESPONSE_FORMAT_TO_PROMPT_MAP = {
    WorkflowLLMResponseType.JSON.value: {
        _INSTRUCTION_NAME: "jsonInstruction",
        _TEMPLATE_NAME: "llm_json_formatting"
    },
    WorkflowLLMResponseType.MARKDOWN.value: {
        _INSTRUCTION_NAME: "markdownInstruction",
        _TEMPLATE_NAME: "llm_markdown_formatting"
    }
}


class LLMPromptFormatter:
    """格式化对话历史中的最后一条用户消息，追加输出格式指令。"""

    # 常量模板，避免在函数内重复创建
    _DEFAULT_MARKDOWN_INSTRUCTION = (
        "Please return the answer in markdown format.\n"
        "- For headings, use number signs (#).\n"
        "- For list items, start with dashes (-).\n"
        "- To emphasize text, wrap it with asterisks (*).\n"
        "- For code or commands, surround them with backticks (`).\n"
        "- For quoted text, use greater than signs (>).\n"
        "- For links, wrap the text in square brackets [], followed by the URL in parentheses ().\n"
        "- For images, use square brackets [] for the alt text, followed by the image URL in parentheses ().\n"
        "The question is: ${query}."
    )

    _DEFAULT_JSON_INSTRUCTION = (
        "Carefully consider the user's question to ensure your answer is logical and makes sense.\n"
        "- Make sure your explanation is concise and easy to understand, not verbose.\n"
        "- Strictly return the answer in valid JSON format only, and "
        "\"DO NOT ADD ANY COMMENTS BEFORE OR AFTER IT\" to ensure it could be formatted "
        "as a JSON instance that conforms to the JSON schema below.\n"
        "Here is the JSON schema: ${json_schema}.\n"
        "The question is: ${query}."
    )

    @staticmethod
    def _find_last_user_index(history: List[Dict[str, Any]]) -> int | None:
        """返回最后一条 role=user 的索引；不存在则返回 None。"""
        for idx in range(len(history) - 1, -1, -1):
            if history[idx].get("role") == "user":
                return idx
        return None

    @staticmethod
    def format_prompt(
            history: List[Dict[str, Any]],
            response_format: Dict[str, Any],
            output_config: dict,
    ) -> List[Dict[str, Any]]:
        """根据 response_format 格式化最后一条用户消息。"""
        res_type = response_format.get("type")
        if res_type == "text":
            return history

        last_user_idx = LLMPromptFormatter._find_last_user_index(history)
        if last_user_idx is None:
            return history
        query = history[last_user_idx]["content"]

        if res_type == "markdown":
            instruction = (
                    response_format.get("markdownInstruction")
                    or LLMPromptFormatter._DEFAULT_MARKDOWN_INSTRUCTION
            )
            prompt = instruction.replace("${query}", query)

        elif res_type == "json":
            json_schema = SchemaGenerator.generate_json_schema(output_config)
            instruction = (
                    response_format.get("jsonInstruction")
                    or LLMPromptFormatter._DEFAULT_JSON_INSTRUCTION
            )
            prompt = (
                instruction
                .replace("${json_schema}", json.dumps(json_schema, ensure_ascii=False))
                .replace("${query}", query)
            )

        else:
            ValidationUtils.raise_invalid_params_error(f"'{res_type}' is not supported")

        history[last_user_idx]["content"] = prompt
        return history


@dataclass
class LLMCompConfig(ComponentConfig):
    model: 'ModelConfig' = None
    deployMode: str = ""
    template_content: List[Any] = field(default_factory=list)
    response_format: Dict[str, Any] = field(default_factory=dict)
    enable_history: bool = True
    user_fields: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)


class LLMExecutable(ComponentExecutable):
    def __init__(self, component_config: LLMCompConfig):
        super().__init__()
        self._config = component_config
        self._llm: BaseChatModel = None
        self._initialized: bool = False

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        try:
            self._set_runtime(runtime)
            model_inputs = self._prepare_model_inputs(inputs)
            logger.info("[%s] model inputs %s", self._runtime.executable_id(), model_inputs)
            llm_response = await self._llm.ainvoke(
                model_name=self._config.model.model_info.model_name, messages=model_inputs)
            response = llm_response.content

            # 临时调试：用于调用streamWriter实现流式输出
            await runtime.write_custom_stream({"streamOutput": response})

            self._runtime.update_global_state({"response": response})
            logger.info("[%s] model outputs %s", self._runtime.executable_id(), response)
            return self._create_output(response)
        except JiuWenBaseException:
            raise
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise JiuWenBaseException(error_code=StatusCode.WORKFLOW_LLM_INIT_ERROR.code,
                                      message=StatusCode.WORKFLOW_LLM_INIT_ERROR.errmsg.format(msg=str(e))) from e

    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        try:
            self._set_runtime(runtime)
            response_format_type = self._get_response_format().get(_TYPE)

            if response_format_type == WorkflowLLMResponseType.JSON.value:
                async for out in self._invoke_for_json_format(inputs):
                    yield out
            else:
                async for out in self._stream_with_chunks(inputs):
                    yield out
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.WORKFLOW_LLM_STREAMING_OUTPUT_ERROR.code,
                message=StatusCode.WORKFLOW_LLM_STREAMING_OUTPUT_ERROR.errmsg.format(msg=str(e))
            ) from e

    async def interrupt(self, message: dict):
        raise InterruptException(
            error_code=StatusCode.CONTROLLER_INTERRUPTED_ERROR.code,
            message=json.dumps(message, ensure_ascii=False)
        )

    def _initialize_if_needed(self):
        if not self._initialized:
            try:
                self._llm = self._create_llm_instance()
                self._initialized = True
            except Exception as e:
                raise JiuWenBaseException(
                    error_code=StatusCode.WORKFLOW_LLM_INIT_ERROR.code,
                    message=StatusCode.WORKFLOW_LLM_INIT_ERROR.errmsg.format(msg=str(e))
                ) from e

    def _create_llm_instance(self):
        return ModelFactory().get_model(model_provider=self._config.model.model_provider,
                                        api_base=self._config.model.model_info.api_base,
                                        api_key=self._config.model.model_info.api_key)

    def _validate_inputs(self, inputs: Input) -> None:
        if not inputs or not inputs.get(QUERY):
            raise JiuWenBaseException(
                error_code=StatusCode.WORKFLOW_LLM_TEMPLATE_ASSEMBLE_ERROR.code,
                message=StatusCode.WORKFLOW_LLM_TEMPLATE_ASSEMBLE_ERROR.errmsg
            )

    def _process_inputs(self, inputs: dict) -> dict:
        processed_inputs = {}
        if inputs:
            processed_inputs = inputs.copy()
            if self._runtime:
                chat_history: list = self._runtime.get_global_state(WORKFLOW_CHAT_HISTORY)
                chat_history = chat_history[:-1] if chat_history else []
                full_input = ""
                for history in chat_history[-CHAT_HISTORY_MAX_TURN:]:
                    full_input += "{}：{}\n".format(ROLE_MAP.get(history.get("role", "user"), "用户"),
                                                   history.get("content"))
                inputs.update({"CHAT_HISTORY": full_input})
        return processed_inputs

    def _build_prompt_message(self, inputs: dict) -> Template:
        template_content_list = self._config.template_content
        user_prompt = [element for element in template_content_list if element.get(_ROLE, "") == MessageRole.USER.value]
        if not user_prompt or not isinstance(user_prompt[0], dict):
            raise JiuWenBaseException(
                error_code=StatusCode.WORKFLOW_LLM_INIT_ERROR.code,
                message=StatusCode.WORKFLOW_LLM_INIT_ERROR.errmsg.format(msg="Failed to retrieve llm template content")
            )
        default_template = Template(name="default", content=str(user_prompt[0].get("content")))
        return default_template.format(inputs).content

    def _get_model_input(self, inputs: dict):
        system_prompt = self._build_system_prompt(inputs)
        user_prompt = self._build_prompt_message(inputs)
        history = self._get_history(system_prompt, user_prompt)
        return LLMPromptFormatter.format_prompt(history=history,
                                                response_format=self._config.response_format,
                                                output_config=self._config.output_config)

    def _get_history(self, system_prompt: list, user_prompt: str):
        original_history = system_prompt if isinstance(system_prompt, list) else []
        if self._runtime:
            chat_history: list = self._runtime.get_global_state(WORKFLOW_CHAT_HISTORY)
            if chat_history and self._config.enable_history:
                original_history = chat_history
        original_history.append({"role": "user", "content": user_prompt})
        return original_history

    def _get_response_format(self):
        try:
            response_format = self._config.response_format
            if not response_format:
                return {}

            format_type = response_format.get(_TYPE)
            if not format_type or format_type not in RESPONSE_FORMAT_TO_PROMPT_MAP:
                return response_format

            format_config = RESPONSE_FORMAT_TO_PROMPT_MAP[format_type]
            instruction_name = format_config.get(_INSTRUCTION_NAME)

            if response_format.get(instruction_name):
                return response_format

            instruction_content = self._get_instruction_from_template(format_config)
            if instruction_content:
                response_format[instruction_name] = instruction_content

            return response_format

        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.WORKFLOW_LLM_TEMPLATE_ASSEMBLE_ERROR.code,
                message=StatusCode.WORKFLOW_LLM_TEMPLATE_ASSEMBLE_ERROR.errmsg
            ) from e

    def _get_instruction_from_template(self, format_config: dict) -> Optional[str]:
        template_name = format_config.get(_TEMPLATE_NAME)
        try:
            if not template_name:
                return None
            filters = self._build_template_filters()

            template_manager = TemplateManager()
            template = template_manager.get(name=template_name, filters=filters)

            return getattr(template, "content", None) if template else None
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.WORKFLOW_LLM_TEMPLATE_ASSEMBLE_ERROR.code,
                message=StatusCode.WORKFLOW_LLM_TEMPLATE_ASSEMBLE_ERROR.errmsg
            )(e)

    def _build_template_filters(self) -> dict:
        filters = {}

        model_name = self._config.model.model_info.model_name
        if model_name:
            filters["model_name"] = model_name

        return filters

    def _create_output(self, llm_output) -> Output:
        formatted_res = OutputFormatter.format_response(llm_output,
                                                        self._config.response_format,
                                                        self._config.output_config)
        return formatted_res

    def _set_runtime(self, runtime: Runtime):
        self._runtime = runtime

    def _prepare_model_inputs(self, inputs):
        self._initialize_if_needed()

        processed_inputs = self._process_inputs(inputs)
        return self._get_model_input(processed_inputs)

    async def _invoke_for_json_format(self, inputs: Input) -> AsyncIterator[Output]:
        model_inputs = self._prepare_model_inputs(inputs)
        logger.info("[%s] model inputs %s", self._runtime.executable_id(), model_inputs)
        llm_output = await self._llm.ainvoke(model_name=self._config.model.model_info.model_name, messages=model_inputs)  # 如果 invoke 是异步接口，要加 await
        yield self._create_output(llm_output)

    async def _stream_with_chunks(self, inputs: Input) -> AsyncIterator[Output]:
        model_inputs = self._prepare_model_inputs(inputs)
        # 假设 self._llm.stream 本身就是异步生成器
        async for chunk in self._llm.astream(model_name=self._config.model.model_info.model_name, messages=model_inputs):
            content = WorkflowLLMUtils.extract_content(chunk)
            formatted_res = OutputFormatter.format_response(content,
                                                            self._config.response_format,
                                                            self._config.output_config)
            stream_out = formatted_res
            yield stream_out

    def _format_response_content(self, response_content: str) -> dict:
        pass

    async def _stream_llm_with_stream_writer(self, model_inputs, stream_writer) -> AIMessage:
        final_response = AIMessage()
        response_format_type = self._get_response_format().get(_TYPE)

        if response_format_type == WorkflowLLMResponseType.JSON.value:
            final_response = await self._llm.ainvoke(
                model_name=self._config.model.model_info.model_name, messages=model_inputs)
        else:
            index = 0
            result = ""
            async for chunk in self._llm.astream(model_name=self._config.model.model_info.model_name,
                                                 messages=model_inputs):
                if chunk.content:
                    if stream_writer:
                        await stream_writer.write(OutputSchema(type="workflow", index=index, payload=chunk.content))
                    result += chunk.content
                index += 1
            final_response.content = result

        return final_response

    def _build_system_prompt(self, inputs: dict):
        system_prompt = []
        for element in self._config.template_content:
            if element.get(_ROLE, "") == "system":
                system_prompt.append(element)
            else:
                break
        system_prompt_template = Template(name="default_system_prompt", content=system_prompt)
        return system_prompt_template.format(inputs).content


class LLMComponent(WorkflowComponent):
    def __init__(self, component_config: Optional[LLMCompConfig] = None):
        super().__init__()
        self._executable = None
        self._config = component_config

    @property
    def executable(self) -> LLMExecutable:
        if self._executable is None:
            self._executable = self.to_executable()
        return self._executable

    def to_executable(self) -> LLMExecutable:
        return LLMExecutable(self._config)
