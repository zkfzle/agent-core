#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import ast
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, List, Dict, Iterator, AsyncIterator, Union

from pydantic import BaseModel, Field, ConfigDict

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.component.base import ComponentConfig, WorkflowComponent
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.context_engine.base import Context
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.graph.executable import Executable, Input, Output
from jiuwen.core.graph.interrupt.interaction import Interaction
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.prompt.template.template_manager import TemplateManager

START_STR = "start"
END_STR = "end"
USER_INTERACT_STR = "user_interact"

SUB_PLACEHOLDER_PATTERN = r'\{\{([^}]*)\}\}'
CONTINUE_ASK_STATEMENT = "请您提供{non_extracted_key_fields_names}相关的信息"
WORKFLOW_CHAT_HISTORY = "workflow_chat_history"
TEMPLATE_NAME = "questioner"
QUESTIONER_STATE_KEY = "questioner_state"

QUESTIONER_SYSTEM_TEMPLATE = """\
你是一个信息收集助手，你需要根据指定的参数收集用户的信息，然后提交到系统。
请注意：不要使用任何工具、不用理会问题的具体含义，并保证你的输出仅有 JSON 格式的结果数据。
请严格遵循如下规则：
  1. 让我们一步一步思考。
  2. 用户输入中没有提及的参数提取为 None，并直接向询问用户没有明确提供的参数。
  3. 通过用户提供的对话历史以及当前输入中提取 {{required_name}}，不要追问任何其他信息。
  4. 参数收集完成后，将收集到的信息通过 JSON 的方式展示给用户。

## 指定参数
{{required_params_list}}

## 约束
{{extra_info}}

## 示例
{{example}}
"""

QUESTIONER_USER_TEMPLATE = """\
对话历史
{{dialogue_history}}

请充分考虑以上对话历史及用户输入，正确提取最符合约束要求的 JSON 格式参数。
"""

def questioner_default_template():
    return [
        {"role": "system", "content": QUESTIONER_SYSTEM_TEMPLATE},
        {"role": "user", "content": QUESTIONER_USER_TEMPLATE},
    ]

class ExecutionStatus(Enum):
    START = START_STR
    USER_INTERACT = USER_INTERACT_STR
    END = END_STR

class QuestionerEvent(Enum):
    START_EVENT = START_STR
    END_EVENT = END_STR
    USER_INTERACT_EVENT = USER_INTERACT_STR

class ResponseType(Enum):
    ReplyDirectly = "reply_directly"

class FieldInfo(BaseModel):
    field_name: str = Field(default="")
    description: str = Field(default="")
    cn_field_name: str = Field(default="")
    required: bool = Field(default=False)
    default_value: Any = Field(default="")

@dataclass
class QuestionerConfig(ComponentConfig):
    model: Optional[ModelConfig] = field(default=None)
    response_type: str = field(default=ResponseType.ReplyDirectly.value)
    question_content: str = field(default="")
    extract_fields_from_response: bool = field(default=True)
    field_names: List[FieldInfo] = field(default_factory=list)
    max_response: int = field(default=3)
    with_chat_history: bool = field(default=True)
    chat_history_max_rounds: int = field(default=5)
    prompt_template: List[Dict] = field(default_factory=questioner_default_template)
    extra_prompt_for_fields_extraction: str = field(default="")
    example_content: str = field(default="")


class QuestionerInput(BaseModel):
    model_config = ConfigDict(extra='allow')   # 允许任意额外字段
    query: Union[str, None] = Field(default="")


class OutputCache(BaseModel):
    user_response: str = Field(default="")
    question: str = Field(default="")
    key_fields: dict = Field(default_factory=dict)


class QuestionerOutput(BaseModel):
    user_response: str = Field(default="")
    question: str = Field(default="")
    model_config = ConfigDict(extra='allow')  # 允许任意额外字段


class QuestionerState(BaseModel):
    response_num: int = Field(default=0)
    user_response: str = Field(default="")
    extracted_key_fields: Dict[str, Any] = Field(default_factory=dict)
    status: ExecutionStatus = Field(default=ExecutionStatus.START)

    @classmethod
    def deserialize(cls, raw_state: dict):
        state = cls.model_validate(raw_state)
        return state.handle_event(QuestionerEvent(state.status.value))

    def serialize(self) -> dict:
        return self.model_dump()

    def handle_event(self, event: QuestionerEvent):
        if event == QuestionerEvent.START_EVENT:
            return QuestionerStartState.from_state(self)
        if event == QuestionerEvent.USER_INTERACT_EVENT:
            return QuestionerInteractState.from_state(self)
        if event == QuestionerEvent.END_EVENT:
            return QuestionerEndState.from_state(self)
        return self

    def is_undergoing_interaction(self):
        return self.status in [ExecutionStatus.USER_INTERACT]


class QuestionerStartState(QuestionerState):
    @classmethod
    def from_state(cls, questioner_state: QuestionerState):
        return cls(response_num=questioner_state.response_num,
                   user_response=questioner_state.user_response,
                   extracted_key_fields=questioner_state.extracted_key_fields,
                   status=ExecutionStatus.START)

    def handle_event(self, event: QuestionerEvent):
        if event == QuestionerEvent.USER_INTERACT_EVENT:
            return QuestionerInteractState.from_state(self)
        if event == QuestionerEvent.END_EVENT:
            return QuestionerEndState.from_state(self)
        return self

class QuestionerInteractState(QuestionerState):
    status: ExecutionStatus = Field(default=ExecutionStatus.USER_INTERACT)

    @classmethod
    def from_state(cls, questioner_state: QuestionerState):
        return cls(response_num=questioner_state.response_num,
                   user_response=questioner_state.user_response,
                   extracted_key_fields=questioner_state.extracted_key_fields,
                   status=ExecutionStatus.USER_INTERACT)

    def handle_event(self, event: QuestionerEvent):
        if event == QuestionerEvent.END_EVENT:
            return QuestionerEndState.from_state(self)
        return self


class QuestionerEndState(QuestionerState):
    status: ExecutionStatus = Field(default=ExecutionStatus.END)

    @classmethod
    def from_state(cls, questioner_state: QuestionerState):
        return cls(response_num=questioner_state.response_num,
                   user_response=questioner_state.user_response,
                   extracted_key_fields=questioner_state.extracted_key_fields,
                   status=ExecutionStatus.END)

    def handle_event(self, event: QuestionerEvent):
        return self


class QuestionerUtils:
    @staticmethod
    def format_template(template: str, user_fields: dict):
        def replace(match):
            key = match.group(1)
            return str(user_fields.get(key))

        try:
            result = re.sub(SUB_PLACEHOLDER_PATTERN, replace, template)
            return result
        except (KeyError, TypeError, AttributeError):
            return ""

    @staticmethod
    def get_latest_k_rounds_chat(chat_history, rounds):
        return chat_history[-rounds * 2 - 1:]

    @staticmethod
    def format_continue_ask_question(non_extracted_key_fields: List[FieldInfo]):
        non_extracted_key_fields_names = list()
        for param in non_extracted_key_fields:
            non_extracted_key_fields_names.append(param.cn_field_name or param.description)
        result = ", ".join(non_extracted_key_fields_names)
        return CONTINUE_ASK_STATEMENT.format(non_extracted_key_fields_names=result)


class QuestionerDirectReplyHandler:
    def __init__(self):
        self._config = None
        self._model = None
        self._state = None
        self._prompt = None
        self._query = ""

    def config(self, config: QuestionerConfig):
        self._config = config
        return self

    def model(self, model: BaseChatModel):
        self._model = model
        return self

    def state(self, state: QuestionerState):
        self._state = state
        return self

    def get_state(self):
        return self._state

    def prompt(self, prompt):
        self._prompt = prompt
        return self

    async def handle(self, inputs: Input, runtime: Runtime):
        if self._state.status == ExecutionStatus.START:
            return self._handle_start_state(inputs, runtime)
        if self._state.status == ExecutionStatus.USER_INTERACT:
            return await self._handle_user_interact_state(inputs, runtime)
        if self._state.status == ExecutionStatus.END:
            return self._handle_end_state(inputs, runtime)
        return dict()

    def _handle_start_state(self, inputs, runtime):
        questioner_input = QuestionerInput.model_validate(inputs)
        output = OutputCache()
        self._query = questioner_input.query or ""
        chat_history = self._get_latest_chat_history(runtime)
        if self._is_set_question_content():
            user_fields = questioner_input.model_dump(exclude={'query'})
            output.question = QuestionerUtils.format_template(self._config.question_content, user_fields)
            self._state = self._state.handle_event(QuestionerEvent.USER_INTERACT_EVENT)
            return self._format_questioner_output(output)

        if self._need_extract_fields():
            is_continue_ask = self._initial_extract_from_chat_history(chat_history, output)
            event = QuestionerEvent.USER_INTERACT_EVENT if is_continue_ask else QuestionerEvent.END_EVENT
            self._state = self._state.handle_event(event)
        else:
            raise JiuWenBaseException(
                error_code=StatusCode.WORKFLOW_QUESTIONER_QUESTION_EMPTY_DIRECT_COLLECTION_ERROR.code,
                message=StatusCode.WORKFLOW_QUESTIONER_QUESTION_EMPTY_DIRECT_COLLECTION_ERROR.errmsg
            )
        return self._format_questioner_output(output)

    async def _handle_user_interact_state(self, inputs, runtime: Runtime):
        output = OutputCache()
        self._query = await runtime.interact("")
        chat_history = self._get_latest_chat_history(runtime)
        user_response = chat_history[-1].get("content", "") if chat_history else ""

        if self._is_set_question_content() and not self._need_extract_fields():
            output.user_response = user_response
            self._state = self._state.handle_event(QuestionerEvent.END_EVENT)
            return self._format_questioner_output(output)

        if self._need_extract_fields():
            is_continue_ask = self._repeat_extract_from_chat_history(chat_history, output)
            event = QuestionerEvent.USER_INTERACT_EVENT if is_continue_ask else QuestionerEvent.END_EVENT
            self._state = self._state.handle_event(event)
        else:
            raise JiuWenBaseException(
                error_code=StatusCode.WORKFLOW_QUESTIONER_QUESTION_EMPTY_DIRECT_COLLECTION_ERROR.code,
                message=StatusCode.WORKFLOW_QUESTIONER_QUESTION_EMPTY_DIRECT_COLLECTION_ERROR.errmsg
            )
        return self._format_questioner_output(output)

    def _handle_end_state(self, inputs, runtime):
        output = QuestionerOutput(**self._state.extracted_key_fields)
        output.user_response = self._state.user_response
        return output.model_dump(exclude_defaults=True)

    def _is_set_question_content(self):
        return isinstance(self._config.question_content, str) and len(self._config.question_content) > 0

    def _need_extract_fields(self):
        return (self._config.extract_fields_from_response and
                len(self._config.field_names) > len(self._state.extracted_key_fields))

    def _initial_extract_from_chat_history(self, chat_history, output: OutputCache) -> bool:
        self._invoke_llm_and_parse_result(chat_history, output)

        self._update_param_default_value(output)
        self._update_state_of_key_fields(output.key_fields)

        return self._check_if_continue_ask(output)

    def _repeat_extract_from_chat_history(self, chat_history, output: OutputCache) -> bool:
        self._invoke_llm_and_parse_result(chat_history, output)

        self._update_param_default_value(output)
        self._update_state_of_key_fields(output.key_fields)

        return self._check_if_continue_ask(output)

    def _get_latest_chat_history(self, runtime: Runtime) -> List:
        result = list()
        if self._config.with_chat_history:
            raw_chat_history = runtime.store().read(WORKFLOW_CHAT_HISTORY) or list() # FIXME: remove to context engine
            if raw_chat_history:
                result = QuestionerUtils.get_latest_k_rounds_chat(raw_chat_history, self._config.chat_history_max_rounds)
        if not result or "user" == result[-1].get("role", ""):
            result.append(dict(role="user", content=self._query))
        return result

    def _build_llm_inputs(self, chat_history: list = None) -> List[BaseMessage]:
        prompt_template_input = self._create_prompt_template_keywords(chat_history)
        formatted_template: Template = self._prompt.format(prompt_template_input)
        return formatted_template.to_messages()

    def _create_prompt_template_keywords(self, chat_history):
        params_list, required_name_list = list(), list()
        for param in self._config.field_names:
            params_list.append(f"{param.field_name}: {param.description}")
            if param.required:
                required_name_list.append(param.cn_field_name or param.description)
        required_name_str = "、".join(required_name_list) + f"{len(required_name_list)}个必要信息"
        all_param_str = "\n".join(params_list)
        dialogue_history_str = "\n".join([f"{_.get('role', '')}：{_.get('content', '')}" for _ in chat_history])

        return dict(required_name=required_name_str, required_params_list=all_param_str,
                    extra_info=self._config.extra_prompt_for_fields_extraction, example=self._config.example_content,
                    dialogue_history=dialogue_history_str)

    def _invoke_llm_for_extraction(self, llm_inputs: List[BaseMessage]):
        try:
            response = self._model.invoke(
                model_name=self._config.model.model_info.model_name, messages=llm_inputs).content
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.INVOKE_LLM_FAILED.code,
                message=StatusCode.INVOKE_LLM_FAILED.errmsg
            ) from e

        result = dict()
        try:
            result = json.loads(response, strict=False)
            result = {k: v for k, v in result.items() if v is not None and str(v)}
        except json.JSONDecodeError as e:
            try:
                result = {k: v for k, v in ast.literal_eval(response).items() if v is not None and str(v)}
            except (SyntaxError, AttributeError, ValueError):
                return result
        return result

    def _filter_non_extracted_key_fields(self) -> List[FieldInfo]:
        return [_ for _ in self._config.field_names if _.field_name not in self._state.extracted_key_fields]

    def _update_state_of_key_fields(self, key_fields):
        self._state.extracted_key_fields.update(key_fields)

    def _update_param_default_value(self, output: OutputCache):
        result = dict()
        extracted_key_fields = self._state.extracted_key_fields
        for param in self._config.field_names:
            param_name = param.field_name
            default_value = param.default_value
            if default_value and param_name not in extracted_key_fields:
                result.update({param_name: default_value})
        output.key_fields.update(result)

    def _increment_state_of_response_num(self):
        self._state.response_num += 1

    def _exceed_max_response(self):
        return self._state.response_num > self._config.max_response

    def _check_if_continue_ask(self, output: OutputCache):
        is_continue_ask = False
        non_extracted_key_fields: List[FieldInfo] = self._filter_non_extracted_key_fields()
        if non_extracted_key_fields:
            if not self._exceed_max_response():
                output.question = QuestionerUtils.format_continue_ask_question(non_extracted_key_fields)
                is_continue_ask = True
            else:
                raise JiuWenBaseException(
                    error_code=StatusCode.WORKFLOW_QUESTIONER_EXCEED_LOOP.code,
                    message=StatusCode.WORKFLOW_QUESTIONER_EXCEED_LOOP.errmsg
                )
        if is_continue_ask:
            output.key_fields.clear()
        else:
            output.key_fields.update(self._state.extracted_key_fields)
        return is_continue_ask

    def _invoke_llm_and_parse_result(self, chat_history, output):
        llm_inputs = self._build_llm_inputs(chat_history=chat_history)
        extracted_key_fields = self._invoke_llm_for_extraction(llm_inputs)
        output.key_fields.update(extracted_key_fields)
        self._increment_state_of_response_num()
        self._update_state_of_key_fields(extracted_key_fields)

    @staticmethod
    def _format_questioner_output(output_cache: OutputCache) -> Dict:
        output = QuestionerOutput(**output_cache.key_fields)
        output.user_response = output_cache.user_response
        output.question = output_cache.question
        return output.model_dump(exclude_defaults=True)


class QuestionerExecutable(ComponentExecutable):
    def __init__(self, config: QuestionerConfig):
        super().__init__()
        self._config = config
        self._llm = self._create_llm_instance()
        self._prompt: Template = self._init_prompt()
        self._state = None

    @staticmethod
    def _load_state_from_runtime(runtime: Runtime) -> QuestionerState:
        questioner_state = runtime.get_state()
        state_dict = questioner_state.get(QUESTIONER_STATE_KEY) if isinstance(questioner_state, dict) else None
        if state_dict:
            return QuestionerState.deserialize(state_dict)
        return QuestionerState()

    @staticmethod
    def _store_state_to_runtime(state: QuestionerState, runtime: Runtime):
        state_dict = state.serialize()
        runtime.update_state({QUESTIONER_STATE_KEY: state_dict})

    def state(self, state: QuestionerState):
        self._state = state
        return self

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        await runtime.trace({"on_invoke_data": "extra trace data"})

        state_from_runtime = self._load_state_from_runtime(runtime)
        if state_from_runtime.is_undergoing_interaction():
            self._state = state_from_runtime

        if self._state is None:
            raise JiuWenBaseException(
                error_code=StatusCode.WORKFLOW_QUESTIONER_INIT_STATE_ERROR.code,
                message=StatusCode.WORKFLOW_QUESTIONER_INIT_STATE_ERROR.errmsg
            )
        self._state = self._state.handle_event(QuestionerEvent.START_EVENT)

        invoke_result = dict()
        if self._config.response_type == ResponseType.ReplyDirectly.value:
            invoke_result = await self._handle_questioner_direct_reply(inputs, runtime)

        self._store_state_to_runtime(self._state, runtime)

        # 向用户追问
        if self._state.is_undergoing_interaction():
            await runtime.interact(invoke_result.get("question", ""))

        return invoke_result

    def _create_llm_instance(self) -> BaseChatModel:
        return ModelFactory().get_model(model_provider=self._config.model.model_provider,
                                        api_base=self._config.model.model_info.api_base,
                                        api_key=self._config.model.model_info.api_key)

    def _init_prompt(self) -> Template:
        if self._config.prompt_template:
            return Template(name="question_user_prompt", content=self._config.prompt_template)

        filters = dict(model_name=self._config.model.model_info.model_name)
        return TemplateManager().get(name=TEMPLATE_NAME, filters=filters)

    async def _handle_questioner_direct_reply(self, inputs: Input, runtime: Runtime):
        handler = (QuestionerDirectReplyHandler()
                   .config(self._config).model(self._llm).state(self._state).prompt(self._prompt))
        result = await handler.handle(inputs, runtime)
        self._state = handler.get_state()
        return result



class QuestionerComponent(WorkflowComponent):
    def __init__(self, questioner_comp_config: QuestionerConfig = None):
        super().__init__()
        self._questioner_config = questioner_comp_config
        self._questioner_state = QuestionerState()
        self._executable = None

    def to_executable(self) -> Executable:
        return QuestionerExecutable(self._questioner_config).state(self._questioner_state)
