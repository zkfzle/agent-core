import asyncio
import unittest
from unittest.mock import patch

import pytest

from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.runtime.wrapper import TaskRuntime
from jiuwen.core.common.constants.constant import INTERACTION
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.questioner_comp import FieldInfo, QuestionerConfig, QuestionerComponent
from jiuwen.core.component.start_comp import Start
from jiuwen.core.graph.executable import Input
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.runtime.workflow import WorkflowRuntime
from jiuwen.core.stream.writer import TraceSchema, OutputSchema
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.workflow.base import Workflow, WorkflowExecutionState, WorkflowOutput


class MockLLMModel:
    pass

class QuestionerTest(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    @staticmethod
    def invoke_workflow(inputs: Input, context: Runtime, flow: Workflow):
        loop = asyncio.get_event_loop()
        feature = asyncio.ensure_future(flow.invoke(inputs=inputs, runtime=context.create_workflow_runtime()))
        loop.run_until_complete(feature)
        return feature.result()

    @staticmethod
    def invoke_workflow_with_workflow_context(inputs: Input, runtime: WorkflowRuntime, flow: Workflow):
        loop = asyncio.get_event_loop()
        feature = asyncio.ensure_future(flow.invoke(inputs=inputs, runtime=runtime))
        loop.run_until_complete(feature)
        return feature.result()

    @staticmethod
    def _create_context(session_id):
        return TaskRuntime(trace_id=session_id)

    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._invoke_llm_for_extraction")
    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._build_llm_inputs")
    @patch("jiuwen.core.component.questioner_comp.QuestionerExecutable._init_prompt")
    @patch("jiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model")
    def test_invoke_questioner_component_in_workflow_initial_ask(self, mock_get_model, mock_init_prompt, mock_llm_inputs,
                                                                 mock_extraction):
        mock_get_model.return_value = MockLLMModel()
        mock_prompt_template = [
            dict(role="system", content="系统提示词"),
            dict(role="user", content="你是一个AI助手")
        ]
        mock_init_prompt.return_value = Template(name="test", content=mock_prompt_template)
        mock_llm_inputs.return_value = mock_prompt_template
        mock_extraction.return_value = dict(location="hangzhou")

        context = TaskRuntime(trace_id="test")
        flow = Workflow()

        key_fields = [
            FieldInfo(field_name="location", description="地点", required=True),
            FieldInfo(field_name="time", description="时间", required=True, default_value="today")
        ]

        start_component = Start(
            {
                "inputs": [
                    {"id": "query", "type": "String", "required": "true", "sourceType": "ref"}
                ]
            }
        )
        end_component = End({"responseTemplate": "{{location}} | {{time}}"})

        model_config = ModelConfig(model_provider="openai")
        questioner_config = QuestionerConfig(
            model=model_config,
            question_content="",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=False
        )
        questioner_component = QuestionerComponent(questioner_comp_config=questioner_config)

        flow.set_start_comp("s", start_component, inputs_schema={"query": "${query}"})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"location": "${questioner.location}", "time": "${questioner.time}"})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${s.query}"})

        flow.add_connection("s", "questioner")
        flow.add_connection("questioner", "e")

        result = self.invoke_workflow({"query": "查询杭州的天气"}, context, flow)
        assert result == WorkflowOutput(
            result={'output': {}, 'responseContent': "hangzhou | today"},
            state=WorkflowExecutionState.COMPLETED)


    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._invoke_llm_for_extraction")
    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._build_llm_inputs")
    @patch("jiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model")
    def test_invoke_questioner_component_in_workflow_repeat_ask(self, mock_get_model, mock_llm_inputs,
                                                                mock_extraction):
        """
        测试提问器中断恢复流程
        """
        mock_get_model.return_value = MockLLMModel()
        mock_prompt_template = [
            dict(role="system", content="系统提示词"),
            dict(role="user", content="你是一个AI助手")
        ]
        mock_llm_inputs.return_value = mock_prompt_template
        mock_extraction.return_value = dict(location="hangzhou")

        flow = Workflow()

        key_fields = [
            FieldInfo(field_name="location", description="地点", required=True),
            FieldInfo(field_name="time", description="时间", required=True, default_value="today")
        ]

        start_component = Start(
            {
                "inputs": [
                    {"id": "query", "type": "String", "required": "true", "sourceType": "ref"}
                ]
            }
        )
        end_component = End({"responseTemplate": "{{location}} | {{time}}"})

        model_config = ModelConfig(model_provider="openai")
        questioner_config = QuestionerConfig(
            model=model_config,
            question_content="查询什么城市的天气",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=False,
            prompt_template=mock_prompt_template
        )
        questioner_component = QuestionerComponent(questioner_comp_config=questioner_config)

        flow.set_start_comp("s", start_component, inputs_schema={"query": "${query}"})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"location": "${questioner.location}", "time": "${questioner.time}"})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${s.query}"})

        flow.add_connection("s", "questioner")
        flow.add_connection("questioner", "e")

        session_id = "test_questioner"
        workflow_context = TaskRuntime(trace_id=session_id).create_workflow_runtime()
        first_question = self.invoke_workflow_with_workflow_context({"query": "你好"}, workflow_context, flow)
        first_question = first_question.result[0] if first_question else dict()
        payload = first_question.get("payload")
        if isinstance(payload, dict) and payload.get("id") is not None:
            component_id = payload.get("id")
        else:
            assert False
        user_input = InteractiveInput()
        user_input.update(component_id, "地点是杭州")  # 第一个入参是组件id

        workflow_context = TaskRuntime(trace_id=session_id).create_workflow_runtime()
        final_result = self.invoke_workflow_with_workflow_context(user_input, workflow_context, flow)    # workflow实例、session id保持一致
        assert final_result.result.get("responseContent") == "hangzhou | today"

    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._invoke_llm_for_extraction")
    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._build_llm_inputs")
    @patch("jiuwen.core.component.questioner_comp.QuestionerExecutable._init_prompt")
    @patch("jiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model")
    def test_stream_questioner_component_in_workflow_initial_ask_with_tracer(self, mock_get_model, mock_init_prompt,
                                                                            mock_llm_inputs, mock_extraction):
        '''
        tracer使用问题记录：
        1. 必须调用workflow、agent的 stream方法，才能获取到tracer的数据帧
        2. workflow组件的输入输出，tracer都已经记录了，组件只需要关注额外的数据
        3. agent用agent_tracer，workflow用workflow_tracer
        4. event有定义，参考handler.py的@trigger_event
        5. on_invoke_data是固定的结构，结构为 dict(on_invoke_data={"on_invoke_data": "extra trace data"})
        6. span只有agent需要
        7. workflow不需要context显式地调用set_tracer
        '''

        mock_get_model.return_value = MockLLMModel()
        mock_prompt_template = [
            dict(role="system", content="系统提示词"),
            dict(role="user", content="你是一个AI助手")
        ]
        mock_init_prompt.return_value = Template(name="test", content=mock_prompt_template)
        mock_llm_inputs.return_value = mock_prompt_template
        mock_extraction.return_value = dict(location="hangzhou")

        context = TaskRuntime(trace_id="test")
        flow = Workflow()

        key_fields = [
            FieldInfo(field_name="location", description="地点", required=True),
            FieldInfo(field_name="time", description="时间", required=True, default_value="today")
        ]

        start_component = Start(
            {
                "inputs": [
                    {"id": "query", "type": "String", "required": "true", "sourceType": "ref"}
                ]
            }
        )
        end_component = End({"responseTemplate": "{{location}} | {{time}}"})

        model_config = ModelConfig(model_provider="openai")
        questioner_config = QuestionerConfig(
            model=model_config,
            question_content="",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=False
        )
        questioner_component = QuestionerComponent(questioner_comp_config=questioner_config)

        flow.set_start_comp("s", start_component, inputs_schema={"query": "${query}"})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"location": "${questioner.location}", "time": "${questioner.time}"})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${s.query}"})

        flow.add_connection("s", "questioner")
        flow.add_connection("questioner", "e")

        async def _async_stream_workflow_for_tracer(_flow, _inputs, _runtime, _tracer_chunks):
            async for chunk in flow.stream(_inputs, _runtime):
                if isinstance(chunk, TraceSchema):
                    _tracer_chunks.append(chunk)

        tracer_chunks = []
        self.loop.run_until_complete(_async_stream_workflow_for_tracer(flow, {"query": "查询杭州的天气"},
                                                                       context.create_workflow_runtime(),
                                                                       tracer_chunks))
        print(tracer_chunks)


class TestQuestionerStream:
    @pytest.mark.asyncio
    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._invoke_llm_for_extraction")
    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._build_llm_inputs")
    @patch("jiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model")
    async def test_invoke_questioner_component_in_workflow_repeat_ask_with_stream_writer(self, mock_get_model,
                                                                                         mock_llm_inputs,
                                                                                         mock_extraction):
        """
        测试提问器中断恢复流程
        """
        mock_get_model.return_value = MockLLMModel()
        mock_prompt_template = [
            dict(role="system", content="系统提示词"),
            dict(role="user", content="你是一个AI助手")
        ]
        mock_llm_inputs.return_value = mock_prompt_template
        mock_extraction.return_value = dict(location="hangzhou")

        flow = Workflow()

        key_fields = [
            FieldInfo(field_name="location", description="地点", required=True),
            FieldInfo(field_name="time", description="时间", required=True, default_value="today")
        ]

        start_component = Start(
            {
                "inputs": [
                    {"id": "query", "type": "String", "required": "true", "sourceType": "ref"}
                ]
            }
        )
        end_component = End({"responseTemplate": "{{location}} | {{time}}"})

        model_config = ModelConfig(model_provider="openai")
        questioner_config = QuestionerConfig(
            model=model_config,
            question_content="查询什么城市的天气",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=False,
            prompt_template=mock_prompt_template
        )
        questioner_component = QuestionerComponent(questioner_comp_config=questioner_config)

        flow.set_start_comp("s", start_component, inputs_schema={"query": "${query}"})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"output": "${questioner.userFields.key_fields}"})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${start.query}"})

        flow.add_connection("s", "questioner")
        flow.add_connection("questioner", "e")

        session_id = "test_questioner"
        is_interaction = False
        component_id = ""
        workflow_context = TaskRuntime(trace_id=session_id).create_workflow_runtime()
        async for chunk in flow.stream({"query": "你好"}, workflow_context):
            if isinstance(chunk, OutputSchema) and chunk.type == INTERACTION:
                is_interaction = True | is_interaction
                interaction_chunk = chunk
        if is_interaction:
            if interaction_chunk and isinstance(interaction_chunk.payload, tuple) and len(interaction_chunk.payload) == 2:
                component_id, question = interaction_chunk.payload
                assert question == "查询什么城市的天气"

        user_input = InteractiveInput()
        user_input.update(component_id, "地点是杭州")  # 第一个入参是组件id
        workflow_context = TaskRuntime(trace_id=session_id).create_workflow_runtime()
        async for chunk in flow.stream(user_input, workflow_context):
            print(f"stream output >>> {chunk}")
