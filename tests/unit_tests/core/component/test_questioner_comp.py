import asyncio
import os
import time
import unittest
from unittest.mock import patch

import pytest

from openjiuwen.core.context_engine.config import ContextEngineConfig
from openjiuwen.core.context_engine.engine import ContextEngine
from openjiuwen.core.runtime.interaction.interaction import InteractionOutput
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.runtime.wrapper import TaskRuntime
from openjiuwen.core.common.constants.constant import INTERACTION
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.component.end_comp import End
from openjiuwen.core.component.questioner_comp import FieldInfo, QuestionerConfig, QuestionerComponent
from openjiuwen.core.component.start_comp import Start
from openjiuwen.core.graph.executable import Input
from openjiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from openjiuwen.core.runtime.workflow import WorkflowRuntime
from openjiuwen.core.stream.base import TraceSchema, OutputSchema
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.prompt.template.template import Template
from openjiuwen.core.workflow.base import Workflow, WorkflowExecutionState, WorkflowOutput
from openjiuwen.core.workflow.workflow_config import WorkflowConfig

class MockLLMModel:
    pass


class TestQuestionComp:
    @pytest.fixture(autouse=True)
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

    @patch("openjiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._invoke_llm_for_extraction")
    @patch("openjiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._build_llm_inputs")
    @patch("openjiuwen.core.component.questioner_comp.QuestionerExecutable._init_prompt")
    @patch("openjiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model")
    def test_invoke_questioner_component_in_workflow_initial_ask(self, mock_get_model, mock_init_prompt,
                                                                 mock_llm_inputs,
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

    @patch("openjiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._invoke_llm_for_extraction")
    @patch("openjiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._build_llm_inputs")
    @patch("openjiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model")
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
        payload = first_question.payload
        if isinstance(payload, InteractionOutput) and payload.id is not None:
            component_id = payload.id
        else:
            assert False
        user_input = InteractiveInput()
        user_input.update(component_id, "地点是杭州")  # 第一个入参是组件id

        workflow_context = TaskRuntime(trace_id=session_id).create_workflow_runtime()
        final_result = self.invoke_workflow_with_workflow_context(user_input, workflow_context,
                                                                  flow)  # workflow实例、session id保持一致
        assert final_result.result.get("responseContent") == "hangzhou | today"

    @patch("openjiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._invoke_llm_for_extraction")
    @patch("openjiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._build_llm_inputs")
    @patch("openjiuwen.core.component.questioner_comp.QuestionerExecutable._init_prompt")
    @patch("openjiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model")
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
    @patch("openjiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._invoke_llm_for_extraction")
    @patch("openjiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._build_llm_inputs")
    @patch("openjiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model")
    async def test_invoke_questioner_component_in_workflow_repeat_ask_with_stream_writer_and_context_engine(
            self,
            mock_get_model,
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
        mock_extraction.return_value = dict(time="tomorrow")

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
            with_chat_history=True
        )
        questioner_component = QuestionerComponent(questioner_comp_config=questioner_config)

        flow.set_start_comp("s", start_component, inputs_schema={"query": "${query}"})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"location": "${questioner.location}", "time": "${questioner.time}"})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${s.query}"})

        flow.add_connection("s", "questioner")
        flow.add_connection("questioner", "e")

        session_id = "test_questioner"
        config = ContextEngineConfig()
        ce_engine = ContextEngine("123", config)
        workflow_context = ce_engine.get_workflow_context(workflow_id="questioner_workflow", session_id=session_id)
        workflow_runtime = TaskRuntime(trace_id=session_id).create_workflow_runtime()
        interaction_output_schema = []
        async for chunk in flow.stream({"query": "你好"}, workflow_runtime, workflow_context):
            if isinstance(chunk, OutputSchema) and chunk.type == INTERACTION:
                interaction_output_schema.append(chunk)

        mock_extraction.return_value = dict(location="hangzhou")

        if interaction_output_schema:
            user_input = InteractiveInput()
            for item in interaction_output_schema:
                component_id = item.payload.id
                user_input.update(component_id, "杭州")
            workflow_runtime = TaskRuntime(trace_id=session_id).create_workflow_runtime()
            async for chunk in flow.stream(user_input, workflow_runtime, workflow_context):
                print(f"stream output >>> {chunk}")

    @unittest.skip("skip system test")
    @pytest.mark.asyncio  # 新增
    async def test_real_workflow_stream_start_questioner_end_with_interaction(self):
        flow = Workflow(workflow_config=WorkflowConfig())

        start_component = Start(
            {
                "inputs": [
                    {"id": "query", "type": "String", "required": "true", "sourceType": "ref"}
                ]
            }
        )
        end_component = End({"responseTemplate": "{{location}} | {{time}}"})

        API_BASE = os.getenv("API_BASE", "")
        API_KEY = os.getenv("API_KEY", "")
        MODEL_NAME = os.getenv("MODEL_NAME", "")
        MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")
        model_config = ModelConfig(model_provider=MODEL_PROVIDER,
                                   model_info=BaseModelInfo(
                                       model=MODEL_NAME,
                                       api_base=API_BASE,
                                       api_key=API_KEY,
                                       temperature=0.7,
                                       top_p=0.9,
                                       timeout=30  # 添加超时设置
                                   ))

        key_fields = [
            FieldInfo(field_name="location", description="地点", required=True),
            FieldInfo(field_name="time", description="时间", required=True, default_value="today")
        ]
        questioner_config = QuestionerConfig(
            model=model_config,
            question_content="",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=True
        )
        questioner_component = QuestionerComponent(questioner_comp_config=questioner_config)

        flow.set_start_comp("s", start_component, inputs_schema={"query": "${query}"})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"location": "${questioner.location}", "time": "${questioner.time}"})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${s.query}"})

        flow.add_connection("s", "questioner")
        flow.add_connection("questioner", "e")

        session_id = "test_questioner"
        config = ContextEngineConfig()
        ce_engine = ContextEngine("123", config)
        workflow_context = ce_engine.get_workflow_context(workflow_id="questioner_workflow", session_id=session_id)
        workflow_runtime = TaskRuntime(trace_id=session_id).create_workflow_runtime()
        interaction_output_schema = list()
        async for chunk in flow.stream({"query": "时间为2025-10-01"}, workflow_runtime, workflow_context):
            if isinstance(chunk, OutputSchema) and chunk.type == INTERACTION:
                interaction_output_schema.append(chunk)

        if interaction_output_schema:
            user_input = InteractiveInput()
            for item in interaction_output_schema:
                component_id = item.payload.id
                user_input.update(component_id, "地点是杭州")
            workflow_runtime = TaskRuntime(trace_id=session_id).create_workflow_runtime()
            async for chunk in flow.stream(user_input, workflow_runtime, workflow_context):
                print(f"stream output >>> {chunk}")

    @unittest.skip("skip system test")
    @pytest.mark.asyncio  # 新增
    async def test_real_workflow_invoke_start_questioner_end_with_interaction(self):
        flow = Workflow(workflow_config=WorkflowConfig())

        start_component = Start(
            {
                "inputs": [
                    {"id": "query", "type": "String", "required": "true", "sourceType": "ref"}
                ]
            }
        )
        end_component = End({"responseTemplate": "{{location}} | {{time}}"})

        API_BASE = os.getenv("API_BASE", "")
        API_KEY = os.getenv("API_KEY", "")
        MODEL_NAME = os.getenv("MODEL_NAME", "")
        MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")
        model_config = ModelConfig(model_provider=MODEL_PROVIDER,
                                   model_info=BaseModelInfo(
                                       model=MODEL_NAME,
                                       api_base=API_BASE,
                                       api_key=API_KEY,
                                       temperature=0.7,
                                       top_p=0.9,
                                       timeout=30  # 添加超时设置
                                   ))

        key_fields = [
            FieldInfo(field_name="location", description="地点", required=True),
            FieldInfo(field_name="time", description="时间", required=True, default_value="today")
        ]
        questioner_config = QuestionerConfig(
            model=model_config,
            question_content="",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=True
        )
        questioner_component = QuestionerComponent(questioner_comp_config=questioner_config)

        flow.set_start_comp("s", start_component, inputs_schema={"query": "${query}"})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"location": "${questioner.location}", "time": "${questioner.time}"})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${s.query}"})

        flow.add_connection("s", "questioner")
        flow.add_connection("questioner", "e")

        session_id = "test_questioner"
        workflow_runtime = TaskRuntime(trace_id=session_id).create_workflow_runtime()
        workflow_result = await flow.invoke({"query": "时间为2025-10-01"}, workflow_runtime)
        assert workflow_result.state == WorkflowExecutionState.INPUT_REQUIRED

        time.sleep(3)

        if workflow_result.state == WorkflowExecutionState.INPUT_REQUIRED:
            component_id = workflow_result.result[0].payload.id
            assert component_id == "questioner"
            workflow_runtime = TaskRuntime(trace_id=session_id).create_workflow_runtime()
            user_feedback = InteractiveInput()
            user_feedback.update(component_id, "地点是杭州")
            workflow_result = await flow.invoke(user_feedback, workflow_runtime)
            assert workflow_result.state == WorkflowExecutionState.COMPLETED
            assert workflow_result.result.get("responseContent", "") == "杭州 | 2025-10-01"
