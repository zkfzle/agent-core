import os
import unittest
from datetime import datetime
from unittest.mock import patch

import pytest

from jiuwen.agent.common.enum import SubTaskType, ControllerType
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.agent.react_agent import create_react_agent_config, create_react_agent, ReActAgent
from jiuwen.agent.workflow_agent import WorkflowAgent
from jiuwen.core.agent.controller.react_controller import ReActControllerOutput
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.questioner_comp import FieldInfo, QuestionerConfig, QuestionerComponent
from jiuwen.core.component.start_comp import Start
from jiuwen.core.stream.writer import OutputSchema
from jiuwen.core.utils.llm.base import BaseModelInfo
from jiuwen.core.utils.llm.messages import AIMessage
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import WorkflowConfig, WorkflowMetadata
from jiuwen.graph.pregel.graph import PregelGraph

API_BASE = os.getenv("API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")

def build_current_date():
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d")

class MockLLMModel:
    def model_provider(self):
        return MODEL_PROVIDER


class ReActAgentInterruptTest(unittest.IsolatedAsyncioTestCase):  # ① 关键改动
    @staticmethod
    def _create_model():
        return ModelConfig(model_provider=MODEL_PROVIDER,
                           model_info=BaseModelInfo(
                               model=MODEL_NAME,
                               api_base=API_BASE,
                               api_key=API_KEY,
                               temperature=0.7,
                               top_p=0.9,
                               timeout=30  # 添加超时设置
                           ))

    @staticmethod
    def _create_prompt_template():
        system_prompt = "你是一个AI助手，在适当的时候调用合适的工作流，帮助我查询一下天气"
        return [
            dict(role="system", content=system_prompt.format(build_current_date()))
        ]

    # Todo: 临时关闭
    @unittest.skip("skip system test")
    @pytest.mark.asyncio
    @patch("jiuwen.core.agent.controller.react_controller.ReActController.invoke")
    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._invoke_llm_for_extraction")
    @patch("jiuwen.core.component.questioner_comp.QuestionerDirectReplyHandler._build_llm_inputs")
    @patch("jiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model")
    async def test_react_agent_invoke_with_workflow_interrupt(self, mock_get_model, mock_llm_inputs,
                                                                mock_extraction, mock_react_controller_invoke):
        mock_get_model.return_value = MockLLMModel()

        react_agent_prompt_template = self._create_prompt_template()

        mock_prompt_template = [
            dict(role="system", content="系统提示词"),
            dict(role="user", content="你是一个AI助手")
        ]

        mock_llm_inputs.return_value = mock_prompt_template
        mock_extraction.return_value = dict(location="hangzhou")

        questioner_workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                name="questioner",
                id="questioner_workflow",
                version="1.0",
            )
        )

        flow = Workflow(workflow_config=questioner_workflow_config)

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

        workflow_schema = WorkflowSchema(
            id = flow.config().metadata.id,
            name = flow.config().metadata.name,
            version = flow.config().metadata.version,
            description = "追问器工作流",
            inputs = {"query": {
                "type": "string",
            }}
        )

        sub_task = SubTask(
            id = workflow_schema.id,
            sub_task_type = SubTaskType.WORKFLOW,
            func_id = f"{workflow_schema.id}_{workflow_schema.version}",
            func_name = workflow_schema.name,
        )

        react_agent_config = create_react_agent_config(
            agent_id="react_agent_123",
            agent_version="0.0.1",
            description="AI助手",
            plugins=[],
            workflows=[workflow_schema],
            model=model_config,
            prompt_template=react_agent_prompt_template
        )

        # react_agent要创建，但要打桩下面的逻辑：1. 大模型创建； 2. 大模型输出
        react_agent: ReActAgent = create_react_agent(
            agent_config=react_agent_config,
            workflows=[flow],
            tools=[]
        )

        # 第一次大模型返回的结果让调用sub_task
        mock_react_controller_invoke.return_value = ReActControllerOutput(
            should_continue = True,
            llm_output=AIMessage(content = "This is first mock LLM output"),
            sub_tasks=[sub_task],
        )

        result = await react_agent.invoke({"conversation_id": "12345", "query": "查询杭州的天气"})
        print(f"ReActAgent 第一次输出结果：{result}")

        # 第二次大模型返回的结果不让调用sub_task
        mock_react_controller_invoke.return_value = ReActControllerOutput(
            should_continue=False,
            llm_output=AIMessage(content="This is second mock LLM output"),
            sub_tasks=[sub_task],
        )
        if result.get("result_type") == 'question':
            result = await react_agent.invoke({"conversation_id": "12345", "query": "查询杭州天气"})
            print(f"ReActAgent 第二次输出结果：{result}")

    @unittest.skip("skip system test")
    async def test_real_react_agent_invoke_with_workflow_interrupt(self):
        react_agent_prompt_template = self._create_prompt_template()

        prompt_template = [
            dict(role="system", content="系统提示词"),
            dict(role="user", content="你是一个AI助手")
        ]

        questioner_workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                name="questioner",
                id="questioner_workflow",
                version="1.0",
            )
        )

        flow = Workflow(workflow_config=questioner_workflow_config)

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
        end_component = End({"responseTemplate": "{{output}}"})

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

        questioner_config = QuestionerConfig(
            model=model_config,
            question_content="查询什么城市的天气",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=False,
            prompt_template=prompt_template
        )
        questioner_component = QuestionerComponent(questioner_comp_config=questioner_config)

        flow.set_start_comp("s", start_component, inputs_schema={"query": "${query}"})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"output": "${questioner.userFields.key_fields}"})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${start.query}"})

        flow.add_connection("s", "questioner")
        flow.add_connection("questioner", "e")

        workflow_schema = WorkflowSchema(
            id=flow.config().metadata.id,
            name=flow.config().metadata.name,
            version=flow.config().metadata.version,
            description="追问器工作流",
            inputs={"query": {
                "type": "string",
            }}
        )

        sub_task = SubTask(
            id=workflow_schema.id,
            sub_task_type=SubTaskType.WORKFLOW,
            func_id=f"{workflow_schema.id}_{workflow_schema.version}",
            func_name=workflow_schema.name,
        )

        react_agent_config = create_react_agent_config(
            agent_id="react_agent_123",
            agent_version="0.0.1",
            description="AI助手",
            plugins=[],
            workflows=[workflow_schema],
            model=model_config,
            prompt_template=react_agent_prompt_template
        )

        # react_agent要创建，但要打桩下面的逻辑：1. 大模型创建； 2. 大模型输出
        react_agent: ReActAgent = create_react_agent(
            agent_config=react_agent_config,
            workflows=[flow],
            tools=[]
        )

        # 第一次大模型返回的结果要让调用sub_task
        result = await react_agent.invoke({"conversation_id": "12345", "query": "查询今天天气"})
        print(f"ReActAgent 第一次输出结果：{result}")

        # 第二次大模型返回的结果不让调用sub_task
        if result.get("result_type") == 'question':
            result = await react_agent.invoke({"conversation_id": "12345", "query": "查询杭州天气"})
            print(f"ReActAgent 第二次输出结果：{result}")


    @unittest.skip("skip system test")
    async def test_real_workflow_agent_invoke_with_workflow_interrupt(self):
        questioner_workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                name="questioner",
                id="questioner_workflow",
                version="1.0",
            )
        )

        flow = Workflow(workflow_config=questioner_workflow_config)

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

        workflow_schema = WorkflowSchema(
            id=flow.config().metadata.id,
            name=flow.config().metadata.name,
            version=flow.config().metadata.version,
            description="追问器工作流",
            inputs={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户输入",
                        "required": True
                    }
                }
            }
        )

        config = WorkflowAgentConfig(
            id="write_agent",
            version="0.1.0",
            description="interrupt workflow agent",
            workflows=[workflow_schema],
            controller_type=ControllerType.WorkflowController,
        )

        workflow_agent = WorkflowAgent(config)
        workflow_agent.bind_workflows([flow])

        result = await workflow_agent.invoke({"conversation_id": "12345", "query": "查询今天天气"})
        print(f"WorkflowAgent 第一次输出结果：{result}")

        if result.get("result_type") == 'question':
            result = await workflow_agent.invoke({"conversation_id": "12345", "query": "地点是杭州"})
            print(f"WorkflowActAgent 第二次输出结果：{result}")

    @unittest.skip("skip system test")
    async def test_real_workflow_agent_stream_with_workflow_interrupt(self):
        questioner_workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                name="questioner",
                id="questioner_workflow",
                version="1.0",
            )
        )

        flow = Workflow(workflow_config=questioner_workflow_config)

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

        workflow_schema = WorkflowSchema(
            id=flow.config().metadata.id,
            name=flow.config().metadata.name,
            version=flow.config().metadata.version,
            description="追问器工作流",
            inputs={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户输入",
                        "required": True
                    }
                }
            }
        )

        config = WorkflowAgentConfig(
            id="write_agent",
            version="0.1.0",
            description="interrupt workflow agent",
            workflows=[workflow_schema],
            controller_type=ControllerType.WorkflowController,
        )

        workflow_agent = WorkflowAgent(config)
        workflow_agent.bind_workflows([flow])

        is_interaction = False
        async for result in workflow_agent.stream({"conversation_id": "12345", "query": "查询今天天气"}):
            print(f"WorkflowAgent stream 第一次输出结果：{result}")
            if isinstance(result, OutputSchema) and result.type == "__interaction__":
                is_interaction = True

        if is_interaction:
            async for result in workflow_agent.stream({"conversation_id": "12345", "query": "地点是杭州"}):
                print(f"WorkflowActAgent 第二次输出结果：{result}")