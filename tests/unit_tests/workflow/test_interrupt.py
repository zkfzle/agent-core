import os
import unittest
from datetime import datetime
from unittest.mock import patch

from jiuwen.agent.common.enum import SubTaskType
from jiuwen.agent.common.schema import PluginSchema, WorkflowSchema
from jiuwen.agent.react_agent import create_react_agent_config, create_react_agent, ReActAgent
from jiuwen.core.agent.controller.react_controller import ReActControllerOutput
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.questioner_comp import FieldInfo, QuestionerConfig, QuestionerComponent
from jiuwen.core.component.start_comp import Start
from jiuwen.core.utils.llm.base import BaseModelInfo
from jiuwen.core.utils.llm.messages import AIMessage
from jiuwen.core.utils.tool.service_api.param import Param
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import WorkflowConfig, WorkflowMetadata
from jiuwen.graph.pregel.graph import PregelGraph
from tests.unit_tests.workflow.test_workflow import create_flow

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
        system_prompt = "你是一个AI助手，在适当的时候调用合适的工作流，帮助我问一下查询什么城市的天气"
        return [
            dict(role="system", content=system_prompt.format(build_current_date()))
        ]

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

        flow = Workflow(workflow_config=questioner_workflow_config, graph=PregelGraph())

        key_fields = [
            FieldInfo(field_name="location", description="地点", required=True),
            FieldInfo(field_name="time", description="时间", required=True, default_value="today")
        ]

        start_component = Start("s",
                                {
                                    "userFields": {"inputs": [], "outputs": []},
                                    "systemFields": {"input": [
                                        {"id": "query", "type": "String", "required": "true", "sourceType": "ref"}
                                    ]
                                    }
                                }
                                )
        end_component = End("e", "e", {"responseTemplate": "{{output}}"})

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

        flow.set_start_comp("s", start_component, inputs_schema={"systemFields": {"query": "${query}"}})
        flow.set_end_comp("e", end_component,
                          inputs_schema={"userFields": {"output": "${questioner.userFields.key_fields}"}})
        flow.add_workflow_comp("questioner", questioner_component, inputs_schema={"query": "${start.query}"})

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
            result = await react_agent.invoke({"conversation_id": "12345", "query": "杭州"})
            print(f"ReActAgent 第二次输出结果：{result}")
