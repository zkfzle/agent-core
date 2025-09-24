# tests/test_workflow_agent_invoke_real.py
import os
from datetime import datetime
import unittest

import pytest

from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.core.runtime.wrapper import TaskRuntime
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.intent_detection_comp import IntentDetectionComponent, IntentDetectionConfig
from jiuwen.core.component.llm_comp import LLMComponent, LLMCompConfig
from jiuwen.core.component.questioner_comp import QuestionerComponent, QuestionerConfig, FieldInfo
from jiuwen.core.component.start_comp import Start
from jiuwen.core.component.tool_comp import ToolComponent, ToolComponentConfig
from jiuwen.core.runtime.agent_context import AgentContext
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.utils.llm.base import BaseModelInfo
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.tool.param import Param
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import WorkflowConfig, WorkflowMetadata
from jiuwen.graph.pregel.graph import PregelGraph

API_BASE = os.getenv("API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")
# Mock RESTful Api 元信息
_MOCK_TOOL = RestfulApi(
    name="test",
    description="test",
    params=[
        Param(name="location", description="地点", type="string"),
        Param(name="date", description="日期", type="int"),
    ],
    path="http://127.0.0.1:8000",
    headers={},
    method="GET",
    response=[],
)

_FINAL_RESULT: str = "上海今天晴 30°C"

# --------------------------- Prompt 模板 --------------------------- #
_QUESTIONER_SYSTEM_TEMPLATE = """\
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

_QUESTIONER_USER_TEMPLATE = """\
对话历史
{{dialogue_history}}

请充分考虑以上对话历史及用户输入，正确提取最符合约束要求的 JSON 格式参数。
"""

SYSTEM_PROMPT_TEMPLATE = "你是一个query改写的AI助手。今天的日期是{}。"


def build_current_date():
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d")


class WorkflowAgentTest(unittest.IsolatedAsyncioTestCase):
    """专门用于测试 WorkflowAgent.invoke 的类。"""

    @staticmethod
    def _create_model_config() -> ModelConfig:
        """根据环境变量构造模型配置。"""
        return ModelConfig(
            model_provider=MODEL_PROVIDER,
            model_info=BaseModelInfo(
                model=MODEL_NAME,
                api_base=API_BASE,
                api_key=API_KEY,
                temperature=0.7,
                top_p=0.9,
                timeout=30,
            ),
        )

    @staticmethod
    def _create_intent_detection_component() -> IntentDetectionComponent:
        """创建意图识别组件。"""
        model_config = WorkflowAgentTest._create_model_config()
        user_prompt = """
        {{user_prompt}}

        当前可供选择的功能分类如下：
        {{category_info}}

        用户与助手的对话历史：
        {{chat_history}}

        当前输入：
        {{input}}

        请根据当前输入和对话历史分析并输出最适合的功能分类。输出格式为 JSON：
        {"class": "分类xx"}
        如果没有合适的分类，请输出 {{default_class}}。
        """
        config = IntentDetectionConfig(
            user_prompt="请判断用户意图",
            category_info="",
            category_list=["分类1", "分类2"],
            category_name_list=["默认意图", "查询某地天气"],
            default_class="分类1",
            model=model_config,
            intent_detection_template=Template(
                name="default",
                content=[{"role": "user", "content": user_prompt}],
            ),
            enable_input=True,
        )
        component = IntentDetectionComponent(config)
        component.add_branch("${intent.classificationId} == 0", ["end"], "默认分支")
        component.add_branch("${intent.classificationId} == 1", ["llm"], "查询天气分支")
        return component

    @staticmethod
    def _create_llm_component() -> LLMComponent:
        """创建 LLM 组件，仅用于抽取结构化字段（location/date）。"""
        model_config = WorkflowAgentTest._create_model_config()
        current_date = build_current_date()
        user_prompt = ("\n原始query为：{{query}}\n\n帮我改写原始query，要求：\n"
                       "1. 只把地名改为英文，其他信息保留中文；\n"
                       "2. 改写后的query必须包含当前的日期，默认日期为今天；\n"
                       "3. 日期为YYYY-MM-DD格式。")
        config = LLMCompConfig(
            model=model_config,
            template_content=[{"role": "user", "content": SYSTEM_PROMPT_TEMPLATE.format(current_date) + user_prompt}],
            response_format={"type": "text"},
            output_config={
                "query": {"type": "string", "description": "改写后的query", "required": True}
            },
        )
        return LLMComponent(config)

    @staticmethod
    def _create_questioner_component() -> QuestionerComponent:
        """创建信息收集组件。"""
        key_fields = [
            FieldInfo(field_name="location", description="地点", required=True),
            FieldInfo(
                field_name="date",
                description="时间",
                required=True,
                default_value="today",
            ),
        ]
        model_config = WorkflowAgentTest._create_model_config()
        config = QuestionerConfig(
            model=model_config,
            question_content="",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=False,
            prompt_template=[
                {"role": "system", "content": _QUESTIONER_SYSTEM_TEMPLATE},
                {"role": "user", "content": _QUESTIONER_USER_TEMPLATE},
            ],
        )
        return QuestionerComponent(config)

    @staticmethod
    def _create_plugin_component() -> ToolComponent:
        """创建插件组件，真正调用外部 RESTful API。"""
        tool_config = ToolComponentConfig(needValidate=False)
        weather_tool = RestfulApi(
            name="WeatherReporter",
            description="天气查询插件",
            params=[
                Param(name="location", description="地点", type="string", required=True),
                Param(name="date", description="日期", type="string", required=True),
            ],
            path="http://127.0.0.1:9000/weather",
            headers={},
            method="GET",
            response=[],
        )
        return ToolComponent(tool_config).set_tool(weather_tool)

    @staticmethod
    def _create_start_component():
        return Start({"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]})

    @staticmethod
    def _create_end_component():
        return End({"responseTemplate": "{{output}}"})

    def _build_workflow(self) -> tuple[BaseRuntime, Workflow]:
        """
        根据 mock 工具函数构建完整工作流拓扑。

        返回 (context, workflow) 二元组，可直接用于 invoke。
        """
        # 1. 初始化工作流与上下文
        id = "test_weather_agent"
        version = "1.0"
        name = "weather"
        workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                name=name,
                id=id,
                version=version,
            )
        )
        flow = Workflow(
            workflow_config=workflow_config,
            graph=PregelGraph(),
        )
        context = TaskRuntime(trace_id="test")

        # 2. 实例化各组件
        start = self._create_start_component()
        intent = self._create_intent_detection_component()
        llm = self._create_llm_component()
        questioner = self._create_questioner_component()
        plugin = self._create_plugin_component()
        end = self._create_end_component()

        # 3. 注册组件到工作流
        flow.set_start_comp(
            "start",
            start,
            inputs_schema={"query": "${query}"},
        )
        flow.add_workflow_comp(
            "intent",
            intent,
            inputs_schema={"input": "${start.systemFields.query}"},
        )
        flow.add_workflow_comp(
            "llm",
            llm,
            inputs_schema={"userFields": {"query": "${start.systemFields.query}"}},
        )
        flow.add_workflow_comp(
            "questioner",
            questioner,
            inputs_schema={"query": "${llm.userFields.query}"}
        )
        flow.add_workflow_comp(
            "plugin",
            plugin,
            inputs_schema={
                "userFields": "${questioner.userFields.key_fields}",
                "validated": True,
            },
        )
        flow.set_end_comp("end", end, inputs_schema={"userFields": {"output": "${plugin.data}"}})

        # 4. 连接拓扑
        flow.add_connection("start", "intent")
        flow.add_connection("llm", "questioner")
        flow.add_connection("questioner", "plugin")
        flow.add_connection("plugin", "end")

        return context.create_workflow_runtime(), flow

    @staticmethod
    def _create_workflow_schema(id, name: str, version: str) -> WorkflowSchema:
        return WorkflowSchema(id=id,
                              name=name,
                              description="天气查询工作流",
                              version=version,
                              inputs={"query": {
                                  "type": "string",
                              }})

    def _create_agent(self, workflow):
        """根据 workflow 实例化 WorkflowAgent。"""
        from jiuwen.agent.workflow_agent import WorkflowAgent  # 你给出的类
        workflow_id = workflow.config().metadata.id
        workflow_name = workflow.config().metadata.name
        workflow_version = workflow.config().metadata.version
        schema = self._create_workflow_schema(workflow_id, workflow_name, workflow_version)
        config = WorkflowAgentConfig(
            id="test_weather_agent",
            version="0.1.0",
            description="测试用天气 agent",
            workflows=[schema],
        )
        agent = WorkflowAgent(config, agent_context=AgentContext())
        agent.bind_workflows([workflow])
        return agent

    # ===== 核心测试用例 =====
    @unittest.skip("skip system test")
    @pytest.mark.asyncio
    async def test_real_workflow_agent_invoke(self):
        """端到端测试：WorkflowAgent.invoke 走完整链路（插件被 mock）。"""
        # 1. 构造真实 workflow
        _, workflow = self._build_workflow()

        # 2. 构造 workflow agent 并调用
        agent = self._create_agent(workflow)
        result = await agent.invoke({"query": "查询上海的天气", "conversation_id": "c123"})

        # 5. 断言
        print(f"Workflow Agent输出的最终结果：{result}")
