# tests/test_workflow_agent_invoke_real.py
import os

os.environ["LLM_SSL_VERIFY"] = "false"
os.environ["RESTFUL_SSL_VERIFY"] = "false"

import asyncio
from datetime import datetime
import unittest
import pytest

from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.agent.workflow_agent.workflow_agent import WorkflowAgent
from jiuwen.core.runtime.wrapper import TaskRuntime
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.intent_detection_comp import IntentDetectionComponent, IntentDetectionCompConfig
from jiuwen.core.component.llm_comp import LLMComponent, LLMCompConfig
from jiuwen.core.component.questioner_comp import QuestionerComponent, QuestionerConfig, FieldInfo
from jiuwen.core.component.start_comp import Start
from jiuwen.core.component.tool_comp import ToolComponent, ToolComponentConfig
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.utils.llm.base import BaseModelInfo
from jiuwen.core.utils.tool.param import Param
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import WorkflowConfig, WorkflowMetadata
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.stream.base import OutputSchema
from typing import List

API_BASE = os.getenv("API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "siliconflow")
os.environ.setdefault("LLM_SSL_VERIFY", "false")

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
                timeout=120,  # 增加超时时间到120秒，避免网络问题
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
        config = IntentDetectionCompConfig(
            user_prompt="请判断用户意图",
            category_name_list=["查询某地天气"],
            model=model_config,
        )
        component = IntentDetectionComponent(config)
        component.add_branch("${intent.classification_id} == 0", ["end"], "默认分支")
        component.add_branch("${intent.classification_id} == 1", ["questioner"], "查询天气分支")
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
        )
        return QuestionerComponent(config)

    @staticmethod
    def _create_plugin_component() -> ToolComponent:
        """创建插件组件，真正调用外部 RESTful API。"""
        tool_config = ToolComponentConfig()
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
        return ToolComponent(tool_config).bind_tool(weather_tool)

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
            workflow_config=workflow_config
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
            inputs_schema={"input": "${start.query}"},
        )
        flow.add_workflow_comp(
            "llm",
            llm,
            inputs_schema={"query": "${start.query}"},
        )
        flow.add_workflow_comp(
            "questioner",
            questioner,
            inputs_schema={"query": "${llm.query}"}
        )
        flow.add_workflow_comp(
            "plugin",
            plugin,
            inputs_schema={
                "location": "${questioner.location}",
                "date": "${questioner.date}",
            },
        )
        flow.set_end_comp("end", end, inputs_schema={"output": "${plugin.data}"})

        # 4. 连接拓扑
        flow.add_connection("start", "intent")
        # flow.add_connection("intent", "llm")
        # flow.add_connection("intent", "end")
        flow.add_connection("llm", "questioner")
        flow.add_connection("questioner", "plugin")
        flow.add_connection("plugin", "end")

        return context.create_workflow_runtime(), flow

    def _build_interrupt_workflow(self) -> tuple[BaseRuntime, Workflow]:
        """
        构建包含交互式组件的工作流，用于测试中断恢复功能。

        返回 (context, workflow) 二元组，可直接用于 invoke。
        """
        # 1. 初始化工作流与上下文
        id = "test_interrupt_workflow"
        version = "1.0"
        name = "interrupt_test"
        workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                name=name,
                id=id,
                version=version,
            )
        )
        flow = Workflow(
            workflow_config=workflow_config
        )
        context = TaskRuntime(trace_id="test")

        # 2. 实例化各组件
        start = self._create_start_component()
        intent = self._create_intent_detection_component()
        questioner = self._create_questioner_component()
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
            inputs_schema={"query": "${start.query}"},
        )
        flow.add_workflow_comp(
            "questioner",
            questioner,
            inputs_schema={"query": "${start.query}"}
        )
        flow.set_end_comp("end", end, inputs_schema={"output": "${questioner.location}"})

        # 4. 连接拓扑
        flow.add_connection("start", "intent")
        # intent 组件通过分支路由自动连接到 questioner 或 end
        flow.add_connection("questioner", "end")

        return context.create_workflow_runtime(), flow

    def _build_prefixed_workflow(self, workflow_id: str, workflow_name: str, prefix: str) -> Workflow:
        workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                name=workflow_name,
                id=workflow_id,
                version="1.0",
            )
        )
        flow = Workflow(workflow_config=workflow_config)
        start = self._create_start_component()
        end = End({"responseTemplate": f"{prefix}{{{{output}}}}"})

        flow.set_start_comp(
            "start",
            start,
            inputs_schema={"query": "${query}"},
        )
        flow.set_end_comp(
            "end",
            end,
            inputs_schema={"output": "${start.query}"},
        )
        flow.add_connection("start", "end")
        return flow

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
        from jiuwen.agent.workflow_agent.workflow_agent import WorkflowAgent
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
        agent = WorkflowAgent(config)
        agent.bind_workflows([workflow])
        return agent

    @unittest.skip("skip system test ")
    async def test_multi_workflow_routing_via_intent_detection(self):
        """多工作流场景下，意图识别结果应跳转到目标工作流（使用真实模型）。"""
        print("=== 测试多工作流意图识别路由 ===")

        # 创建两个工作流的schema，描述要清晰明确
        weather_schema = WorkflowSchema(
            id="weather_flow",
            name="天气查询",
            description="查询某地的天气情况、温度、气象信息",
            version="1.0",
            inputs={"query": {"type": "string"}}
        )
        stock_schema = WorkflowSchema(
            id="stock_flow",
            name="股票查询",
            description="查询股票价格、股市行情、股票走势等金融信息",
            version="1.0",
            inputs={"query": {"type": "string"}}
        )

        weather_workflow = self._build_prefixed_workflow("weather_flow", "天气查询", "weather:")
        stock_workflow = self._build_prefixed_workflow("stock_flow", "股票查询", "stock:")

        config = WorkflowAgentConfig(
            id="test_multi_workflow_agent",
            version="0.1.0",
            description="多工作流意图识别测试",
            workflows=[weather_schema, stock_schema],
            model=self._create_model_config(),
        )

        agent = WorkflowAgent(config)
        agent.bind_workflows([weather_workflow, stock_workflow])

        # 使用真实模型调用，不使用任何mock（设置30秒超时）
        print("发送请求：查看上海股票走势")
        try:
            result = await asyncio.wait_for(
                agent.invoke({
                    "query": "查看上海股票走势",
                    "conversation_id": "conv-1"
                }),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print("❌ 调用超时！")
            raise

        print(f"返回结果：{result}")

        # 校验结果
        self.assertIsInstance(result, dict, "应该返回字典类型的结果")
        self.assertEqual(result["result_type"], "answer", "结果类型应该是answer")

        # 检查是否路由到了股票工作流（应该包含"stock:"前缀）
        response_content = result["output"].result["responseContent"]
        print(f"响应内容：{response_content}")

        self.assertIn("stock:", response_content, "应该路由到股票工作流")
        self.assertIn("股票", response_content, "响应应该包含查询内容")
        print(f"✅ 测试通过：成功路由到股票工作流，返回结果：{response_content}")

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

    def _test_interaction_detection(self, result, method_name):
        """检测交互请求的通用方法"""
        if isinstance(result, List) and isinstance(result[0], OutputSchema) and result[0].type == '__interaction__':
            print(f"✅ {method_name} 检测到交互请求!")
            return result
        return []

    def _create_interactive_input(self, interaction_outputs):
        """创建InteractiveInput的通用方法"""
        interactive_input = InteractiveInput()
        for item in interaction_outputs:
            component_id = item.payload.id
            interactive_input.update(component_id, "上海")
        return interactive_input

    @unittest.skip("skip system test - requires network")
    async def test_workflow_agent_invoke_with_interrupt_recovery(self):
        """端到端测试：WorkflowAgent.invoke 带中断恢复逻辑。"""
        print("=== 测试 WorkflowAgent.invoke 方法 ===")
        _, workflow = self._build_interrupt_workflow()
        agent = self._create_agent(workflow)

        # 第一次调用 - 应该触发中断（设置30秒超时）
        try:
            result = await asyncio.wait_for(
                agent.invoke({"query": "查询天气", "conversation_id": "c123"}),
                timeout=50.0
            )
        except asyncio.TimeoutError:
            print("❌ 第一次调用超时！")
            raise
        print(f"Workflow Agent第一次输出结果 >>> {result}")

        # 校验第一次调用结果：应该返回交互请求
        self.assertIsInstance(result, list, "第一次调用应该返回交互请求列表")
        self.assertEqual(result[0].type, '__interaction__', "应该返回交互类型")
        print(f"✅ 第一次调用校验通过：返回交互请求")

        interaction_outputs = self._test_interaction_detection(result, "invoke")
        if interaction_outputs:
            print("检测到交互请求，准备进行中断恢复...")
            # 注意：外部调用者只传入字符串，不需要手动创建 InteractiveInput
            # WorkflowMessageHandler 会根据中断状态自动封装

            # 第二次调用 - 传入字符串格式的回答，agent内部会自动处理中断恢复
            try:
                result2 = await asyncio.wait_for(
                    agent.invoke({"query": "上海", "conversation_id": "c123"}),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                print("❌ 第二次调用（恢复）超时！")
                raise
            print(f"Workflow Agent中断恢复后输出结果 >>> {result2}")

            # 校验第二次调用结果：应该返回完成状态
            self.assertIsInstance(result2, dict, "第二次调用应该返回字典")
            self.assertEqual(result2['result_type'], 'answer', "应该返回answer类型")
            self.assertEqual(result2['output'].state.value, 'COMPLETED', "工作流应该完成")
            self.assertEqual(result2['output'].result['responseContent'], '上海', "应该返回上海")
            print(f"✅ 第二次调用校验通过：工作流完成，返回结果正确")

            return result, result2  # 返回结果用于比对
        else:
            print("未检测到交互请求，测试可能未按预期执行")
            self.fail("应该检测到交互请求")

    @unittest.skip("skip system test - requires network")
    async def test_multi_workflow_jump_and_recovery(self):
        """
        测试多工作流间的跳转和恢复功能。
        
        场景：
        1. query1 -> workflow1（天气查询）-> 提问器中断（询问地点）
        2. query2 -> 意图识别 -> workflow2（股票查询）-> 提问器中断（询问股票代码）
        3. query3（InteractiveInput）-> 恢复 workflow1，提供地点信息 -> 完成
        4. query4（InteractiveInput）-> 恢复 workflow2，提供股票代码 -> 完成
        """
        print("=== 测试多工作流跳转和恢复 ===")

        # 创建两个带提问器的工作流
        weather_workflow = self._build_questioner_workflow(
            workflow_id="weather_flow",
            workflow_name="天气查询",
            question_field="location",
            question_desc="地点"
        )
        stock_workflow = self._build_questioner_workflow(
            workflow_id="stock_flow",
            workflow_name="股票查询",
            question_field="stock_code",
            question_desc="股票代码"
        )

        # 创建 schema
        weather_schema = WorkflowSchema(
            id="weather_flow",
            name="天气查询",
            description="查询某地的天气情况、温度、气象信息",
            version="1.0",
            inputs={"query": {"type": "string"}}
        )
        stock_schema = WorkflowSchema(
            id="stock_flow",
            name="股票查询",
            description="查询股票价格、股市行情、股票走势等金融信息",
            version="1.0",
            inputs={"query": {"type": "string"}}
        )

        # 创建 agent
        config = WorkflowAgentConfig(
            id="test_multi_workflow_jump_agent",
            version="0.1.0",
            description="多工作流跳转恢复测试",
            workflows=[weather_schema, stock_schema],
            model=self._create_model_config(),
        )
        agent = WorkflowAgent(config)
        agent.bind_workflows([weather_workflow, stock_workflow])

        conversation_id = "test-jump-recovery-001"

        # ========== 步骤1: query1 -> workflow1 -> 中断 ==========
        print("\n【步骤1】发送 query1: 查询天气")
        try:
            result1 = await asyncio.wait_for(
                agent.invoke({"query": "查询天气", "conversation_id": conversation_id}),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print("❌ 步骤1 超时！")
            raise

        print(f"步骤1 结果类型: {type(result1)}")
        if isinstance(result1, list):
            print(f"步骤1 返回列表，长度: {len(result1)}")
            if result1 and hasattr(result1[0], 'type'):
                print(f"步骤1 第一个元素类型: {result1[0].type}")

        # 校验：应该触发中断（提问器询问地点）
        self.assertIsInstance(result1, list, "步骤1应该返回交互请求列表")
        self.assertTrue(len(result1) > 0, "步骤1应该有交互请求")
        self.assertEqual(result1[0].type, '__interaction__', "步骤1应该返回交互类型")
        print(f"✅ 步骤1成功：workflow1 触发中断，询问地点")

        # 记录 workflow1 的中断信息
        workflow1_interaction = result1[0]
        workflow1_component_id = workflow1_interaction.payload.id
        print(f"   Workflow1 中断组件ID: {workflow1_component_id}")

        # ========== 步骤2: query2 -> workflow2 -> 中断 ==========
        print("\n【步骤2】发送 query2: 查看股票")
        try:
            result2 = await asyncio.wait_for(
                agent.invoke({"query": "查看股票", "conversation_id": conversation_id}),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print("❌ 步骤2 超时！")
            raise

        print(f"步骤2 结果类型: {type(result2)}")
        if isinstance(result2, list):
            print(f"步骤2 返回列表，长度: {len(result2)}")
            if result2 and hasattr(result2[0], 'type'):
                print(f"步骤2 第一个元素类型: {result2[0].type}")

        # 校验：应该触发中断（提问器询问股票代码）
        self.assertIsInstance(result2, list, "步骤2应该返回交互请求列表")
        self.assertTrue(len(result2) > 0, "步骤2应该有交互请求")
        self.assertEqual(result2[0].type, '__interaction__', "步骤2应该返回交互类型")
        print(f"✅ 步骤2成功：workflow2 触发中断，询问股票代码")

        # 记录 workflow2 的中断信息
        workflow2_interaction = result2[0]
        workflow2_component_id = workflow2_interaction.payload.id
        print(f"   Workflow2 中断组件ID: {workflow2_component_id}")

        # ========== 步骤3: query3 -> 恢复 workflow1 ==========
        print("\n【步骤3】发送 query3: 提供地点信息，恢复 workflow1")
        try:
            result3 = await asyncio.wait_for(
                agent.invoke({"query": "查询北京天气", "conversation_id": conversation_id}),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print("❌ 步骤3 超时！")
            raise

        print(f"步骤3 结果: {result3}")

        # 校验：workflow1 应该完成
        self.assertIsInstance(result3, dict, "步骤3应该返回字典")
        self.assertEqual(result3['result_type'], 'answer', "步骤3应该返回answer类型")
        self.assertEqual(result3['output'].state.value, 'COMPLETED', "步骤3 workflow1应该完成")
        response_content_3 = result3['output'].result.get('responseContent', '')
        print(f"✅ 步骤3成功：workflow1 恢复并完成，返回: {response_content_3}")

        # ========== 步骤4: query4 -> 恢复 workflow2 ==========
        print("\n【步骤4】发送 query4: 提供股票代码，恢复 workflow2")
        try:
            result4 = await asyncio.wait_for(
                agent.invoke({"query": "查看AAPL股票", "conversation_id": conversation_id}),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print("❌ 步骤4 超时！")
            raise

        print(f"步骤4 结果: {result4}")

        # 校验：workflow2 应该完成
        self.assertIsInstance(result4, dict, "步骤4应该返回字典")
        self.assertEqual(result4['result_type'], 'answer', "步骤4应该返回answer类型")
        self.assertEqual(result4['output'].state.value, 'COMPLETED', "步骤4 workflow2应该完成")
        response_content_4 = result4['output'].result.get('responseContent', '')
        print(f"✅ 步骤4成功：workflow2 恢复并完成，返回: {response_content_4}")

        print("\n🎉 所有步骤完成！多工作流跳转和恢复测试通过！")

    def _build_questioner_workflow(self, workflow_id: str, workflow_name: str,
                                   question_field: str, question_desc: str) -> Workflow:
        """
        构建包含提问器的简单工作流。
        
        Args:
            workflow_id: 工作流ID
            workflow_name: 工作流名称
            question_field: 提问字段名
            question_desc: 提问字段描述
        
        Returns:
            Workflow: 包含 start -> questioner -> end 的工作流
        """
        workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                name=workflow_name,
                id=workflow_id,
                version="1.0",
            )
        )
        flow = Workflow(workflow_config=workflow_config)

        # 创建组件
        start = self._create_start_component()

        # 创建提问器
        key_fields = [
            FieldInfo(field_name=question_field, description=question_desc, required=True),
        ]
        model_config = self._create_model_config()
        questioner_config = QuestionerConfig(
            model=model_config,
            question_content="",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=False,
            extra_prompt_for_fields_extraction="",
            example_content="",
        )
        questioner = QuestionerComponent(questioner_config)

        # End 组件，返回提问器收集的字段值
        end = End({"responseTemplate": f"{{{{{question_field}}}}}"})

        # 注册组件
        flow.set_start_comp("start", start, inputs_schema={"query": "${query}"})
        flow.add_workflow_comp("questioner", questioner, inputs_schema={"query": "${start.query}"})
        flow.set_end_comp("end", end, inputs_schema={question_field: f"${{questioner.{question_field}}}"})

        # 连接拓扑
        flow.add_connection("start", "questioner")
        flow.add_connection("questioner", "end")

        return flow

    @unittest.skip("skip system test - requires network")
    async def test_workflow_agent_stream_with_interrupt_recovery(self):
        """端到端测试：WorkflowAgent.stream 带中断恢复逻辑。"""
        print("=== 测试 WorkflowAgent.stream 方法 ===")
        _, workflow = self._build_interrupt_workflow()
        agent = self._create_agent(workflow)

        # 第一次调用 - 应该触发中断（设置50秒超时）
        interaction_outputs = []
        first_chunks = []
        try:
            async def collect_first_stream():
                chunks = []
                async for chunk in agent.stream({"query": "查询天气", "conversation_id": "c123"}):
                    print(f"Workflow Agent第一次输出结果 >>> {chunk}")
                    chunks.append(chunk)
                    if isinstance(chunk, OutputSchema) and chunk.type == "__interaction__":
                        print("✅ stream 检测到交互请求!")
                        interaction_outputs.append(chunk)
                return chunks

            first_chunks = await asyncio.wait_for(collect_first_stream(), timeout=50.0)
        except asyncio.TimeoutError:
            print("❌ 第一次调用超时！")
            raise

        # 校验第一次调用结果：应该包含交互请求
        self.assertTrue(len(interaction_outputs) > 0, "第一次调用应该包含交互请求")
        self.assertEqual(interaction_outputs[0].type, '__interaction__', "应该返回交互类型")
        print(f"✅ 第一次调用校验通过：返回 {len(interaction_outputs)} 个交互请求")

        if interaction_outputs:
            print("检测到交互请求，准备进行中断恢复...")
            interactive_input = self._create_interactive_input(interaction_outputs)

            # 第二次调用 - 使用InteractiveInput进行恢复（设置30秒超时）
            second_chunks = []
            workflow_final_chunk = None
            try:
                async def collect_second_stream():
                    chunks = []
                    async for chunk in agent.stream({"query": interactive_input, "conversation_id": "c123"}):
                        print(f"Workflow Agent中断恢复后输出结果 >>> {chunk}")
                        chunks.append(chunk)
                    return chunks

                second_chunks = await asyncio.wait_for(collect_second_stream(), timeout=30.0)
            except asyncio.TimeoutError:
                print("❌ 第二次调用（恢复）超时！")
                raise

            # 校验第二次调用结果：应该包含 workflow_final
            for chunk in second_chunks:
                if isinstance(chunk, OutputSchema) and chunk.type == "workflow_final":
                    workflow_final_chunk = chunk
                    break

            self.assertIsNotNone(workflow_final_chunk, "第二次调用应该包含 workflow_final 结果")
            self.assertIsInstance(workflow_final_chunk.payload, dict, "workflow_final payload 应该是字典")

            # 检查是否是错误响应
            if workflow_final_chunk.payload.get('error'):
                error_msg = workflow_final_chunk.payload.get('message', 'Unknown error')
                print(f"⚠️ 工作流执行遇到错误: {error_msg}")
                # 如果是 LLM 调用错误，这是外部依赖问题，测试跳过
                if 'invoke llm error' in error_msg or 'Failed to invoke llm' in error_msg:
                    self.skipTest(f"LLM 调用失败（外部依赖问题）: {error_msg}")
                else:
                    self.fail(f"工作流执行失败: {error_msg}")

            # 校验正常响应
            self.assertEqual(workflow_final_chunk.payload['result_type'], 'answer', "应该返回answer类型")

            # 校验工作流完成状态
            output = workflow_final_chunk.payload['output']
            self.assertEqual(output.state.value, 'COMPLETED', "工作流应该完成")
            self.assertEqual(output.result['responseContent'], '上海', "应该返回上海")
            print(f"✅ 第二次调用校验通过：工作流完成，返回结果正确")

            return first_chunks, second_chunks  # 返回结果用于比对
        else:
            print("未检测到交互请求，测试可能未按预期执行")
            self.fail("应该检测到交互请求")
