# tests/test_multi_workflow_agent.py
"""
多工作流测试套件
测试场景：
1. 多工作流意图识别路由
2. 多工作流跳转和恢复
"""
import os

from jiuwen.core.runner.runner import Runner

os.environ["LLM_SSL_VERIFY"] = "false"
os.environ["RESTFUL_SSL_VERIFY"] = "false"

import asyncio
from datetime import datetime
import unittest

from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.workflow_config import WorkflowAgentConfig
from jiuwen.agent.workflow_agent.workflow_agent import WorkflowAgent
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.questioner_comp import QuestionerComponent, QuestionerConfig, FieldInfo
from jiuwen.core.component.start_comp import Start
from jiuwen.core.utils.llm.base import BaseModelInfo
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.workflow.workflow_config import WorkflowConfig, WorkflowMetadata

API_BASE = os.getenv("API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")
os.environ.setdefault("LLM_SSL_VERIFY", "false")

SYSTEM_PROMPT_TEMPLATE = "你是一个query改写的AI助手。今天的日期是{}。"


def build_current_date():
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d")


class MultiWorkflowAgentTest(unittest.IsolatedAsyncioTestCase):
    """专门用于测试多工作流场景的类。"""

    async def asyncSetUp(self):
        await Runner.start()

    async def asyncTearDown(self):
        await Runner.stop()

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
                timeout=120,
            ),
        )

    @staticmethod
    def _create_start_component():
        return Start({"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]})

    def _build_prefixed_workflow(self, workflow_id: str, workflow_name: str, prefix: str) -> Workflow:
        """
        构建简单的工作流，输出带指定前缀的结果。
        用于测试工作流路由。
        """
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

    @unittest.skip("skip system test")
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

    @unittest.skip
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
                timeout=50.0
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
                timeout=50.0
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
                timeout=50.0
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
                timeout=50.0
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

