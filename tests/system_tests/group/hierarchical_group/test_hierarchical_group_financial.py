#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
HierarchicalGroup 金融场景测试 - 使用 HierarchicalMainController + WorkflowAgent

场景：
- 1个主 agent（使用 HierarchicalMainController 进行意图识别和任务分发）
- 3个子 workflow agent：转账、查余额、理财
- 每个 workflow 都有 QuestionerComponent 中断节点
"""

import os

os.environ["LLM_SSL_VERIFY"] = "false"
os.environ["RESTFUL_SSL_VERIFY"] = "false"

import asyncio
import unittest

from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.agent.config.workflow_config import WorkflowAgentConfig
from openjiuwen.agent.workflow_agent.workflow_agent import WorkflowAgent
from openjiuwen.agent_group.hierarchical_group import (
    HierarchicalGroup,
    HierarchicalGroupConfig
)
from openjiuwen.agent_group.hierarchical_group.agents.main_controller import (
    HierarchicalMainController
)
from openjiuwen.core.agent.agent import ControllerAgent
from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.component.end_comp import End
from openjiuwen.core.component.questioner_comp import (
    FieldInfo,
    QuestionerComponent,
    QuestionerConfig
)
from openjiuwen.core.component.start_comp import Start
from openjiuwen.core.common.constants import constant as const
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.workflow.base import Workflow
from openjiuwen.core.workflow.workflow_config import WorkflowConfig, WorkflowMetadata

# 模型配置
API_BASE = os.getenv("API_BASE", "mock://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", "sk-fake")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")
os.environ.setdefault("LLM_SSL_VERIFY", "false")


class TestHierarchicalGroupFinancial(unittest.IsolatedAsyncioTestCase):
    """金融场景测试 - HierarchicalGroup + HierarchicalMainController + WorkflowAgent"""

    async def asyncSetUp(self):
        await Runner.start()

    async def asyncTearDown(self):
        await Runner.stop()

    @staticmethod
    def _create_model_config() -> ModelConfig:
        """创建模型配置"""
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
        """创建 Start 组件"""
        return Start({
            "inputs": [
                {
                    "id": "query",
                    "type": "String",
                    "required": "true",
                    "sourceType": "ref"
                }
            ]
        })

    def _build_financial_workflow(
            self,
            workflow_id: str,
            workflow_name: str,
            workflow_desc: str,
            field_name: str,
            field_desc: str
    ) -> Workflow:
        """
        构建金融业务工作流（带中断节点）

        Args:
            workflow_id: 工作流ID
            workflow_name: 工作流名称
            workflow_desc: 工作流描述
            field_name: 提问字段名
            field_desc: 提问字段描述

        Returns:
            Workflow: 包含 start -> questioner -> end 的工作流
        """
        workflow_config = WorkflowConfig(
            metadata=WorkflowMetadata(
                name=workflow_name,
                id=workflow_id,
                version="1.0",
                description=workflow_desc,
            )
        )
        flow = Workflow(workflow_config=workflow_config)

        # 创建组件
        start = self._create_start_component()

        # 创建提问器（中断节点）
        key_fields = [
            FieldInfo(
                field_name=field_name,
                description=field_desc,
                required=True
            ),
        ]
        model_config = self._create_model_config()
        questioner_config = QuestionerConfig(
            model=model_config,
            question_content="",
            extract_fields_from_response=True,
            field_names=key_fields,
            with_chat_history=False,
        )
        questioner = QuestionerComponent(questioner_config)

        # End 组件
        end = End({"responseTemplate": f"{workflow_name}完成: {{{{{field_name}}}}}"})

        # 注册组件
        flow.set_start_comp("start", start, inputs_schema={"query": "${query}"})
        flow.add_workflow_comp(
            "questioner", questioner, inputs_schema={"query": "${start.query}"}
        )
        flow.set_end_comp(
            "end", end, inputs_schema={field_name: f"${{questioner.{field_name}}}"}
        )

        # 连接拓扑: start -> questioner -> end
        flow.add_connection("start", "questioner")
        flow.add_connection("questioner", "end")

        return flow

    def _create_workflow_agent(
            self,
            agent_id: str,
            description: str,
            workflow: Workflow
    ) -> WorkflowAgent:
        """创建 WorkflowAgent"""
        config = WorkflowAgentConfig(
            id=agent_id,
            version="1.0",
            description=description,
            workflows=[],
            model=self._create_model_config(),
        )
        agent = WorkflowAgent(config)
        agent.add_workflows([workflow])
        return agent

    @unittest.skip("skip system test - requires network")
    async def test_financial_workflow_with_interrupt_invoke(self):
        """
        金融场景完整用例：HierarchicalGroup + 工作流中断恢复

        测试流程：
        1. 创建 HierarchicalGroup，主 agent 使用 HierarchicalMainController
        2. 添加 3 个金融 WorkflowAgent（每个都有中断节点）
        3. 发送转账请求 -> 路由到转账 agent -> 触发中断（询问金额）
        4. 提供金额 -> 恢复工作流 -> 完成
        """
        print("\n=== 金融场景 HierarchicalGroup 测试 ===")

        # 1. 创建金融业务工作流
        transfer_workflow = self._build_financial_workflow(
            workflow_id="transfer_flow",
            workflow_name="转账服务",
            workflow_desc="处理用户转账请求，支持转账到指定账户",
            field_name="amount",
            field_desc="转账金额（数字）"
        )

        balance_workflow = self._build_financial_workflow(
            workflow_id="balance_flow",
            workflow_name="余额查询",
            workflow_desc="查询用户账户余额信息",
            field_name="account",
            field_desc="账户号码"
        )

        invest_workflow = self._build_financial_workflow(
            workflow_id="invest_flow",
            workflow_name="理财服务",
            workflow_desc="提供理财产品推荐和购买服务",
            field_name="product",
            field_desc="理财产品名称"
        )

        # 2. 创建 WorkflowAgent
        transfer_agent = self._create_workflow_agent(
            agent_id="transfer_agent",
            description="转账服务，处理用户的转账请求",
            workflow=transfer_workflow
        )

        balance_agent = self._create_workflow_agent(
            agent_id="balance_agent",
            description="余额查询服务，查询用户账户余额",
            workflow=balance_workflow
        )

        invest_agent = self._create_workflow_agent(
            agent_id="invest_agent",
            description="理财服务，提供理财产品推荐和购买",
            workflow=invest_workflow
        )

        # 3. 创建 HierarchicalGroup
        config = HierarchicalGroupConfig(
            group_id="financial_group",
            leader_agent_id="main_controller"
        )
        hierarchical_group = HierarchicalGroup(config)

        # 4. 创建主 agent（HierarchicalMainController）
        main_config = AgentConfig(
            id="main_controller",
            description="金融服务主控制器，识别用户意图并分发任务"
        )
        main_controller = HierarchicalMainController()
        main_agent = ControllerAgent(main_config, controller=main_controller)

        # 5. 添加所有 agent 到 group
        hierarchical_group.add_agent("main_controller", main_agent)
        hierarchical_group.add_agent("transfer_agent", transfer_agent)
        hierarchical_group.add_agent("balance_agent", balance_agent)
        hierarchical_group.add_agent("invest_agent", invest_agent)

        conversation_id = "financial_test_001"

        # ========== 步骤1: 发送转账请求 -> 中断 ==========
        # 不指定 receiver_id，让消息自动路由到 leader
        # Leader (HierarchicalMainController) 会通过 LLM 意图识别找到目标 agent
        print("\n【步骤1】发送转账请求")
        message1 = Message.create_user_message(
            content="我要转账",
            conversation_id=conversation_id
        )
        # 不设置 receiver_id，消息会自动路由到 leader，由 leader 做意图识别

        try:
            result1 = await asyncio.wait_for(
                hierarchical_group.invoke(message1),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            print("❌ 步骤1 超时！")
            raise

        print(f"步骤1 结果类型: {type(result1)}")

        # 校验：应该触发中断
        self.assertIsInstance(result1, list, "步骤1应该返回交互请求列表")
        self.assertTrue(len(result1) > 0, "步骤1应该有交互请求")
        self.assertEqual(
            result1[0].type, const.INTERACTION, "步骤1应该返回交互类型"
        )
        print(f"✅ 步骤1成功：转账工作流触发中断，询问金额")

        # ========== 步骤2: 提供金额 -> 恢复 -> 完成 ==========
        # 不指定 receiver_id，leader 会自动检测到有中断的 agent 并恢复
        print("\n【步骤2】提供转账金额")
        message2 = Message.create_user_message(
            content="100元",
            conversation_id=conversation_id
        )
        # 不设置 receiver_id，leader 会通过 _get_last_interrupted_agent 恢复到中断的 agent

        try:
            result2 = await asyncio.wait_for(
                hierarchical_group.invoke(message2),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            print("❌ 步骤2 超时！")
            raise

        print(f"步骤2 结果: {result2}")

        # 校验：工作流应该完成
        self.assertIsInstance(result2, dict, "步骤2应该返回字典")
        self.assertEqual(
            result2['result_type'], 'answer', "步骤2应该返回answer类型"
        )
        self.assertEqual(
            result2['output'].state.value, 'COMPLETED', "步骤2工作流应该完成"
        )
        response_content = result2['output'].result.get('responseContent', '')
        print(f"✅ 步骤2成功：转账工作流完成，返回: {response_content}")

        print("\n🎉 金融场景测试完成！")

    @unittest.skip("skip system test - requires network")
    async def test_financial_workflow_with_interrupt_stream(self):
        """
        金融场景 Stream 用例：HierarchicalGroup.stream + 工作流中断恢复

        测试流程：
        1. 创建 HierarchicalGroup，主 agent 使用 HierarchicalMainController
        2. 添加 3 个金融 WorkflowAgent（每个都有中断节点）
        3. 使用 stream() 发送转账请求 -> 触发中断
        4. 使用 stream() 提供金额 -> 恢复工作流 -> 完成
        """
        print("\n=== 金融场景 HierarchicalGroup Stream 测试 ===")

        # 1. 创建金融业务工作流
        transfer_workflow = self._build_financial_workflow(
            workflow_id="transfer_flow_stream",
            workflow_name="转账服务",
            workflow_desc="处理用户转账请求，支持转账到指定账户",
            field_name="amount",
            field_desc="转账金额（数字）"
        )

        balance_workflow = self._build_financial_workflow(
            workflow_id="balance_flow_stream",
            workflow_name="余额查询",
            workflow_desc="查询用户账户余额信息",
            field_name="account",
            field_desc="账户号码"
        )

        invest_workflow = self._build_financial_workflow(
            workflow_id="invest_flow_stream",
            workflow_name="理财服务",
            workflow_desc="提供理财产品推荐和购买服务",
            field_name="product",
            field_desc="理财产品名称"
        )

        # 2. 创建 WorkflowAgent
        transfer_agent = self._create_workflow_agent(
            agent_id="transfer_agent",
            description="转账服务，处理用户的转账请求",
            workflow=transfer_workflow
        )

        balance_agent = self._create_workflow_agent(
            agent_id="balance_agent",
            description="余额查询服务，查询用户账户余额",
            workflow=balance_workflow
        )

        invest_agent = self._create_workflow_agent(
            agent_id="invest_agent",
            description="理财服务，提供理财产品推荐和购买",
            workflow=invest_workflow
        )

        # 3. 创建 HierarchicalGroup
        config = HierarchicalGroupConfig(
            group_id="financial_group_stream",
            leader_agent_id="main_controller"
        )
        hierarchical_group = HierarchicalGroup(config)

        # 4. 创建主 agent（HierarchicalMainController）
        main_config = AgentConfig(
            id="main_controller",
            description="金融服务主控制器，识别用户意图并分发任务"
        )
        main_controller = HierarchicalMainController()
        main_agent = ControllerAgent(main_config, controller=main_controller)

        # 5. 添加所有 agent 到 group
        hierarchical_group.add_agent("main_controller", main_agent)
        hierarchical_group.add_agent("transfer_agent", transfer_agent)
        hierarchical_group.add_agent("balance_agent", balance_agent)
        hierarchical_group.add_agent("invest_agent", invest_agent)

        conversation_id = "financial_stream_test_001"

        # ========== 步骤1: 使用 stream 发送转账请求 -> 中断 ==========
        # 不指定 receiver_id，让消息自动路由到 leader
        print("\n【步骤1】使用 stream 发送转账请求")
        message1 = Message.create_user_message(
            content="我要转账",
            conversation_id=conversation_id
        )
        # 不设置 receiver_id，消息会自动路由到 leader，由 leader 做意图识别

        # 收集流式输出
        chunks1 = []
        try:
            async def collect_stream1():
                async for chunk in hierarchical_group.stream(message1):
                    print(f"  Stream chunk: {chunk.type}")
                    chunks1.append(chunk)

            await asyncio.wait_for(collect_stream1(), timeout=120.0)
        except asyncio.TimeoutError:
            print("❌ 步骤1 超时！")
            raise

        print(f"步骤1 收到 {len(chunks1)} 个 chunks")

        # 校验：应该触发中断
        self.assertTrue(len(chunks1) > 0, "步骤1应该有流式输出")
        final_chunk1 = chunks1[-1]
        # 交互请求会直接透传 INTERACTION 类型
        self.assertEqual(
            final_chunk1.type, const.INTERACTION, "步骤1应该返回交互类型"
        )
        print(f"✅ 步骤1成功：转账工作流触发中断，询问金额")

        # ========== 步骤2: 使用 stream 提供金额 -> 恢复 -> 完成 ==========
        # 不指定 receiver_id，leader 会自动检测到有中断的 agent 并恢复
        print("\n【步骤2】使用 stream 提供转账金额")
        message2 = Message.create_user_message(
            content="200元",
            conversation_id=conversation_id
        )
        # 不设置 receiver_id，leader 会通过 _get_last_interrupted_agent 恢复到中断的 agent

        # 收集流式输出
        chunks2 = []
        try:
            async def collect_stream2():
                async for chunk in hierarchical_group.stream(message2):
                    print(f"  Stream chunk: {chunk.type}")
                    chunks2.append(chunk)

            await asyncio.wait_for(collect_stream2(), timeout=120.0)
        except asyncio.TimeoutError:
            print("❌ 步骤2 超时！")
            raise

        print(f"步骤2 收到 {len(chunks2)} 个 chunks")

        # 校验：工作流应该完成
        self.assertTrue(len(chunks2) > 0, "步骤2应该有流式输出")

        # 找到最终结果 chunk（workflow_final 类型）
        final_chunk2 = None
        for chunk in chunks2:
            if chunk.type == 'workflow_final':
                final_chunk2 = chunk
                break

        self.assertIsNotNone(final_chunk2, "步骤2应该有 workflow_final chunk")

        payload2 = final_chunk2.payload
        response_content = payload2.get('responseContent', '')
        self.assertIn('200', response_content, "步骤2应该包含转账金额")
        print(f"✅ 步骤2成功：转账工作流完成，返回: {response_content}")

        print("\n🎉 金融场景 Stream 测试完成！")

    @unittest.skip("skip system test - requires network")
    async def test_multi_agent_jump_and_recovery_stream(self):
        """
        多子Agent跳转恢复测试：HierarchicalGroup.stream + 多Agent中断跳转恢复

        场景：
        1. query1 "我要转账" -> 路由到 transfer_agent -> 中断（询问金额）
        2. query2 "我想理财" -> 路由到 invest_agent -> 中断（询问产品）
        3. query3 "100元" -> 恢复 transfer_agent -> 完成转账
        4. query4 "稳健型产品" -> 恢复 invest_agent -> 完成理财

        验证：
        - 多个子Agent可以同时处于中断状态
        - 系统能正确识别用户意图并恢复对应的Agent
        - 每个Agent的中断状态独立维护
        """
        print("\n=== 多子Agent跳转恢复测试 ===")

        # 1. 创建金融业务工作流
        transfer_workflow = self._build_financial_workflow(
            workflow_id="transfer_flow_multi",
            workflow_name="转账服务",
            workflow_desc="处理用户转账请求，支持转账到指定账户",
            field_name="amount",
            field_desc="转账金额（数字）"
        )

        balance_workflow = self._build_financial_workflow(
            workflow_id="balance_flow_multi",
            workflow_name="余额查询",
            workflow_desc="查询用户账户余额信息",
            field_name="account",
            field_desc="账户号码"
        )

        invest_workflow = self._build_financial_workflow(
            workflow_id="invest_flow_multi",
            workflow_name="理财服务",
            workflow_desc="提供理财产品推荐和购买服务",
            field_name="product",
            field_desc="理财产品名称"
        )

        # 2. 创建 WorkflowAgent
        transfer_agent = self._create_workflow_agent(
            agent_id="transfer_agent",
            description="转账服务，处理用户的转账请求",
            workflow=transfer_workflow
        )

        balance_agent = self._create_workflow_agent(
            agent_id="balance_agent",
            description="余额查询服务，查询用户账户余额",
            workflow=balance_workflow
        )

        invest_agent = self._create_workflow_agent(
            agent_id="invest_agent",
            description="理财服务，提供理财产品推荐和购买",
            workflow=invest_workflow
        )

        # 3. 创建 HierarchicalGroup
        config = HierarchicalGroupConfig(
            group_id="financial_group_multi",
            leader_agent_id="main_controller"
        )
        hierarchical_group = HierarchicalGroup(config)

        # 4. 创建主 agent（HierarchicalMainController）
        main_config = AgentConfig(
            id="main_controller",
            description="金融服务主控制器，识别用户意图并分发任务",
            model=self._create_model_config()
        )
        main_controller = HierarchicalMainController()
        main_agent = ControllerAgent(main_config, controller=main_controller)

        # 5. 添加所有 agent 到 group
        hierarchical_group.add_agent("main_controller", main_agent)
        hierarchical_group.add_agent("transfer_agent", transfer_agent)
        hierarchical_group.add_agent("balance_agent", balance_agent)
        hierarchical_group.add_agent("invest_agent", invest_agent)

        conversation_id = "financial_multi_agent_jump_test_001"

        # ========== 步骤1: 发送转账请求 -> transfer_agent 中断 ==========
        print("\n【步骤1】发送转账请求 -> transfer_agent 中断")
        message1 = Message.create_user_message(
            content="我要转账",
            conversation_id=conversation_id
        )

        chunks1 = []
        try:
            async def collect_stream1():
                async for chunk in hierarchical_group.stream(message1):
                    print(f"  Stream chunk: {chunk.type}")
                    chunks1.append(chunk)

            await asyncio.wait_for(collect_stream1(), timeout=120.0)
        except asyncio.TimeoutError:
            print("❌ 步骤1 超时！")
            raise

        print(f"步骤1 收到 {len(chunks1)} 个 chunks")

        # 校验：transfer_agent 触发中断
        self.assertTrue(len(chunks1) > 0, "步骤1应该有流式输出")
        final_chunk1 = chunks1[-1]
        self.assertEqual(
            final_chunk1.type, const.INTERACTION, "步骤1应该返回交互类型"
        )
        print(f"✅ 步骤1成功：transfer_agent 触发中断，询问金额")

        # ========== 步骤2: 发送理财请求 -> invest_agent 中断 ==========
        print("\n【步骤2】发送理财请求 -> invest_agent 中断（跳转到新Agent）")
        message2 = Message.create_user_message(
            content="我想理财",
            conversation_id=conversation_id
        )

        chunks2 = []
        try:
            async def collect_stream2():
                async for chunk in hierarchical_group.stream(message2):
                    print(f"  Stream chunk: {chunk.type}")
                    chunks2.append(chunk)

            await asyncio.wait_for(collect_stream2(), timeout=120.0)
        except asyncio.TimeoutError:
            print("❌ 步骤2 超时！")
            raise

        print(f"步骤2 收到 {len(chunks2)} 个 chunks")

        # 校验：invest_agent 触发中断
        self.assertTrue(len(chunks2) > 0, "步骤2应该有流式输出")
        final_chunk2 = chunks2[-1]
        self.assertEqual(
            final_chunk2.type, const.INTERACTION, "步骤2应该返回交互类型"
        )
        print(f"✅ 步骤2成功：invest_agent 触发中断，询问产品")

        # ========== 步骤3: 提供金额 -> 恢复 transfer_agent -> 完成 ==========
        print("\n【步骤3】提供金额 -> 恢复 transfer_agent -> 完成转账")
        message3 = Message.create_user_message(
            content="我要转账100元",
            conversation_id=conversation_id
        )

        chunks3 = []
        try:
            async def collect_stream3():
                async for chunk in hierarchical_group.stream(message3):
                    print(f"  Stream chunk: {chunk.type}")
                    chunks3.append(chunk)

            await asyncio.wait_for(collect_stream3(), timeout=120.0)
        except asyncio.TimeoutError:
            print("❌ 步骤3 超时！")
            raise

        print(f"步骤3 收到 {len(chunks3)} 个 chunks")

        # 校验：transfer_agent 恢复并完成
        self.assertTrue(len(chunks3) > 0, "步骤3应该有流式输出")

        # 找到最终结果 chunk
        final_chunk3 = None
        for chunk in chunks3:
            if chunk.type == 'workflow_final':
                final_chunk3 = chunk
                break

        self.assertIsNotNone(final_chunk3, "步骤3应该有 workflow_final chunk")
        payload3 = final_chunk3.payload
        response_content3 = payload3.get('responseContent', '')
        self.assertIn('100', response_content3, "步骤3应该包含转账金额")
        print(f"✅ 步骤3成功：transfer_agent 恢复并完成，返回: {response_content3}")

        # ========== 步骤4: 提供产品 -> 恢复 invest_agent -> 完成 ==========
        print("\n【步骤4】提供产品 -> 恢复 invest_agent -> 完成理财")
        message4 = Message.create_user_message(
            content="我要购买稳健型理财产品",
            conversation_id=conversation_id
        )

        chunks4 = []
        try:
            async def collect_stream4():
                async for chunk in hierarchical_group.stream(message4):
                    print(f"  Stream chunk: {chunk.type}")
                    chunks4.append(chunk)

            await asyncio.wait_for(collect_stream4(), timeout=120.0)
        except asyncio.TimeoutError:
            print("❌ 步骤4 超时！")
            raise

        print(f"步骤4 收到 {len(chunks4)} 个 chunks")

        # 校验：invest_agent 恢复并完成
        self.assertTrue(len(chunks4) > 0, "步骤4应该有流式输出")

        # 找到最终结果 chunk
        final_chunk4 = None
        for chunk in chunks4:
            if chunk.type == 'workflow_final':
                final_chunk4 = chunk
                break

        self.assertIsNotNone(final_chunk4, "步骤4应该有 workflow_final chunk")
        payload4 = final_chunk4.payload
        response_content4 = payload4.get('responseContent', '')
        self.assertIn('稳健', response_content4, "步骤4应该包含理财产品名称")
        print(f"✅ 步骤4成功：invest_agent 恢复并完成，返回: {response_content4}")

        print("\n🎉 多子Agent跳转恢复测试完成！")
        print("   - transfer_agent: 中断 -> 跳转 -> 恢复 -> 完成")
        print("   - invest_agent: 中断 -> 恢复 -> 完成")


if __name__ == "__main__":
    unittest.main()
