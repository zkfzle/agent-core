#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
ReActAgent - 极简版 ReAct Agent（无中断、无Controller）
"""

import json
import asyncio
from typing import Dict, Any, AsyncIterator, List
from openjiuwen.core.agent.agent import BaseAgent
from openjiuwen.agent.config.react_config import ReActAgentConfig
from openjiuwen.agent.common.schema import WorkflowSchema, PluginSchema
from openjiuwen.core.runtime.runtime import Runtime, Workflow
from openjiuwen.core.stream.base import OutputSchema
from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.messages import AIMessage, ToolMessage
from openjiuwen.agent.utils import MessageUtils


class ReActAgent(BaseAgent):
    """ReAct Agent - 极简实现（无中断、无Controller）
    """

    def __init__(
            self,
            agent_config: ReActAgentConfig,
            workflows: List[Workflow] = None,
            tools: List[Tool] = None
    ):
        """初始化 ReActAgent
        
        Args:
            agent_config: ReAct 配置
            workflows: 工作流列表
            tools: 工具列表
        """
        # 调用父类初始化（BaseAgent 会创建 runtime, context_engine 等）
        super().__init__(agent_config)

        # LLM 实例（延迟创建）
        self._llm = None
        
        # 通过 BaseAgent 的接口添加 tools 和 workflows（自动同步）
        if tools:
            self.add_tools(tools)
        if workflows:
            self.add_workflows(workflows)

    def _get_llm(self):
        """获取 LLM 实例"""
        if self._llm is None:
            self._llm = ModelFactory().get_model(
                model_provider=self._agent_config.model.model_provider,
                **self._agent_config.model.model_info.model_dump(exclude=['model_name', 'streaming'])
            )
        return self._llm

    async def call_model(self, user_input: str, runtime: Runtime, is_first_call: bool = False):
        """调用 LLM 进行推理
        
        Args:
            user_input: 用户输入或工具结果
            runtime: Runtime 实例
            is_first_call: 是否是第一次调用（第一次需要添加用户消息）
        
        Returns:
            llm_output: LLM 的输出（包含 content 和 tool_calls）
        """
        # 1. 如果是第一次调用，添加用户消息
        if is_first_call:
            MessageUtils.add_user_message(user_input, self.context_engine, runtime)

        # 2. 获取对话历史
        chat_history = MessageUtils.get_chat_history(
            self.context_engine, runtime, self._agent_config
        )

        # 3. 格式化 prompt
        messages = []
        # 添加系统提示
        for prompt in self._agent_config.prompt_template:
            messages.append(prompt)

        # 添加对话历史（需要完整转换 BaseMessage 对象）
        for msg in chat_history:
            # 使用 model_dump 完整导出消息，排除 None 值
            msg_dict = msg.model_dump(exclude_none=True)
            messages.append(msg_dict)

        # 4. 获取可用工具信息
        tools = runtime.get_tool_info()

        # 5. 调用 LLM
        llm = self._get_llm()
        llm_output = await llm.ainvoke(
            self._agent_config.model.model_info.model_name,
            messages,
            tools
        )

        # 6. 保存 AI 响应到对话历史
        ai_message = AIMessage(
            content=llm_output.content,
            tool_calls=llm_output.tool_calls
        )
        MessageUtils.add_ai_message(ai_message, self.context_engine, runtime)

        return llm_output

    async def _execute_tool_call(self, tool_call, runtime: Runtime) -> Any:
        """执行单个工具调用
        
        Args:
            tool_call: LLM 返回的 tool_call 对象
            runtime: Runtime 实例
        
        Returns:
            工具执行结果
        """
        # 解析工具名称和参数
        tool_name = tool_call.name
        try:
            tool_args = json.loads(tool_call.arguments) if isinstance(tool_call.arguments, str) else tool_call.arguments
        except (json.JSONDecodeError, AttributeError):
            tool_args = {}

        # 获取并执行工具
        tool = runtime.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        result = await tool.ainvoke(tool_args)

        # 将工具结果添加到对话历史
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call.id
        )
        MessageUtils.add_tool_message(tool_message, self.context_engine, runtime)

        return result

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """同步调用 - 完整的 ReAct 循环
        
        Args:
            inputs: 输入数据，必须包含 'query' 字段
            runtime: 可选的 Runtime（如果不提供，使用 BaseAgent 的 _runtime）
        
        Returns:
            执行结果
        """
        # 1. 准备 Runtime
        session_id = inputs.get("conversation_id", "default_session")
        runtime_created = False
        if runtime is None:
            # 使用 BaseAgent 的 _runtime，需要创建 task runtime
            runtime = await self._runtime.pre_run(session_id=session_id, inputs=inputs)
            runtime_created = True

        try:
            user_input = inputs.get("query", "")
            if not user_input:
                return {"output": "No query provided", "result_type": "error"}

            # 2. ReAct 循环
            iteration = 0
            max_iteration = self._agent_config.constrain.max_iteration
            is_first_call = True

            while iteration < max_iteration:
                iteration += 1
                logger.info(f"ReAct iteration {iteration}")

                # 2.1 调用模型进行推理
                llm_output = await self.call_model(
                    user_input,
                    runtime,
                    is_first_call=is_first_call
                )
                is_first_call = False  # 第一次调用后设为 False

                # 2.2 如果没有工具调用，说明 LLM 认为问题已解决
                if not llm_output.tool_calls:
                    logger.info("No tool calls, task completed")
                    return {
                        "output": llm_output.content,
                        "result_type": "answer"
                    }

                # 2.3 执行工具调用（工具结果已在 _execute_tool_call 中添加到历史）
                for tool_call in llm_output.tool_calls:
                    tool_name = tool_call.name
                    logger.info(f"Executing tool: {tool_name}")
                    result = await self._execute_tool_call(tool_call, runtime)
                    logger.info(f"Tool {tool_name} completed with result: {result}")

            # 3. 超过最大迭代次数
            logger.warning(f"Exceeded max iteration {max_iteration}")
            return {
                "output": "Exceeded max iteration",
                "result_type": "error"
            }
        finally:
            # 4. 清理 runtime（如果是我们创建的）
            if runtime_created:
                await runtime.post_run()

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """流式调用 - 极简版
        """
        # 准备runtime
        session_id = inputs.get("conversation_id", "default_session")
        runtime_created = False
        if runtime is None:
            # 使用 BaseAgent 的 _runtime，需要创建 task runtime
            runtime = await self._runtime.pre_run(session_id=session_id, inputs=inputs)
            runtime_created = True

        async def stream_process():
            try:
                final_result = await self.invoke(inputs, runtime)
                await runtime.write_stream(OutputSchema(type="answer", index=0,
                                                        payload={"output": final_result, "result_type": "answer"}))
            except Exception as e:
                logger.error(f"ReActAgent stream error: {e}")
            finally:
                # 清理 runtime（如果是我们创建的）
                if runtime_created:
                    await runtime.post_run()

        task = asyncio.create_task(stream_process())
        async for result in runtime.stream_iterator():
            yield result
        await task

# ===== 工厂函数 =====

def create_react_agent_config(
        agent_id: str,
        agent_version: str,
        description: str,
        workflows: List[WorkflowSchema],
        plugins: List[PluginSchema],
        model: ModelConfig,
        prompt_template: List[Dict],
        tools: List[str] = None
) -> ReActAgentConfig:
    """创建 ReAct Agent 配置
    
    Args:
        agent_id: Agent ID
        agent_version: Agent 版本
        description: Agent 描述
        workflows: 工作流 Schema 列表
        plugins: 插件 Schema 列表
        model: 模型配置
        prompt_template: 提示词模板
        tools: 工具名称列表
    
    Returns:
        ReActAgentConfig 实例
    """
    return ReActAgentConfig(
        id=agent_id,
        version=agent_version,
        description=description,
        workflows=workflows,
        plugins=plugins,
        model=model,
        prompt_template=prompt_template,
        tools=tools or []
    )


def create_react_agent(
        agent_config: ReActAgentConfig,
        workflows: List[Workflow] = None,
        tools: List[Tool] = None
) -> ReActAgent:
    """创建 ReAct Agent
    
    Args:
        agent_config: ReAct 配置
        workflows: 工作流列表
        tools: 工具列表
    
    Returns:
        ReActAgent 实例
    """
    return ReActAgent(agent_config, workflows, tools)
