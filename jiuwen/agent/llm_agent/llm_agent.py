#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import Dict, List, Any, AsyncIterator

from jiuwen.agent.common.enum import ControllerType
from jiuwen.agent.common.schema import WorkflowSchema, PluginSchema
from jiuwen.agent.config.react_config import ReActAgentConfig
from jiuwen.agent.llm_agent.llm_message_handler import ReActMessageHandler
from jiuwen.core.agent.agent import Agent
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.runtime.config import Config
from jiuwen.core.runtime.runtime import Runtime, Workflow
from jiuwen.core.utils.tool.base import Tool


def create_react_agent_config(agent_id: str,
                              agent_version: str,
                              description: str,
                              workflows: List[WorkflowSchema],
                              plugins: List[PluginSchema],
                              model: ModelConfig,
                              prompt_template: List[Dict],
                              tools: List[str] = []):
    config = ReActAgentConfig(id=agent_id,
                              version=agent_version,
                              description=description,
                              workflows=workflows,
                              plugins=plugins,
                              model=model,
                              prompt_template=prompt_template,
                              tools=tools)
    return config


def create_react_agent(agent_config: ReActAgentConfig,
                       workflows: List[Workflow] = None,
                       tools: List[Tool] = None):
    agent = ReActAgent(agent_config)
    agent.bind_workflows(workflows)
    agent.bind_tools(tools or [])
    return agent


class ReActAgent(Agent):
    """ReAct模式的Agent - 使用LLM reasoning生成执行计划"""
    
    def __init__(self, agent_config: ReActAgentConfig):
        # 验证 controller_type
        if agent_config.controller_type != ControllerType.ReActController:
            raise NotImplementedError(f"ReActAgent requires ReActController, got {agent_config.controller_type}")

        # 创建配置并初始化基类
        config = Config()
        config.set_agent_config(agent_config=agent_config)
        super().__init__(config)
        
        # 设置消息处理器
        self.set_message_handler(ReActMessageHandler)

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """批调用 - 使用基类的通用实现"""
        return await self.controller_invoke(inputs, runtime)

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """流式调用 - 使用基类的通用实现"""
        async for result in self.controller_stream(inputs, runtime):
            yield result
