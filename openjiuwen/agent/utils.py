#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
MessageUtils - 消息处理工具类

职责：处理 Agent 与 ContextEngine 之间的消息添加和获取操作
"""

from typing import List, Any
from openjiuwen.core.utils.llm.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from openjiuwen.core.context_engine.engine import ContextEngine
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.agent.config.base import AgentConfig


class MessageUtils:
    """消息工具类 - 处理消息的添加和获取"""

    @staticmethod
    def should_add_user_message(query: str, context_engine: ContextEngine, runtime: Runtime) -> bool:
        """判断是否应该添加用户消息
        
        Args:
            query: 用户输入
            context_engine: 上下文引擎
            runtime: Runtime 实例
        
        Returns:
            bool: 是否应该添加用户消息
        """
        agent_context = context_engine.get_agent_context(runtime.session_id())
        last_message = agent_context.get_latest_message()

        if not last_message:
            return True

        if last_message.role == 'tool':
            logger.info("Skipping user message - post-tool-call request")
            return False

        if last_message.role == 'user' and last_message.content == query:
            logger.info("Skipping duplicate user message")
            return False

        return True

    @staticmethod
    def add_user_message(query: Any, context_engine: ContextEngine, runtime: Runtime):
        """添加用户消息到对话历史
        
        Args:
            query: 用户输入
            context_engine: 上下文引擎
            runtime: Runtime 实例
        """
        if MessageUtils.should_add_user_message(query, context_engine, runtime):
            agent_context = context_engine.get_agent_context(runtime.session_id())
            user_message = HumanMessage(content=query)
            agent_context.add_message(user_message)
            if UserConfig.is_sensitive():
                logger.info(f"Added user message")
            else:
                logger.info(f"Added user message: {query}")

    @staticmethod
    def add_ai_message(ai_message: AIMessage, context_engine: ContextEngine, runtime: Runtime):
        """添加 AI 消息到对话历史
        
        Args:
            ai_message: AI 消息对象
            context_engine: 上下文引擎
            runtime: Runtime 实例
        """
        if ai_message:
            agent_context = context_engine.get_agent_context(runtime.session_id())
            agent_context.add_message(ai_message)

    @staticmethod
    def add_tool_message(tool_message: ToolMessage, context_engine: ContextEngine, runtime: Runtime):
        """添加工具消息到对话历史
        
        Args:
            tool_message: 工具消息对象
            context_engine: 上下文引擎
            runtime: Runtime 实例
        """
        if tool_message:
            agent_context = context_engine.get_agent_context(runtime.session_id())
            agent_context.add_message(tool_message)

    @staticmethod
    def add_workflow_message(message: BaseMessage, workflow_id: str,
                            context_engine: ContextEngine, runtime: Runtime):
        """添加消息到 workflow 的聊天历史
        
        Args:
            message: 消息对象
            workflow_id: 工作流 ID
            context_engine: 上下文引擎
            runtime: Runtime 实例
        """
        workflow_context = context_engine.get_workflow_context(
            workflow_id=workflow_id,
            session_id=runtime.session_id()
        )
        workflow_context.add_message(message)

    @staticmethod
    def get_chat_history(context_engine: ContextEngine, runtime: Runtime, config: AgentConfig) -> List[BaseMessage]:
        """获取对话历史
        
        Args:
            context_engine: 上下文引擎
            runtime: Runtime 实例
            config: Agent 配置
        
        Returns:
            List[BaseMessage]: 对话历史消息列表
        """
        agent_context = context_engine.get_agent_context(runtime.session_id())
        chat_history = agent_context.get_messages()
        max_rounds = config.constrain.reserved_max_chat_rounds
        return chat_history[-2 * max_rounds:]

