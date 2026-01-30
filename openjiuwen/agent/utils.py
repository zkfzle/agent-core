# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import List, Any
from openjiuwen.core.utils.llm.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from openjiuwen.core.context_engine.engine import ContextEngine
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.agent.config.base import AgentConfig


class MessageUtils:
    """Message utilities for adding and retrieving messages"""

    @staticmethod
    def should_add_user_message(query: str, context_engine: ContextEngine, runtime: Runtime) -> bool:
        """Check if user message should be added
        
        Args:
            query: User input
            context_engine: Context engine
            runtime: Runtime instance
        
        Returns:
            bool: Whether to add user message
        """
        agent_context = context_engine.get_agent_context(runtime.session_id())
        last_message = agent_context.get_latest_message()

        if not last_message:
            return True

        if last_message.role == 'tool':
            logger.info("post-tool-call request")
            return True

        if last_message.role == 'user' and last_message.content == query:
            logger.info("Skipping duplicate user message")
            return False

        return True

    @staticmethod
    def add_user_message(query: Any, context_engine: ContextEngine, runtime: Runtime):
        """Add user message to chat history
        
        Args:
            query: User input
            context_engine: Context engine
            runtime: Runtime instance
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
        """Add AI message to chat history
        
        Args:
            ai_message: AI message object
            context_engine: Context engine
            runtime: Runtime instance
        """
        if ai_message:
            agent_context = context_engine.get_agent_context(runtime.session_id())
            agent_context.add_message(ai_message)

    @staticmethod
    def add_tool_message(tool_message: ToolMessage, context_engine: ContextEngine, runtime: Runtime):
        """Add tool message to chat history
        
        Args:
            tool_message: Tool message object
            context_engine: Context engine
            runtime: Runtime instance
        """
        if tool_message:
            agent_context = context_engine.get_agent_context(runtime.session_id())
            agent_context.add_message(tool_message)

    @staticmethod
    def add_workflow_message(message: BaseMessage, workflow_id: str,
                            context_engine: ContextEngine, runtime: Runtime):
        """Add message to workflow chat history
        
        Args:
            message: Message object
            workflow_id: Workflow ID
            context_engine: Context engine
            runtime: Runtime instance
        """
        workflow_context = context_engine.get_workflow_context(
            workflow_id=workflow_id,
            session_id=runtime.session_id()
        )
        workflow_context.add_message(message)

    @staticmethod
    def get_chat_history(context_engine: ContextEngine, runtime: Runtime, config: AgentConfig) -> List[BaseMessage]:
        """Get chat history with safe truncation

        Ensures the first message after system is always 'user' (API requirement).

        Args:
            context_engine: Context engine
            runtime: Runtime instance
            config: Agent config

        Returns:
            List[BaseMessage]: Chat history message list
        """
        agent_context = context_engine.get_agent_context(runtime.session_id())
        chat_history = agent_context.get_messages()
        max_rounds = config.constrain.reserved_max_chat_rounds

        # 如果消息数量在限制内，直接返回
        if len(chat_history) <= 2 * max_rounds:
            return chat_history

        # 需要截取时，找到安全的截取点
        cut_start = len(chat_history) - 2 * max_rounds

        # 向前搜索，找到一个 user 消息作为起点
        # 这确保截取后的第一条消息是 user（API 要求）
        while cut_start > 0:
            msg = chat_history[cut_start]
            # 支持 BaseMessage 对象和 dict 两种格式
            if hasattr(msg, 'role') and msg.role == 'user':
                break
            elif isinstance(msg, dict) and msg.get('role') == 'user':
                break
            cut_start -= 1

        # 如果找不到 user 消息，从头开始（保留所有消息）
        if cut_start == 0:
            # 检查第一条消息是否是 user
            first_msg = chat_history[0]
            is_user = (hasattr(first_msg, 'role') and first_msg.role == 'user') or \
                      (isinstance(first_msg, dict) and first_msg.get('role') == 'user')
            if not is_user:
                logger.warning("First message is not 'user', keeping all messages to avoid API error")

        return chat_history[cut_start:]

