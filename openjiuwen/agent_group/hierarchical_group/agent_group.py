#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

"""AgentGroup 主类实现"""

from typing import Dict, Any, Optional, AsyncIterator

from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.agent_group import (
    AgentGroupConfig,
    GroupScheduler,
    GroupMessagePool,
)
from openjiuwen.agent_group.hierarchical_group.default_message_handler import DefaultGroupMessageHandler
from openjiuwen.core.common.logging import logger


class AgentGroup:
    """AgentGroup - 多Agent协同管理，有状态设计
    
    核心职责：
    1. 管理多个Agent的注册和注销
    2. 提供统一的调用接口（invoke/stream）
    3. 管理AgentGroup状态（通过runtime）
    4. 协调消息路由和调度
    """

    def __init__(self, config: AgentGroupConfig):
        """初始化AgentGroup
        
        Args:
            config: AgentGroup配置
        """
        self.config = config
        self.group_id = config.group_id
        self.agents: Dict[str, Any] = {}

        # 消息服务（全局共享）
        self.message_service = None  # TODO: 从MessageService获取
        self.message_broker = None  # TODO: 从message_service获取

        # TODO Runtime（用于状态管理和流式输出）
        self.runtime = self._create_runtime()

        # 组调度器
        self.group_scheduler: Optional[GroupScheduler] = None
        self._message_pool: Optional[GroupMessagePool] = None
        self._message_handler: Optional[DefaultGroupMessageHandler] = None

        self._running = False

        logger.info(f"AgentGroup '{self.group_id}' initialized")

    async def start(self) -> None:
        """启动AgentGroup"""
        if self._running:
            logger.warning(f"AgentGroup '{self.group_id}' is already running")
            return

        # 创建消息池和消息处理器
        topic = f"agent_group_{self.group_id}"
        self._message_pool = GroupMessagePool(self.message_broker, topic)
        self._message_handler = DefaultGroupMessageHandler(
            self.message_broker,
            topic,
            self
        )

        # 创建调度器
        self.group_scheduler = GroupScheduler(
            self._message_pool,
            self._message_handler
        )

        # 启动调度器（纯事件驱动模式）
        await self.group_scheduler.start()

        self._running = True
        logger.info(f"AgentGroup '{self.group_id}' started")

    async def stop(self) -> None:
        """停止AgentGroup"""
        if not self._running:
            return

        # 停止调度器
        if self.group_scheduler:
            await self.group_scheduler.stop()

        self._running = False
        logger.info(f"AgentGroup '{self.group_id}' stopped")

    def register_agent(self, agent: Any) -> None:
        """注册Agent到AgentGroup
        
        Args:
            agent: Agent实例
        """
        agent_id = getattr(agent, 'agent_id', None)
        if not agent_id:
            raise ValueError("Agent must have an 'agent_id' attribute")

        if agent_id in self.agents:
            raise ValueError(f"Agent '{agent_id}' is already registered")

        self.agents[agent_id] = agent
        logger.info(f"Agent '{agent_id}' registered to AgentGroup '{self.group_id}'")

    def unregister_agent(self, agent_id: str) -> None:
        """从AgentGroup注销Agent
        
        Args:
            agent_id: Agent ID
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent '{agent_id}' is not registered")

        del self.agents[agent_id]
        logger.info(f"Agent '{agent_id}' unregistered from AgentGroup '{self.group_id}'")

    async def invoke(self, request: Dict[str, Any]) -> Any:
        """同步调用接口 - 返回最终处理结果
        
        Args:
            request: 请求数据，包含conversation_id和query等
            
        Returns:
            处理结果
        """
        if not self._running:
            raise RuntimeError(f"AgentGroup '{self.group_id}' is not running")

        # 创建用户消息
        result = await self.process_inputs(request)

        return result

    async def stream(self, request: Dict[str, Any]) -> AsyncIterator[Any]:
        """流式调用接口 - 流式返回结果
        
        Args:
            request: 请求数据，包含conversation_id和query等
            
        Yields:
            流式处理结果
        """
        if not self._running:
            raise RuntimeError(f"AgentGroup '{self.group_id}' is not running")

        await self.process_inputs(request)

        # 通过runtime的stream_iterator输出流式数据
        async for chunk in self.runtime.stream_iterator():
            yield chunk

    async def process_inputs(self, inputs: Dict[str, Any]) -> Any:
        """从用户请求创建UserMessage

        Args:
            inputs: 请求数据

        Returns:
            消息对象
        """
        # 1. 创建消息
        session_id = inputs.get("conversation_id", "default_session")
        message = Message.create_user_message(
            content=inputs.get("query", ""),
            conversation_id=session_id
        )

        # 发布消息并等待结果
        result = await self._message_pool.publish_and_wait_result(
            message,
            timeout=self.config.message_timeout
        )
        return result

    def _create_runtime(self) -> Any:
        """创建AgentGroup专用的runtime
        
        Returns:
            AgentRuntime实例
        """
        # TODO: 创建实际的runtime
        return None
