#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

"""Agent Group Base Module"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, AsyncIterator

from openjiuwen.core.agent.agent import BaseAgent
from openjiuwen.core.common.logging import logger
# 直接从 openjiuwen.core.agent.agent 导入 AgentRuntime
from openjiuwen.core.agent.agent import AgentRuntime
from openjiuwen.core.agent_group.config import AgentGroupConfig
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runtime.config import Config
from openjiuwen.core.runtime.resources_manager.resource_manager import ResourceMgr
from openjiuwen.core.stream.base import OutputSchema


class AgentGroupRuntime(AgentRuntime):
    """AgentGroup 专用 Runtime
    
    直接继承 openjiuwen.core.agent.agent.AgentRuntime
    复用其所有能力，包括 pre_run() 返回的 TaskRuntime
    
    为什么可以直接继承：
    1. AgentRuntime(config, resource_mgr) 的构造函数签名简单
    2. 它已经包含了 write_stream() 方法
    3. 通过 pre_run() 可以获得 TaskRuntime，后者有 stream_iterator()
    """
    
    def __init__(self, config: Config = None, resource_mgr: ResourceMgr = None):
        """初始化 AgentGroupRuntime
        
        Args:
            config: 配置对象（可选，会自动创建）
            resource_mgr: 资源管理器（可选，会自动创建）
        """
        # 如果没有提供 config，创建一个带有 agent_config 的 Config
        if config is None:
            from openjiuwen.agent.config.base import AgentConfig
            config = Config()
            # 创建一个虚拟的 AgentConfig 用于 Group Runtime
            agent_config = AgentConfig(id="agent_group_runtime")
            config.set_agent_config(agent_config)
        
        # 直接调用父类构造函数
        super().__init__(config, resource_mgr)
    
    # write_stream() 方法已经在父类 AgentRuntime 中实现了
    # 无需重复定义


class BaseGroup(ABC):
    """
    Abstract base class for implementing agent groups.

    This class provides the foundational structure and common functionality
    for managing groups of agents in a multi-agent system. It defines the
    essential interface that all concrete agent group implementations must
    follow, ensuring consistency across different group types.
    """

    def __init__(self, config: AgentGroupConfig):
        """
        Initialize the agent group.

        Args:
            config (AgentGroupConfig): The configuration object for this group.
        """
        self.config: AgentGroupConfig = config
        self.group_id = config.group_id
        self.agents: Dict[str, BaseAgent] = {}

    def add_agent(self, agent_id: str, agent: BaseAgent):
        """
        注册Agent

        Args:
            agent_id: Agent唯一标识符（主键）
            agent: Agent实例

        Raises:
            ValueError: Agent ID已经存在
        """
        if agent_id in self.agents:
            raise JiuWenBaseException(
                StatusCode.AGENT_GROUP_ADD_FAILED.code,
                StatusCode.AGENT_GROUP_ADD_FAILED.errmsg.format("Agent ID already exists")
            )
        else:
            if self.get_agent_count() == self.config.max_agents:
                raise JiuWenBaseException(
                    StatusCode.AGENT_GROUP_ADD_FAILED.code,
                    StatusCode.AGENT_GROUP_ADD_FAILED.errmsg.format("Agent count exceeds max agents"))
            self.agents[agent_id] = agent
            
            # Auto-inject group reference to agent's controller
            # Duck typing: if controller has set_group method, inject self
            if hasattr(agent, 'controller') and agent.controller is not None:
                if hasattr(agent.controller, 'set_group'):
                    agent.controller.set_group(self)
                    logger.debug(
                        f"BaseGroup: Auto-injected group reference to "
                        f"agent '{agent_id}' controller"
                    )

    def get_agent_count(self) -> int:
        """
        Get the number of agents currently in the group.

        Returns:
            int: Number of agents in the group
        """
        return len(self.agents)
    
    @abstractmethod
    async def invoke(self, message, runtime: AgentGroupRuntime = None) -> Any:
        """
        Execute a synchronous operation on the agent group.

        This method processes message through the group of agents and returns
        the collective result. It should handle the complete execution flow
        including task distribution, agent coordination, and result aggregation.
        
        Args:
            message: Message object (for compatibility, also supports Dict for backward compatibility)
            runtime: Runtime for agent group instance
            
        Returns:
            The collective output from the agent group
        """
        raise NotImplementedError(
            f"invoke method or controller method must be implemented {self.__class__.__name__}"
        )
    
    @abstractmethod
    async def stream(self, message, runtime: AgentGroupRuntime = None) -> AsyncIterator[Any]:
        """
        Execute a streaming operation on the agent group.

        This method processes message and returns results as a stream,
        allowing for real-time or progressive output from the agent group.
        Useful for long-running operations or when intermediate results are needed.

        Args:
            message: Message object (for compatibility, also supports Dict for backward compatibility)
            runtime: Runtime for agent group instance

        Returns:
            The collective output from the agent group
        """
        raise NotImplementedError(
            f"stream method must be implemented by {self.__class__.__name__}"
        )


class ControllerGroup(BaseGroup):
    """Agent Group with Controller
    
    Design features (similar to ControllerAgent):
    1. Inherits BaseGroup, reuses agent management capabilities
    2. Holds GroupController, fully delegates message routing logic
    3. Automatically configures GroupController (via setup_from_group)
    4. invoke/stream fully delegated to group_controller
    5. Runtime lifecycle: pre_run -> controller.invoke -> post_run
    """

    def __init__(self, config: AgentGroupConfig, group_controller=None):
        """Initialize ControllerGroup
        
        Args:
            config: AgentGroup configuration object
            group_controller: Optional GroupController instance (will be auto-configured)
        
        Usage:
            # Simplest way - group_controller auto-configured
            group_controller = DefaultGroupController()  # No parameters needed
            group = ControllerGroup(config=config, group_controller=group_controller)
        """
        super().__init__(config)
        self.group_controller = group_controller

        # Initialize runtime (like BaseAgent)
        self._runtime = AgentGroupRuntime()

        # Auto-configure group_controller
        if self.group_controller is not None:
            self._setup_group_controller()

    def _setup_group_controller(self):
        """Auto-configure group_controller (inject group reference)"""
        if hasattr(self.group_controller, 'setup_from_group'):
            self.group_controller.setup_from_group(self)

    def _convert_message(self, message):
        """Convert dict to Message if needed (backward compatibility)"""
        from openjiuwen.core.agent.message.message import Message
        if isinstance(message, dict):
            return Message.create_user_message(
                content=message.get("content") or message.get("query", ""),
                conversation_id=message.get("conversation_id", "default_session")
            )
        return message

    async def invoke(self, message, runtime: AgentGroupRuntime = None) -> Any:
        """Synchronous invocation - Fully delegated to group_controller
        
        Lifecycle: pre_run -> controller.invoke -> post_run
        
        Args:
            message: Message object (carries message_type for routing)
            runtime: Runtime instance (optional, auto-created if None)
        
        Returns:
            Processing result
        """
        if not self.group_controller:
            raise RuntimeError(
                f"{self.__class__.__name__} has no group_controller"
            )

        message = self._convert_message(message)
        session_id = message.context.conversation_id if message.context else "default"

        # If runtime not provided, use self._runtime.pre_run to create task runtime
        if runtime is None:
            task_runtime = await self._runtime.pre_run(session_id=session_id)
            need_cleanup = True
        else:
            task_runtime = runtime
            need_cleanup = False

        try:
            # Fully delegate to group_controller
            result = await self.group_controller.invoke(message, task_runtime)
            return result if result is not None else {"output": "processed"}
        finally:
            if need_cleanup:
                await task_runtime.post_run()

    async def stream(self, message, runtime: AgentGroupRuntime = None) -> AsyncIterator[Any]:
        """Streaming invocation - 真正的流式输出
        
        Design: 
        1. 后台任务执行 group_controller.invoke
        2. group_controller.send_to_agent 会调用 agent.stream 并透传 chunk 到 runtime
        3. 本方法从 runtime.stream_iterator() 实时读取并 yield
        
        流式数据来源：
        - 子 agent 的流式输出通过共享 runtime 透传
        - 包括 __interaction__、workflow_final 等所有类型
        
        Args:
            message: Message object (carries message_type for routing)
            runtime: Runtime instance (optional, auto-created if None)
        
        Yields:
            Streaming output from sub-agents
        """
        if not self.group_controller:
            raise RuntimeError(
                f"{self.__class__.__name__} has no group_controller"
            )

        message = self._convert_message(message)
        session_id = message.context.conversation_id if message.context else "default"

        # If runtime not provided, use self._runtime.pre_run to create task runtime
        if runtime is None:
            task_runtime = await self._runtime.pre_run(session_id=session_id)
            need_cleanup = True
        else:
            task_runtime = runtime
            need_cleanup = False

        # 后台任务执行 group_controller.invoke
        # send_to_agent 会调用 agent.stream 并把 chunk 写入 task_runtime
        async def run_controller():
            try:
                await self.group_controller.invoke(message, task_runtime)
            finally:
                if need_cleanup:
                    await task_runtime.post_run()

        task = asyncio.create_task(run_controller())

        # 真正流式读取：从 stream_iterator 实时获取 chunk
        async for chunk in task_runtime.stream_iterator():
            yield chunk

        # 等待后台任务完成
        await task

