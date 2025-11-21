#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

"""Agent Group Base Module"""

from abc import ABC, abstractmethod
from typing import Any, Dict, AsyncIterator

from openjiuwen.core.agent.agent import BaseAgent, AgentRuntime
from openjiuwen.core.agent_group.config import AgentGroupConfig
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runtime.config import Config
from openjiuwen.core.runtime.resources_manager.resource_manager import ResourceMgr


class AgentGroupRuntime(AgentRuntime):
    def __init__(self, config: Config = None, resource_mgr: ResourceMgr = None):
        super().__init__(config, resource_mgr)


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

    def get_agent_count(self) -> int:
        """
        Get the number of agents currently in the group.

        Returns:
            int: Number of agents in the group
        """
        return len(self.agents)
    
    @abstractmethod
    async def invoke(self, inputs: Dict, runtime: AgentGroupRuntime = None) -> Any:
        """
        Execute a synchronous operation on the agent group.

        This method processes input data through the group of agents and returns
        the collective result. It should handle the complete execution flow
        including task distribution, agent coordination, and result aggregation.
        
        Args:
            inputs: Input data to be processed by the group
            runtime: Runtime for agent group instance
            
        Returns:
            The collective output from the agent group
        """
        raise NotImplementedError(
            f"invoke method or controller method must be implemented {self.__class__.__name__}"
        )
    
    @abstractmethod
    async def stream(self, inputs: Dict, runtime: AgentGroupRuntime = None) -> AsyncIterator[Any]:
        """
        Execute a streaming operation on the agent group.

        This method processes input data and returns results as a stream,
        allowing for real-time or progressive output from the agent group.
        Useful for long-running operations or when intermediate results are needed.

        Args:
            inputs: Input data to be processed by the group
            runtime: Runtime for agent group instance

        Returns:
            The collective output from the agent group
        """
        raise NotImplementedError(
            f"stream method must be implemented by {self.__class__.__name__}"
        )

