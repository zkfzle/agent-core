#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

"""AgentGroup 基础接口"""

from abc import ABC, abstractmethod
from typing import Any, Dict, AsyncIterator


class BaseAgentGroup(ABC):
    """BaseAgentGroup - AgentGroup基础接口
    
    定义AgentGroup的核心接口，所有AgentGroup实现必须遵循此接口。
    """
    
    @abstractmethod
    async def start(self) -> None:
        """启动AgentGroup"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """停止AgentGroup"""
        pass
    
    @abstractmethod
    def register_agent(self, agent: Any) -> None:
        """注册Agent到AgentGroup
        
        Args:
            agent: Agent实例
        """
        pass
    
    @abstractmethod
    def unregister_agent(self, agent_id: str) -> None:
        """从AgentGroup注销Agent
        
        Args:
            agent_id: Agent ID
        """
        pass
    
    @abstractmethod
    async def invoke(self, request: Dict[str, Any]) -> Any:
        """同步调用接口 - 返回最终处理结果
        
        Args:
            request: 请求数据，包含conversation_id和query等
            
        Returns:
            处理结果
        """
        pass
    
    @abstractmethod
    async def stream(self, request: Dict[str, Any]) -> AsyncIterator[Any]:
        """流式调用接口 - 流式返回结果
        
        Args:
            request: 请求数据，包含conversation_id和query等
            
        Yields:
            流式处理结果
        """
        pass

