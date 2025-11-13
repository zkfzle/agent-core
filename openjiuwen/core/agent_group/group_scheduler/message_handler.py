#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

"""AgentGroup 消息处理器基类"""

from abc import ABC, abstractmethod
from typing import Any, Optional, AsyncIterator


class BaseGroupMessageHandler(ABC):
    """BaseGroupMessageHandler - 基础消息处理器，可扩展设计
    
    提供以下扩展点：
    1. handle_message() - 核心消息处理逻辑（非流式）
    2. handle_message_streamed() - 核心消息处理逻辑（流式）
    3. preprocess_message() - 消息预处理
    4. postprocess_result() - 结果后处理
    5. handle_error() - 错误处理
    6. should_retry() - 重试策略
    """

    def __init__(self, message_broker: Any, topic: str, agent_group: Any):
        """初始化消息处理器
        
        Args:
            message_broker: 消息中间件实例
            topic: 订阅的主题
            agent_group: AgentGroup实例引用
        """
        self.message_broker = message_broker
        self.topic = topic
        self.agent_group = agent_group
        self.subscriber: Optional[Any] = None
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """启动消息处理器 - 纯事件驱动模式"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止消息处理器"""
        pass

    @abstractmethod
    async def handle_message(self, message: Any):
        """处理消息 - 子类必须实现此方法
        
        Args:
            message: 消息对象
            
        Returns:
            处理是否成功
        """
        pass

    async def preprocess_message(self, message: Any) -> Optional[Any]:
        """消息预处理 - 子类可重写此方法
        
        Args:
            message: 原始消息
            
        Returns:
            预处理后的消息，如果返回None则跳过处理
        """
        return message

    async def postprocess_result(self, result: bool, message: Any) -> bool:
        """结果后处理 - 子类可重写此方法
        
        Args:
            result: 处理结果
            message: 消息对象
            
        Returns:
            后处理后的结果
        """
        return result
