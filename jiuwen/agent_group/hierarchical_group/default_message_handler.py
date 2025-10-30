#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

"""默认消息处理器实现"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, List

from jiuwen.core.agent_group import BaseGroupMessageHandler
from jiuwen.core.common.logging import logger


@dataclass
class AgentGroupState:
    """AgentGroup状态管理 - 最大程度简化设计

    核心状态字段：
    - current_agent_id: 当前执行的Agent ID
    - interrupted_agents: 中断的Agent列表，用于中断恢复
    - current_agent_calls_count: 当前Agent调用计数
    - last_updated_at: 最后更新时间
    """

    current_agent_id: Optional[str] = None
    interrupted_agents: List[str] = field(default_factory=list)
    current_agent_calls_count: int = 0
    last_updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "current_agent_id": self.current_agent_id,
            "interrupted_agents": self.interrupted_agents,
            "current_agent_calls_count": self.current_agent_calls_count,
            "last_updated_at": self.last_updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentGroupState":
        """从字典创建状态对象"""
        return cls(
            current_agent_id=data.get("current_agent_id"),
            interrupted_agents=data.get("interrupted_agents", []),
            current_agent_calls_count=data.get("current_agent_calls_count", 0),
            last_updated_at=datetime.fromisoformat(data.get("last_updated_at", datetime.now().isoformat()))
        )


class DefaultGroupMessageHandler(BaseGroupMessageHandler):
    """DefaultGroupMessageHandler - 默认消息处理器实现
    
    提供基本的消息处理逻辑，用户可以通过继承BaseGroupMessageHandler
    并重写handle_message()方法来实现自定义处理逻辑。
    """

    def __init__(self, message_broker: Any, topic: str, agent_group: Any):
        """初始化默认消息处理器
        
        Args:
            message_broker: 消息中间件实例
            topic: 订阅的主题
            agent_group: AgentGroup实例引用
        """
        super().__init__(message_broker, topic, agent_group)
        logger.info(f"DefaultGroupMessageHandler initialized for topic '{topic}'")

    async def start(self) -> None:
        """启动消息处理器 - 纯事件驱动模式"""
        if self._running:
            return

        # 创建Subscriber并绑定回调
        subscription_config = {
            "topic": self.topic,
            "consumer_group": f"agent_group_{self.agent_group.group_id}",
            "auto_ack": True
        }

        self.subscriber = self.message_broker.create_subscriber(
            subscription_config,
            self.handle_message  # 绑定回调函数
        )

        await self.subscriber.start()
        self._running = True

        logger.info(f"DefaultGroupMessageHandler started for topic '{self.topic}'")

    async def stop(self) -> None:
        """停止消息处理器"""
        if not self._running:
            return

        if self.subscriber:
            await self.subscriber.stop()
            self.subscriber = None

        self._running = False
        logger.info(f"DefaultGroupMessageHandler stopped for topic '{self.topic}'")

    async def handle_message(self, message: Any):
        """处理消息 - 默认实现，包含消息路由逻辑（非流式）
        
        Args:
            message: 消息对象
            
        Returns:
            处理是否成功
        """
        pass

    async def preprocess_message(self, message: Any) -> Optional[Any]:
        """消息预处理 - 验证消息格式
        
        Args:
            message: 原始消息
            
        Returns:
            预处理后的消息，如果返回None则跳过处理
        """
        # 验证消息格式
        if not message.get('id'):
            logger.warning("Message without id, skipping")
            return None

        return message

    async def postprocess_result(self, result: bool, message: Any):
        """结果后处理 - 记录处理结果
        
        Args:
            result: 处理结果
            message: 消息对象
            
        Returns:
            后处理后的结果
        """
        if result:
            logger.info(f"Message {message.get('id')} processed successfully")
        else:
            logger.warning(f"Message {message.get('id')} processing failed")

        return result
