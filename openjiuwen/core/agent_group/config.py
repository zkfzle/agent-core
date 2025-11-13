#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

"""AgentGroup 配置类"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict


@dataclass
class MessageBrokerConfig:
    """消息中间件配置"""
    broker_type: str = "memory"  # memory, rabbitmq, kafka, redis, nats
    connection_string: Optional[str] = None
    delivery_mode: str = "at_least_once"  # at_most_once, at_least_once, exactly_once
    additional_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentGroupConfig:
    """AgentGroup配置"""
    group_id: str
    max_agents: int = 10
    message_broker_config: Optional[MessageBrokerConfig] = None
    enable_cross_agent_communication: bool = True
    max_concurrent_messages: int = 100
    message_timeout: float = 30.0
    
    def __post_init__(self):
        """初始化默认配置"""
        if self.message_broker_config is None:
            self.message_broker_config = MessageBrokerConfig()

