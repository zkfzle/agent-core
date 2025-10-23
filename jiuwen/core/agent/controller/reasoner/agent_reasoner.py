#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List, Optional
from jiuwen.core.agent.message.message import Message
from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.controller.reasoner.intent_detection import IntentDetection
from jiuwen.core.agent.controller.reasoner.planner import Planner


class AgentReasoner:
    """AgentReasoner - Agent决策模块，负责消息的智能决策和任务生成"""

    def __init__(self, config, context_engine, runtime):
        """
        初始化AgentReasoner
        
        Args:
            config: AgentReasoner配置
            context_engine: 上下文引擎
            runtime: 运行时环境
        """
        self.config = config
        self.context_engine = context_engine
        self.runtime = runtime

        # 子模块
        self.intent_detection: Optional[IntentDetection] = None
        self.planner: Optional[Planner] = None

    async def process_message(self, message: Message) -> List[Task]:
        """
        处理消息 - 统一的决策处理入口
        
        Args:
            message: 输入消息
            
        Returns:
            List[Task]: 生成的任务列表
        """
        pass

    def set_intent_detection(self, intent_detection: IntentDetection) -> 'AgentReasoner':
        """
        设置意图识别模块
        
        Args:
            intent_detection: 意图识别模块实例
            
        Returns:
            AgentReasoner: 支持链式调用
        """
        self.intent_detection = intent_detection
        logger.debug("Intent detection module set")
        return self

    def set_planner(self, planner: Planner) -> 'AgentReasoner':
        """
        设置规划器模块
        
        Args:
            planner: 规划器模块实例
            
        Returns:
            AgentReasoner: 支持链式调用
        """
        self.planner = planner
        return self

    async def use_intent_detection(self, message: Message) -> List[Task]:
        """
        直接使用意图识别模块处理消息
        
        Args:
            message: 输入消息
            
        Returns:
            List[Task]: 生成的任务列表
        """
        if not self.intent_detection:
            raise ValueError("Intent detection module not set")
        return await self.intent_detection.process_message(message)

    async def use_planner(self, message: Message) -> List[Task]:
        """
        直接使用规划器模块处理消息
        
        Args:
            message: 输入消息
            
        Returns:
            List[Task]: 生成的任务列表
        """
        if not self.planner:
            raise ValueError("Planner module not set")
        return await self.planner.process_message(message)

    @property
    def intent_detection_module(self) -> Optional[IntentDetection]:
        """获取意图识别模块"""
        return self.intent_detection

    @property
    def planner_module(self) -> Optional[Planner]:
        """获取规划器模块"""
        return self.planner
