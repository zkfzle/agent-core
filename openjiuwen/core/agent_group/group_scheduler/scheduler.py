#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

"""AgentGroup 调度器"""

from typing import Optional, Any


class GroupScheduler:
    """GroupScheduler - 组级调度器，纯事件驱动模式，无需轮询循环
    
    调度器负责协调消息池和消息处理器，不包含轮询逻辑。
    所有消息处理都通过事件驱动的回调机制完成。
    """
    
    def __init__(self, message_pool: Any, message_handler: Any):
        """初始化调度器
        
        Args:
            message_pool: GroupMessagePool实例（对应Publisher）
            message_handler: BaseGroupMessageHandler实例（对应Subscriber）
        """
        self.message_pool = message_pool
        self.message_handler = message_handler
        self._running = False
    
    async def start(self) -> None:
        """启动调度器 - 纯事件驱动模式，无需轮询循环
        
        启动消息池和消息处理器，所有后续处理都通过事件驱动完成。
        """
        if self._running:
            return
        
        # 启动消息池（Publisher）
        await self.message_pool.start()
        
        # 启动消息处理器（Subscriber，绑定回调）
        await self.message_handler.start()
        
        self._running = True
    
    async def stop(self) -> None:
        """停止调度器"""
        if not self._running:
            return
        
        # 停止消息处理器
        await self.message_handler.stop()
        
        # 停止消息池
        await self.message_pool.stop()
        
        self._running = False

