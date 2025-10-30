#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

"""AgentGroup 消息池（对应Publisher）"""

import asyncio
from typing import Optional, Any, Dict


class GroupMessagePool:
    """GroupMessagePool - 组消息池，对应Publisher，纯事件驱动模式
    
    消息池负责发布消息到消息中间件，支持请求-响应模式。
    """
    
    def __init__(self, message_broker: Any, topic: str):
        """初始化消息池
        
        Args:
            message_broker: 消息中间件实例
            topic: 发布的主题
        """
        self.message_broker = message_broker
        self.topic = topic
        self.publisher: Optional[Any] = None
        self._running = False
        self.result_waiter: Dict[str, asyncio.Future] = {}  # 结果等待器，用于请求-响应模式
    
    async def start(self) -> None:
        """启动消息池"""
        if self._running:
            return
        
        # 创建Publisher
        self.publisher = self.message_broker.create_publisher(self.topic)
        await self.publisher.start()
        
        self._running = True
    
    async def stop(self) -> None:
        """停止消息池"""
        if not self._running:
            return
        
        # 取消所有等待中的Future
        for future in self.result_waiter.values():
            if not future.done():
                future.cancel()
        self.result_waiter.clear()
        
        # 停止Publisher
        if self.publisher:
            await self.publisher.stop()
            self.publisher = None
        
        self._running = False
    
    async def publish_message(self, message: Any) -> bool:
        """发布消息 - 纯事件驱动模式的核心
        
        Args:
            message: 消息对象
            
        Returns:
            是否发布成功
        """
        if not self._running or not self.publisher:
            return False
        
        return await self.publisher.publish(message)
    
    async def publish_and_wait_result(self, message: Any, timeout: float = 30.0) -> Any:
        """发布消息并等待处理结果 - 请求-响应模式
        
        Args:
            message: 消息对象
            timeout: 超时时间（秒）
            
        Returns:
            处理结果
            
        Raises:
            asyncio.TimeoutError: 超时
        """
        if not self._running or not self.publisher:
            raise RuntimeError("MessagePool is not running")
        
        # 创建Future用于等待结果
        message_id = getattr(message, 'id', None)
        if not message_id:
            raise ValueError("Message must have an id field")
        
        future = asyncio.Future()
        self.result_waiter[message_id] = future
        
        try:
            # 发布消息
            success = await self.publisher.publish(message)
            if not success:
                raise RuntimeError("Failed to publish message")
            
            # 等待结果
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
            
        finally:
            # 清理
            self.result_waiter.pop(message_id, None)
    
    def set_result(self, message_id: str, result: Any) -> None:
        """设置消息处理结果
        
        Args:
            message_id: 消息ID
            result: 处理结果
        """
        future = self.result_waiter.get(message_id)
        if future and not future.done():
            future.set_result(result)
    
    def set_exception(self, message_id: str, exception: Exception) -> None:
        """设置消息处理异常
        
        Args:
            message_id: 消息ID
            exception: 异常对象
        """
        future = self.result_waiter.get(message_id)
        if future and not future.done():
            future.set_exception(exception)

