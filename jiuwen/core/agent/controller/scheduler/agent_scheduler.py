#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import asyncio
from typing import Optional
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.controller.scheduler.message_handler import MessageHandler
from jiuwen.core.agent.controller.scheduler.task_handler import TaskHandler
from jiuwen.core.common.logging import logger


class AgentScheduler:
    """AgentScheduler - 纯调度器，管理两个队列"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # 只管理两个核心队列 - 使用默认配置值
        max_message_queue_size = getattr(config, 'max_message_queue_size', 1000)
        max_task_queue_size = getattr(config, 'max_task_queue_size', 1000)
        self.message_queue = asyncio.Queue(maxsize=max_message_queue_size)
        self.task_queue = asyncio.Queue(maxsize=max_task_queue_size)
        
        # 处理器引用（由外部设置）
        self.message_handler: Optional[MessageHandler] = None
        self.task_handler: Optional[TaskHandler] = None
        
        # 调度器状态
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        
    def set_handlers(self, message_handler: MessageHandler, task_handler: TaskHandler):
        """设置处理器引用"""
        self.message_handler = message_handler
        self.task_handler = task_handler
        
    async def start(self):
        """启动调度器 - 启动调度循环"""
        if self._running:
            logger.warning(f"Scheduler {self.agent_id} is already running")
            return
            
        if not self.message_handler or not self.task_handler:
            raise ValueError("Message handler and task handler must be set before starting scheduler")
            
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(f"Scheduler {self.agent_id} started")
        
    async def stop(self):
        """停止调度器 - 停止调度循环"""
        if not self._running:
            logger.warning(f"Scheduler {self.agent_id} is not running")
            return
            
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
                
        logger.info(f"Scheduler {self.agent_id} stopped")
            
    async def schedule_message(self, message: Message):
        """调度消息 - 统一的消息调度入口"""
        try:
            await self.message_queue.put(message)
            logger.debug(f"Message {message.msg_id} scheduled to queue")
        except asyncio.QueueFull:
            logger.error(f"Message queue is full, dropping message {message.msg_id}")
            
    async def schedule_task(self, task: Task):
        """调度任务 - 统一的任务调度入口"""
        try:
            await self.task_queue.put(task)
            logger.debug(f"Task {task.task_id} scheduled to queue")
        except asyncio.QueueFull:
            logger.error(f"Task queue is full, dropping task {task.task_id}")
        
    async def _scheduler_loop(self):
        """调度器主循环 - 只处理两个核心队列"""
        logger.info(f"Scheduler loop started for agent {self.agent_id}")
        
        while self._running:
            try:
                # 1. 处理消息队列 - Message Handler处理消息，生成任务
                await self._process_messages()
                
                # 2. 处理任务队列 - Task Handler执行任务
                await self._process_tasks()
                
                # 3. 如果没有消息，短暂休眠
                if self.message_queue.empty() and self.task_queue.empty():
                    await asyncio.sleep(getattr(self.config, 'scheduler_loop_interval', 0.1))
                    
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)  # 错误后短暂休眠
                
        logger.info(f"Scheduler loop stopped for agent {self.agent_id}")
                
    async def _process_messages(self):
        """处理消息 - Message Handler调用Reasoner，生成任务放入队列"""
        if not self.message_handler:
            return
            
        # 处理消息队列中的消息
        processed_count = 0
        max_concurrent = getattr(self.config, 'max_concurrent_messages', 5)
        
        while not self.message_queue.empty() and processed_count < max_concurrent:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=getattr(self.config, 'message_timeout', 30)
                )
                
                # Message Handler处理消息，调用Reasoner生成任务
                tasks = await self.message_handler.process_message(message)
                
                # 将生成的任务放入任务队列
                for task in tasks:
                    await self.schedule_task(task)
                    
                processed_count += 1
                
            except asyncio.TimeoutError:
                logger.warning("Message processing timeout")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    async def _process_tasks(self):
        """处理任务 - Task Handler执行任务"""
        if not self.task_handler:
            return
            
        # 处理任务队列中的任务
        executed_count = 0
        max_concurrent = getattr(self.config, 'max_concurrent_tasks', 10)
        
        while not self.task_queue.empty() and executed_count < max_concurrent:
            try:
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=getattr(self.config, 'task_timeout', 300)
                )
                
                # Task Handler执行任务
                result = await self.task_handler.execute(task)
                
                executed_count += 1
                
            except asyncio.TimeoutError:
                logger.warning("Task execution timeout")
                break
            except Exception as e:
                logger.error(f"Error executing task: {e}")
                
    def is_running(self) -> bool:
        """检查调度器是否正在运行"""
        return self._running
