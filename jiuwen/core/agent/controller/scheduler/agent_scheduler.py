#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import asyncio
from typing import Optional
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.message.message import Message
from jiuwen.core.agent.controller.scheduler.message_handler import MessageHandler
from jiuwen.core.agent.controller.scheduler.task_handler import TaskHandler
from jiuwen.core.common.logging import logger


class AgentScheduler:
    """AgentScheduler - 纯调度器，管理两个队列"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.id
        
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
        
        # 结果收集
        self._final_result = None
        
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
        self._final_result = None
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
    
    async def run_until_complete(self):
        """运行调度器直到完成，返回最终结果"""
        if self._scheduler_task:
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        return self._final_result
    
    def get_result(self):
        """获取最终结果（同步方法）"""
        return self._final_result
            
    async def schedule_message(self, message: Message):
        """调度消息 - 统一的消息调度入口"""
        try:
            await self.message_queue.put(message)
            logger.info(f"Message {message.msg_id} scheduled to queue")
        except asyncio.QueueFull:
            logger.error(f"Message queue is full, dropping message {message.msg_id}")
            
    async def schedule_task(self, task: Task):
        """调度任务 - 统一的任务调度入口"""
        try:
            await self.task_queue.put(task)
            logger.info(f"Task {task.task_id} scheduled to queue")
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
                
                # 3. 检查是否应该停止
                if not self._running:
                    logger.info(f"Scheduler stopped")
                    break
                
                # 4. 如果队列都为空，短暂休眠
                if self.message_queue.empty() and self.task_queue.empty():
                    await asyncio.sleep(getattr(self.config, 'scheduler_loop_interval', 0.1))
                    
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)
                
        logger.info(f"Scheduler loop stopped for agent {self.agent_id}")
                
    async def _process_messages(self):
        """处理消息 - Message Handler处理消息，根据返回结果决定是否停止"""
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
                
                # Message Handler处理消息，返回处理结果
                result = await self.message_handler.process_message(message)
                
                # 检查是否应该停止调度器
                if not result.should_continue:
                    self._final_result = result.final_result
                    self._running = False
                    logger.info(f"Scheduler stopping: message handler returned should_continue=False")
                    return
                
                # 将生成的任务放入任务队列
                for task in result.tasks:
                    await self.schedule_task(task)
                
                logger.info(f"Message {message.msg_id} processed successfully, generated {len(result.tasks)} tasks")
                processed_count += 1
                
            except asyncio.TimeoutError:
                logger.warning("Message processing timeout")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    async def _process_tasks(self):
        """
        处理任务 - 支持并行执行和依赖管理的可扩展架构
        
        工作流程：
        1. 从队列收集待处理任务
        2. 筛选可执行的任务（扩展点：可重写 _select_executable_tasks）
        3. 不可执行的任务重新入队
        4. 并行执行所有可执行任务
        5. 将结果消息放回消息队列
        """
        if not self.task_handler:
            return
        
        max_concurrent = getattr(self.config, 'max_concurrent_tasks', 10)
        
        # 1. 从队列收集任务
        pending_tasks = self._collect_pending_tasks(max_concurrent)
        if not pending_tasks:
            return
        
        # 2. 筛选可执行的任务（扩展点）
        executable_tasks = self._select_executable_tasks(pending_tasks)
        
        # 3. 不可执行的任务重新入队
        non_executable = [t for t in pending_tasks if t not in executable_tasks]
        for task in non_executable:
            await self.task_queue.put(task)
            logger.info(f"Task {task.task_id} not ready, re-queued for later execution")
        
        if not executable_tasks:
            logger.info("No executable tasks in this cycle")
            return
        
        # 4. 并行执行所有可执行任务
        logger.info(f"Executing {len(executable_tasks)} tasks concurrently")
        results = await self._execute_tasks_concurrently(executable_tasks)
        
        # 5. 将结果消息放回消息队列
        for result in results:
            if isinstance(result, Message):
                await self.schedule_message(result)
    
    def _collect_pending_tasks(self, max_count: int) -> list[Task]:
        """
        从队列收集待处理任务
        
        Args:
            max_count: 最多收集的任务数量
            
        Returns:
            收集到的任务列表
        """
        tasks = []
        while not self.task_queue.empty() and len(tasks) < max_count:
            try:
                task = self.task_queue.get_nowait()
                tasks.append(task)
            except asyncio.QueueEmpty:
                break
        
        if tasks:
            logger.info(f"Collected {len(tasks)} pending tasks from queue")
        return tasks
    
    def _select_executable_tasks(self, tasks: list[Task]) -> list[Task]:
        """
        筛选可执行的任务（扩展点）
        
        默认实现：所有任务都可执行（无依赖检查）
        子类可以重写此方法实现自定义逻辑，例如：
        - 检查前置任务是否完成
        - 检查资源是否可用
        - 检查优先级
        - 任务去重
        
        示例子类实现：
        ```python
        class DependencyAwareScheduler(AgentScheduler):
            def __init__(self, config):
                super().__init__(config)
                self.completed_tasks = set()
            
            def _select_executable_tasks(self, tasks: list[Task]) -> list[Task]:
                return [task for task in tasks if self._is_ready_to_execute(task)]
            
            def _is_ready_to_execute(self, task: Task) -> bool:
                dependencies = getattr(task, 'dependencies', [])
                if not dependencies:
                    return True
                return all(dep_id in self.completed_tasks for dep_id in dependencies)
        ```
        
        Args:
            tasks: 待筛选的任务列表
            
        Returns:
            可立即执行的任务列表
        """
        return [task for task in tasks if self._is_ready_to_execute(task)]
    
    def _is_ready_to_execute(self, task: Task) -> bool:
        """
        判断单个任务是否可以执行（扩展点）
        
        默认实现：总是返回 True（无依赖检查）
        子类可以重写此方法实现依赖检查逻辑
        
        Args:
            task: 待检查的任务
            
        Returns:
            True 如果任务可以执行，False 如果需要等待
        """
        return True
    
    async def _execute_tasks_concurrently(self, tasks: list[Task]) -> list:
        """
        并行执行任务，错误隔离
        
        关键特性：
        - 使用 asyncio.gather 实现真正的并发执行
        - return_exceptions=True 保证单个任务失败不影响其他任务
        - 每个任务都有独立的错误处理
        
        Args:
            tasks: 待执行的任务列表
            
        Returns:
            执行结果列表（Message 对象或异常）
        """
        async def _execute_with_error_handling(task: Task):
            """单个任务的执行包装器，提供错误隔离"""
            try:
                logger.info(f"Task {task.task_id} execution started")
                result = await self.task_handler.execute(task)
                logger.info(f"Task {task.task_id} execution completed successfully")
                return result
            except Exception as e:
                logger.error(f"Task {task.task_id} execution failed: {e}")
                # 返回错误消息而不是抛出异常，保证其他任务继续执行
                return Message.create_error_message(
                    conversation_id=getattr(self.task_handler.runtime, 'session_id', lambda: "")() if self.task_handler.runtime else "",
                    error_msg=f"Task {task.task_id} execution failed: {str(e)}"
                )
        
        # 并行执行所有任务
        results = await asyncio.gather(
            *[_execute_with_error_handling(task) for task in tasks],
            return_exceptions=True
        )
        
        # 过滤掉未处理的异常（理论上不应该发生，因为已经在内部捕获了）
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Unhandled exception in task execution: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
                
    def is_running(self) -> bool:
        """检查调度器是否正在运行"""
        return self._running
