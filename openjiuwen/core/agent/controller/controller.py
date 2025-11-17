#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""Controller of Agent"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional

from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.core.agent.controller.reasoner.agent_reasoner import AgentReasoner
from openjiuwen.core.agent.controller.scheduler import (
    AgentScheduler,
    MessageHandler,
    TaskHandler
)
from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.common.logging import logger
from openjiuwen.core.context_engine.engine import ContextEngine
from openjiuwen.core.runner.message_queue_base import InvokeQueueMessage
from openjiuwen.core.runner.message_queue_inmemory import MessageQueueInMemory
from openjiuwen.core.runtime.runtime import Runtime


class Controller:
    """Controller
    
    Responsible for coordinating MessageHandler, TaskHandler and AgentScheduler
    Handles message flow and task scheduling
    """

    def __init__(
            self,
            config: AgentConfig,
            context_engine: ContextEngine,
            runtime: Runtime,
            message_handler: MessageHandler
    ):
        self._config = config
        self._context_engine = context_engine
        self._runtime = runtime
        self._agent_handler = None

        # Unified component initialization
        self._scheduler = AgentScheduler(config)
        self._reasoner = AgentReasoner(config, context_engine, runtime)
        self._task_handler = TaskHandler(config, context_engine, runtime)

        # Directly use the passed MessageHandler (contains its own state management)
        self._message_handler = message_handler

        # Set component references
        self._scheduler.set_handlers(self._message_handler, self._task_handler)
        self._message_handler.set_reasoner(self._reasoner)

    async def start(self):
        """Start controller - Start scheduler"""
        await self._scheduler.start()

    async def stop(self):
        """Stop controller - Stop scheduler"""
        await self._scheduler.stop()

    async def receive_message(self, message: Message):
        """Receive message - External systems call this interface, unified message entry"""
        # All external messages are uniformly placed in AgentScheduler's message queue for unified scheduling
        await self._scheduler.schedule_message(message)

    async def run_until_complete(self):
        """Wait for scheduler to complete and return final result"""
        return await self._scheduler.run_until_complete()

    async def process_inputs(self, inputs: dict):
        """Process inputs and wait for completion - Unified invocation entry
        
        Args:
            inputs: Input dictionary, contains query and conversation_id
            
        Returns:
            dict: Final result (returned from MessageHandler's final_result)
        """
        # 1. Create message
        session_id = inputs.get("conversation_id", "default_session")
        message = Message.create_user_message(
            content=inputs.get("query", ""),
            conversation_id=session_id
        )

        # 2. Send message to scheduler
        await self._scheduler.schedule_message(message)

        # 3. Wait for scheduler to complete and return result
        return await self._scheduler.run_until_complete()


class BaseController(ABC):
    """Message queue based Controller
    """

    def __init__(
            self,
            config: AgentConfig,
            context_engine: ContextEngine,
            runtime: Runtime
    ):
        """Initialize BaseController
        
        Args:
            config: Agent configuration
            context_engine: Context engine
            runtime: Agent-level Runtime (AgentRuntime)
        """
        # Hold core dependencies (consistent with Controller)
        self._config = config
        self._context_engine = context_engine
        self._runtime = runtime
        
        # Create message queue
        self.msg_queue = MessageQueueInMemory()
        self.msg_queue.start()

        # Subscribe to topic
        self.topic = "controller_messages"
        subscription = self.msg_queue.subscribe(self.topic)
        subscription.set_message_handler(self._handle_message_wrapper)
        subscription.activate()

    async def invoke(self, inputs: Dict, runtime: Runtime) -> Dict:
        """Synchronous invocation entry

        Process:
        1. Create message
        2. Publish message to queue (produce_message)
        3. Wait for processing result
        """
        # 1. Create message
        message = self.create_message(inputs)

        # 2. Create queue message and publish
        queue_message = InvokeQueueMessage()
        queue_message.payload = {"message": message, "runtime": runtime}
        queue_message.response = asyncio.Future()

        # 3. Publish to message queue
        await self.msg_queue.produce_message(self.topic, queue_message)

        # 4. Wait for result
        result = await queue_message.response

        return result if result is not None else {"output": "processed"}

    async def _handle_message_wrapper(self, request: Dict) -> Dict:
        """Message processing wrapper - Automatically called by message queue

        Args:
            request: Dictionary containing message and runtime
            
        Returns:
            dict: Processing result
        """
        message = request["message"]
        runtime = request["runtime"]
        try:
            result = await self.handle_message(message, runtime)
            result_type = type(result)
            has_result = result is not None
            logger.info(
                f"BaseController: handle_message returned: "
                f"{result_type}, {has_result}"
            )
            return result
        except Exception as e:
            error_msg = f"BaseController: handle_message raised exception: {e}"
            logger.error(error_msg, exc_info=True)
            raise

    # ===== Abstract methods (developers must implement) =====
    @abstractmethod
    async def handle_message(self, message: Message, runtime: Runtime) -> Optional[Dict]:
        """Core method for message processing (must be implemented)

        Args:
            message: Message object
            runtime: Runtime context
        Returns:
            Optional[Dict]: Processing result

        Developers implement all business logic here:
        - Dispatch processing based on message.msg_type
        - Execute tasks (synchronously or asynchronously)
        - Manage state
        - Handle interruptions
        - If multi-round processing is needed, loop inside this method
        """
        pass

    # ===== Extension methods (developers can optionally override) =====
    def create_message(self, inputs: Dict) -> Message:
        """Create message object (can be overridden)

        Default: Extract content/query and metadata from inputs, create user input message
        """
        # Support both content and query field names (backward compatible)
        content = inputs.get("content") or inputs.get("query", "")
        conversation_id = inputs.get("conversation_id", "default_session")

        return Message.create_user_message(
            content=content,
            conversation_id=conversation_id
        )

    def stop(self):
        """Stop controller - Clean up resources"""
        self.msg_queue.unsubscribe(self.topic)
        self.msg_queue.stop()
