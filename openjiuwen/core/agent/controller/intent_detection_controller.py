#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Intent Detection Controller - Intent detection and task management"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from openjiuwen.core.agent.controller.controller import BaseController
from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.agent.task.task import Task, TaskStatus
from openjiuwen.core.common.logging import logger
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.runtime.interaction.interactive_input import InteractiveInput


class IntentType(Enum):
    """Intent type"""
    ExecNewTask = "exec_new_task"  # Execute new task
    ResumeTask = "resume_task"  # Resume task
    CancelTask = "cancel_task"  # Cancel task
    Unknown = "unknown"  # Unknown intent


@dataclass
class Intent:
    """Intent object - Encapsulates intent detection result"""
    intent_type: IntentType  # Intent type
    task: Optional[Task] = None  # Associated task object
    workflow: Optional[Any] = None  # Selected workflow (use Any to avoid circular import)
    metadata: Dict[str, Any] = None  # Extended metadata

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IntentDetectionController(BaseController):
    """Intent Detection Controller - Provides task management and intent routing capabilities
    
    Core responsibilities:
    1. Intent detection: Recognize user intent through intent_detection()
    2. Message routing: Route to different handlers based on Intent type
    3. Task execution: Call exec_task() to execute tasks
    4. Interruption handling: Call interrupt_task() to handle interruptions
    """

    def __init__(self, config, context_engine, runtime):
        """Initialize Intent Detection Controller
        
        Args:
            config: Agent configuration
            context_engine: Context engine
            runtime: Agent-level Runtime (AgentRuntime)
        """
        super().__init__(config, context_engine, runtime)

    async def handle_message(self, message: Message, runtime: Runtime) -> Dict:
        """Standard message processing flow: Intent detection -> Route processing
        
        Args:
            message: Message object
            runtime: Runtime context
            
        Returns:
            Processing result
        """
        # 1. Intent detection
        intent = await self.intent_detection(message, runtime)

        # 2. Route processing based on intent type
        if intent.intent_type == IntentType.ExecNewTask:
            result = await self._handle_new_task(message, intent, runtime)
        elif intent.intent_type == IntentType.ResumeTask:
            result = await self._handle_resume(message, intent, runtime)
        elif intent.intent_type == IntentType.CancelTask:
            result = await self._handle_cancel(message, intent, runtime)
        else:
            result = await self._handle_unknown_intent(message, intent, runtime)

        return result

    async def _handle_new_task(
            self,
            message: Message,
            intent: Intent,
            runtime: Runtime
    ) -> Dict:
        """Handle new task: Update state -> Execute
        
        Uses synchronous execution mode because:
        - Users expect invoke() to return results
        - Simplifies implementation, avoids extra queue waiting
        
        Args:
            message: Message object
            intent: Intent object
            runtime: Runtime context
            
        Returns:
            dict: Execution result
        """
        task = intent.task
        if not task:
            return {"status": "error", "message": "Task not found in intent"}

        task.status = TaskStatus.PENDING

        # Execute task directly
        logger.info(f"Handling new task: task_id={task.task_id}")
        result = await self.exec_task(message.content, task, runtime)
        return result

    async def _handle_resume(
            self,
            message: Message,
            intent: Intent,
            runtime: Runtime
    ) -> Dict:
        """Handle task resumption: Update input -> Execute
        
        Key: Must create InteractiveInput with new user input to update task parameters
        
        Args:
            message: Message object
            intent: Intent object
            runtime: Runtime context
            
        Returns:
            dict: Execution result
        """
        task = intent.task
        if not task:
            return {"status": "error", "message": "Task not found in intent"}

        # Task status should already be INTERRUPTED
        if task.status != TaskStatus.INTERRUPTED:
            logger.warning(
                f"Resuming task with unexpected status: {task.status}"
            )

        logger.info(f"Handling resume task: task_id={task.task_id}")

        # Create InteractiveInput from user input
        if hasattr(message.content, 'query'):
            query_text = message.content.query
        else:
            query_text = ""
        interactive_input = InteractiveInput()

        # Get component ID at interruption from workflow_controller
        workflow_id = task.input.target_id
        state = runtime.get_state("workflow_controller")
        component_id = "questioner"  # Default value

        if state:
            state_key = workflow_id.replace('.', '_')
            interrupted_tasks = state.get("interrupted_tasks", {})
            interrupted_info = interrupted_tasks.get(state_key)
            if interrupted_info:
                component_id = interrupted_info.get(
                    "component_id",
                    "questioner"
                )

        interactive_input.update(component_id, query_text)
        logger.info(
            f"Created InteractiveInput for resume: "
            f"component_id={component_id}, query={query_text}"
        )

        # Key: Update task input parameters to InteractiveInput
        task.input.arguments = interactive_input

        # Execute task (resume)
        result = await self.exec_task(message.content, task, runtime)
        return result

    async def _handle_cancel(
            self,
            message: Message,
            intent: Intent,
            runtime: Runtime
    ) -> Dict:
        """Handle task cancellation
        
        Args:
            message: Message object
            intent: Intent object
            runtime: Runtime context
            
        Returns:
            dict: Cancellation result
        """
        task = intent.task
        if not task:
            return {"status": "error", "message": "Task not found in intent"}

        task.status = TaskStatus.CANCELLED

        logger.info(f"Handling cancel task: task_id={task.task_id}")

        return {"status": "cancelled", "task_id": task.task_id}

    async def _handle_unknown_intent(
            self,
            message: Message,
            intent: Intent,
            runtime: Runtime
    ) -> Dict:
        """Handle unknown intent
        
        Args:
            message: Message object
            intent: Intent object
            runtime: Runtime context
            
        Returns:
            dict: Error result
        """
        logger.warning(f"Unknown intent type: {intent.intent_type}")
        return {
            "status": "error",
            "message": f"Unknown intent type: {intent.intent_type}"
        }

    # ===== Abstract methods (subclasses must implement) =====

    @abstractmethod
    async def intent_detection(
            self,
            message: Message,
            runtime: Runtime
    ) -> Intent:
        """Intent detection (subclasses must implement)
        
        Subclasses need to implement:
        - Select execution target (e.g. workflow, tool)
        - Check interruption state
        - Return Intent object
        
        Args:
            message: Message object
            runtime: Runtime context
            
        Returns:
            Intent object
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement intent_detection()"
        )

    @abstractmethod
    async def exec_task(
            self,
            message_content: Any,
            task: Task,
            runtime: Runtime
    ) -> Dict:
        """Execute task (subclasses must implement)
        
        Subclasses need to implement:
        - Decide execution method based on task.status (new/resume)
        - Call runtime to execute workflow/tool
        - Handle execution results and exceptions
        
        Args:
            message_content: Message content
            task: Task object
            runtime: Runtime context
            
        Returns:
            Execution result
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement exec_task()"
        )

    @abstractmethod
    async def interrupt_task(
            self,
            task: Task,
            runtime: Runtime
    ) -> Dict:
        """Interrupt task (subclasses must implement)
        
        Subclasses need to implement:
        - Save interruption state
        - Return interruption information
        
        Args:
            task: Task object
            runtime: Runtime context
            
        Returns:
            Interruption information
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement interrupt_task()"
        )
