# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Intent recognition module

This module implements intent-based event handling, including:
- IntentRecognizer: recognizes intents from user input
- EventHandlerWithIntentRecognition: event handler with intent recognition

Workflow:
1. Receive input events
2. Use IntentRecognizer to recognize intents
3. Call the corresponding handler method based on the intent type

Supported intent types:
- CREATE_TASK: create a new task
- PAUSE_TASK: pause a task
- RESUME_TASK: resume a task
- CONTINUE_TASK: continue a task
- SUPPLEMENT_TASK: supplement task information
- CANCEL_TASK: cancel a task
- MODIFY_TASK: modify a task
- SWITCH_TASK: switch tasks
- UNKNOWN_TASK: unknown intent
"""

from abc import abstractmethod
from typing import TYPE_CHECKING

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.controller.config import ControllerConfig
from openjiuwen.core.controller.modules.event_handler import EventHandler, EventHandlerInput
from openjiuwen.core.controller.modules.task_manager import TaskManager
from openjiuwen.core.controller.schema import Intent
from openjiuwen.core.controller.schema.event import Event
from openjiuwen.core.session import Session

if TYPE_CHECKING:
    from openjiuwen.core.single_agent.agent import AbilityManager


class IntentRecognizer:
    """Intent recognizer

    Responsible for recognizing intents from user input and converting events
    into Intent objects.
    """

    def __init__(
            self,
            config: ControllerConfig,
            task_manager: TaskManager,
            ability_manager: 'AbilityManager',
            context_engine: ContextEngine
    ):
        """Initialize intent recognizer

        Args:
            config: controller configuration
            task_manager: task manager
            ability_manager: ability manager
            context_engine: context engine
        """
        self._config = config
        self._task_manager = task_manager
        self._context_engine = context_engine
        self._ability_manager = ability_manager

    async def recognize(self, event: Event, session: Session) -> Intent:
        """Recognize intent

        Args:
            event: input event
            session: session object

        Returns:
            Intent: recognized intent object
        """
        ...


class EventHandlerWithIntentRecognition(EventHandler):
    """Event handler with intent recognition

    Extends EventHandler with intent recognition and dispatches to the
    corresponding handler methods based on the recognized intent.
    """
    def __init__(self):
        super().__init__()
        self.recognizer = IntentRecognizer(
            self._config,
            self.task_manager,
            self.ability_manager,
            self.context_engine
        )

    async def handle_input(self, inputs: EventHandlerInput):
        """Handle input events

        Recognize the intent from input and call the corresponding handler. Can be overridden.

        Args:
            inputs: event handler input
        """
        ...

    async def handle_task_interaction(self, inputs: EventHandlerInput):
        """Handle task interaction events

        Pass interactions directly to the user. Can be overridden.

        Args:
            inputs: event handler input
        """
        ...

    async def handle_task_completion(self, inputs: EventHandlerInput):
        """Handle task completion events

        Pass task completion information to the user. Can be overridden.

        Args:
            inputs: event handler input
        """
        ...

    async def handle_task_failed(self, inputs: EventHandlerInput):
        """Handle task failure events

        Pass error information to the user. Can be overridden.

        Args:
            inputs: event handler input
        """
        ...

    @abstractmethod
    async def _process_create_task_intent(self, inputs: EventHandlerInput):
        """Process create-task intent

        Custom logic for executing a new task.

        Args:
            inputs: event handler input
        """
        ...

    async def _process_pause_task_intent(self, inputs: EventHandlerInput):
        """Process pause-task intent

        Interrupt the target task by calling task_scheduler.pause_task.

        Args:
            inputs: event handler input
        """
        ...

    async def _process_resume_task_intent(self, inputs: EventHandlerInput):
        """Process resume-task intent

        Set the status of the task to be resumed to submitted.

        Args:
            inputs: event handler input
        """
        ...

    async def _process_continue_task_intent(self, inputs: EventHandlerInput):
        """Process continue-task intent

        Call _process_create_task_intent with the context of the dependent task
        to execute the target task.

        Args:
            inputs: event handler input
        """
        ...

    async def _process_supplement_task_intent(self, inputs: EventHandlerInput):
        """Process supplement-task intent

        Call _process_create_task_intent based on the supplemental information
        to continue executing the target task.

        Args:
            inputs: event handler input
        """
        ...

    async def _process_cancel_task_intent(self, inputs: EventHandlerInput):
        """Process cancel-task intent

        Cancel the target task by calling task_scheduler.cancel_task.

        Args:
            inputs: event handler input
        """
        ...

    async def _process_modify_task_intent(self, inputs: EventHandlerInput):
        """Process modify-task intent

        Modify the target task and set its status to be submitted.

        Args:
            inputs: event handler input
        """
        ...

    async def _process_switch_task_intent(self, inputs: EventHandlerInput):
        """Process switch-task intent

        Interrupt all running tasks and then call _process_create_task_intent
        to execute the target task.

        Args:
            inputs: event handler input
        """
        ...

    async def _process_unknown_task_intent(self, event: Event):
        """Process unknown-task intent

        Return the clarification_prompt field of Intent to the user.

        Args:
            event: input event
        """
        ...
