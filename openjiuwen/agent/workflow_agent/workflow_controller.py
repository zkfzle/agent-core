#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Workflow Controller - Workflow-specific execution logic"""

from typing import Dict, Optional

from openjiuwen.agent.common.enum import TaskStatus, TaskType
from openjiuwen.agent.common.schema import WorkflowSchema
from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.core.agent.controller.config.reasoner_config import (
    IntentDetectionConfig
)
from openjiuwen.core.agent.controller.intent_detection_controller import (
    Intent,
    IntentDetectionController,
    IntentType
)
from openjiuwen.core.agent.controller.reasoner.agent_reasoner import AgentReasoner
from openjiuwen.core.agent.controller.reasoner.intent_detection import (
    IntentDetection
)
from openjiuwen.core.agent.message.message import Message, MessageContent
from openjiuwen.core.agent.task.task import Task, TaskInput
from openjiuwen.core.common.logging import logger
from openjiuwen.core.runner.runner import Runner, resource_mgr
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.workflow.base import WorkflowExecutionState


class WorkflowController(IntentDetectionController):
    """WorkflowController - Implements workflow-specific execution logic
    
    Core responsibilities:
    1. Intent detection: Select workflow + Check interruption state
    2. Task execution: Execute workflow (new/resume)
    3. Interruption handling: Save interruption state to runtime.state
    """

    def __init__(
            self,
            config: AgentConfig,
            context_engine,
            runtime
    ):
        """Initialize WorkflowController
        
        Args:
            config: Agent configuration
            context_engine: Context engine
            runtime: Agent-level Runtime (AgentRuntime)
        """
        super().__init__(config, context_engine, runtime)
        
        # Maintain backward compatible attribute name
        self.agent_config = config
        
        # Initialize reasoner
        self.reasoner = AgentReasoner(
            config,
            context_engine,
            None  # runtime is dynamically passed when used
        )

    async def intent_detection(
            self,
            message: Message,
            runtime: Runtime
    ) -> Intent:
        """Intent detection: Select workflow + Check interruption state
        
        Process:
        1. Get available workflows
        2. Single workflow: Use directly; Multiple workflows: LLM recognition
        3. Check for interrupted tasks
        4. Return Intent (ExecNewTask or ResumeTask)
        
        Args:
            message: Message object
            runtime: Runtime context
            
        Returns:
            Intent: Intent object
        """
        workflows = self.agent_config.workflows or []

        if not workflows:
            raise ValueError("No workflows configured for agent")

        # 1. Select workflow
        if len(workflows) == 1:
            # Single workflow: Use directly
            workflow = workflows[0]
            logger.info(f"Single workflow mode: using {workflow.name}")
        else:
            # Multiple workflows: LLM recognition
            workflow = await self._detect_workflow_via_llm(message, runtime)
            logger.info(f"Multi workflow mode: detected {workflow.name}")

        # 2. Check interruption state
        interrupted_task = self._find_interrupted_task(workflow, runtime)

        if interrupted_task:
            # Resume task
            logger.info(f"Found interrupted task for workflow {workflow.name}")
            return Intent(
                intent_type=IntentType.ResumeTask,
                task=interrupted_task,
                workflow=workflow
            )
        else:
            # New task
            logger.info(f"Creating new task for workflow {workflow.name}")
            new_task = self._create_new_task(message, workflow)
            return Intent(
                intent_type=IntentType.ExecNewTask,
                task=new_task,
                workflow=workflow
            )

    async def exec_task(
            self,
            message_content: MessageContent,
            task: Task,
            runtime: Runtime
    ) -> Dict:
        """Execute workflow task
        
        Execution method depends on task.status:
        - PENDING: New task, use Runner.run_workflow
        - INTERRUPTED: Resume task, use Runner.run_workflow with InteractiveInput
        
        Args:
            message_content: Message content
            task: Task object
            runtime: Runtime context
            
        Returns:
            dict: Execution result
        """
        workflow_id = task.input.target_id

        try:
            # Update task status to running
            task.status = TaskStatus.RUNNING

            # Get workflow object (from controller's agent)
            workflow = self._find_workflow_from_agent(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")

            # Create workflow runtime
            workflow_runtime = runtime.create_workflow_runtime()

            # Prepare input parameters
            inputs = task.input.arguments

            # If resuming task, parameters should already be InteractiveInput (set in _handle_resume)
            if task.status == TaskStatus.INTERRUPTED:
                logger.info(f"Resuming workflow: {workflow_id}, inputs type={type(inputs)}")
                # Parameters should already be InteractiveInput, use directly
            else:
                logger.info(f"Starting workflow: {workflow_id}")

            # Execute workflow
            result = await Runner.run_workflow(
                workflow,
                inputs=inputs,
                runtime=workflow_runtime
            )

            # Check if interrupted
            is_interrupted = self._is_workflow_interrupted(result)
            result_state = "NO STATE"
            if hasattr(result, 'state'):
                result_state = result.state
            logger.info(
                f"Workflow result state: {result_state}, "
                f"interrupted: {is_interrupted}"
            )

            if is_interrupted:
                # Workflow interrupted
                logger.info(f"Workflow interrupted: {workflow_id}")
                task.status = TaskStatus.INTERRUPTED
                
                # Extract interaction list from result
                interaction_data = result.result if hasattr(result, 'result') else None
                await self.interrupt_task(task, runtime, interaction_data)

                # Return interruption response (interaction request)
                return result.result  # Return interaction list
            else:
                # Workflow completed
                logger.info(f"Workflow completed: {workflow_id}")
                task.status = TaskStatus.SUCCESS

                # Clean up interruption state (if any)
                self._clear_interrupted_state(task, runtime)

                # Return completion response
                payload = {"output": result, "result_type": "answer"}
                return payload

        except Exception as e:
            # Execution failed
            logger.error(f"Workflow execution failed: {workflow_id}, error: {e}")
            task.status = TaskStatus.FAILED
            raise

    async def interrupt_task(
            self,
            task: Task,
            runtime: Runtime,
            interaction_data: Optional[list] = None
    ) -> Dict:
        """Interrupt workflow task
        
        Save interruption state to runtime.state
        
        Args:
            task: Task object
            runtime: Runtime context
            interaction_data: Interaction data during interruption (OutputSchema list)
            
        Returns:
            dict: Interruption information
        """
        workflow_id = task.input.target_id

        # 1. Update task status
        task.status = TaskStatus.INTERRUPTED

        # 2. Save interruption state to runtime.state
        state = runtime.get_state("workflow_controller") or {}
        if "interrupted_tasks" not in state:
            state["interrupted_tasks"] = {}

        # Extract component ID from interaction data
        component_id = self._extract_component_id_from_interaction_data(
            interaction_data
        )
        state_key = workflow_id.replace('.', '_')

        state["interrupted_tasks"][state_key] = {
            "task": task.model_dump(),
            "component_id": component_id
        }

        runtime.update_state({"workflow_controller": state})

        logger.info(
            f"Task interrupted: workflow={workflow_id}, "
            f"state_key={state_key}, component_id={component_id}"
        )

        return {
            "status": "interrupted",
            "task_id": task.task_id,
            "workflow_id": workflow_id,
            "message": "Task interrupted, waiting for subsequent input"
        }

    async def _detect_workflow_via_llm(
            self,
            message: Message,
            runtime: Runtime
    ) -> WorkflowSchema:
        """Use LLM to detect workflow
        
        If reasoner exists and model is configured, call reasoner for intent detection
        Otherwise return the first workflow
        
        Args:
            message: Message object
            runtime: Runtime context
            
        Returns:
            WorkflowSchema: Detected workflow schema
        """
        # If no reasoner, return first workflow
        if not self.reasoner:
            logger.warning("No reasoner configured, using first workflow")
            return self.agent_config.workflows[0]

        try:
            # Initialize intent detection module (pass runtime)
            self._ensure_intent_detection_initialized(runtime)

            # Call reasoner to detect intent
            detected_tasks = await self.reasoner.use_intent_detection(message)

            if not detected_tasks:
                logger.warning("Intent detection returned no tasks, using first workflow")
                return self.agent_config.workflows[0]

            # Extract workflow from detected_tasks
            # detected_tasks is a Task list
            # task.input.target_name contains workflow name
            workflow_name = detected_tasks[0].input.target_name

            # Match workflow
            for workflow in self.agent_config.workflows:
                if workflow.name == workflow_name:
                    return workflow

            # If not found, return first workflow
            logger.warning(
                f"Workflow '{workflow_name}' not found, "
                "using first workflow"
            )
            return self.agent_config.workflows[0]

        except Exception as e:
            logger.error(f"Intent detection failed: {e}, using first workflow")
            return self.agent_config.workflows[0]

    def _ensure_intent_detection_initialized(self, runtime: Runtime):
        """Initialize intent detection module
        
        If reasoner already has intent_detection_module, update runtime
        Otherwise create new IntentDetection instance
        
        Args:
            runtime: Runtime context
        """
        if not self.reasoner:
            return

        # If already initialized, update runtime
        has_intent_module = (hasattr(self.reasoner, 'intent_detection_module') and
                             self.reasoner.intent_detection_module)
        if has_intent_module:
            self.reasoner.intent_detection_module.runtime = runtime
            logger.debug("Updated intent detection runtime")
            return

        category_names = [workflow.name for workflow in self.agent_config.workflows]
        intent_config = IntentDetectionConfig(
            category_list=category_names,
            category_info="\n".join(
                f"- {w.name}: {w.description}"
                for w in self.agent_config.workflows
            ),
            enable_history=True,
            enable_input=True,
        )

        intent_detection = IntentDetection(
            intent_config=intent_config,
            agent_config=self.agent_config,
            context_engine=self._context_engine,
            runtime=runtime  # Pass runtime
        )

        self.reasoner.set_intent_detection(intent_detection)
        logger.info("Intent detection module initialized")

    def _find_interrupted_task(
            self,
            workflow: WorkflowSchema,
            runtime: Runtime
    ) -> Optional[Task]:
        """Find interrupted task for specified workflow
        
        Find interrupted task from runtime.state:
        state["workflow_controller"]["interrupted_tasks"][workflow_id]
        
        Note: Since update_dict uses '.' as nested path separator,
        '.' is replaced with '_' when saving, so we need to replace when finding too
        """
        state = runtime.get_state("workflow_controller")
        logger.info(f"_find_interrupted_task: workflow={workflow.name}, state={state}")
        if not state:
            logger.info("_find_interrupted_task: No workflow_controller state found")
            return None

        interrupted_tasks = state.get("interrupted_tasks", {})
        logger.info(f"_find_interrupted_task: interrupted_tasks keys={list(interrupted_tasks.keys())}")

        base_id_with_version = f"{workflow.id}_{workflow.version.replace('.', '_')}"
        possible_ids = [
            base_id_with_version,  # weather_flow_1_0
            workflow.id  # weather_flow
        ]
        logger.info(f"_find_interrupted_task: possible_ids={possible_ids}")

        for workflow_id in possible_ids:
            if workflow_id in interrupted_tasks:
                logger.info(f"_find_interrupted_task: Found interrupted task for {workflow_id}")
                task_data = interrupted_tasks[workflow_id]["task"]
                return Task.model_validate(task_data)

        logger.info(f"_find_interrupted_task: No interrupted task found for workflow {workflow.name}")
        return None

    def _create_new_task(
            self,
            message: Message,
            workflow: WorkflowSchema
    ) -> Task:
        """Create new workflow task
        
        Extract query from message and filter parameters based on workflow.inputs
        """
        # Get query
        query = message.content.get_query() if hasattr(message.content, 'get_query') else ""

        # Filter input parameters
        filtered_inputs = self._filter_workflow_inputs(
            workflow.inputs or {},
            {"query": query}
        )

        logger.info(f"Creating task with inputs: {filtered_inputs}, query: {query}")

        # Create task
        task = Task(
            task_id=f"workflow_{message.msg_id}",
            task_type=TaskType.WORKFLOW,
            status=TaskStatus.PENDING,
            input=TaskInput(
                target_name=workflow.name,
                target_id=f"{workflow.id}_{workflow.version}",
                arguments=filtered_inputs
            )
        )

        return task

    def _filter_workflow_inputs(
            self,
            schema: Dict,
            user_data: Dict
    ) -> Dict:
        """Filter input parameters based on schema
        
        Supports two formats:
        1. Standard JSON Schema: {"type": "object", "properties": {...}}
        2. Simplified format: {"query": {"type": "string"}}
        """
        filtered = {}

        # Try to get properties (standard format)
        properties = schema.get("properties", {})

        # If no properties field, but schema itself contains field definitions (simplified format)
        if not properties and schema:
            # Check if it's simplified format (directly contains field definitions)
            if any(isinstance(v, dict) and "type" in v for v in schema.values()):
                properties = schema

        for key, value in user_data.items():
            if key in properties or not properties:
                # If key is in properties, or properties is empty (allow all fields)
                filtered[key] = value

        return filtered

    def _get_interrupted_component_id(self, task: Task, runtime: Runtime) -> str:
        """Get component ID at interruption
        
        Find component_id of interrupted task from runtime.state
        """
        state = runtime.get_state("workflow_controller")
        if not state:
            return "questioner"  # Default value

        workflow_id = task.input.target_id
        # Replace '.' with '_', consistent with saving
        state_key = workflow_id.replace('.', '_')

        interrupted_info = state.get("interrupted_tasks", {}).get(state_key)
        if interrupted_info:
            return interrupted_info.get("component_id", "questioner")

        return "questioner"

    def _clear_interrupted_state(self, task: Task, runtime: Runtime):
        """Clean up interruption state
        
        Remove interrupted task for specified workflow from runtime.state
        """
        state = runtime.get_state("workflow_controller") or {}
        interrupted_tasks = state.get("interrupted_tasks", {})

        workflow_id = task.input.target_id
        # Replace '.' with '_', consistent with saving
        state_key = workflow_id.replace('.', '_')

        if state_key in interrupted_tasks:
            del interrupted_tasks[state_key]
            runtime.update_state({"workflow_controller": state})
            logger.info(f"Cleared interrupted state for workflow: {workflow_id}, state_key: {state_key}")

    def _extract_component_id_from_interaction_data(
            self,
            interaction_data: Optional[list]
    ) -> str:
        """Extract component ID from interaction data
        
        Reference old implementation: MessageHandler.extract_component_id_from_stream_data
        Find OutputSchema with type '__interaction__' from interaction_data list,
        and extract component_id from payload.id
        
        Args:
            interaction_data: OutputSchema list, containing interaction requests during interruption
            
        Returns:
            str: Component ID, default return "questioner"
        """
        if not interaction_data:
            logger.warning("No interaction_data provided, using default component_id")
            return "questioner"
        
        try:
            # Iterate through interaction_data, find output with type '__interaction__'
            for output_schema in interaction_data:
                if (hasattr(output_schema, 'type') and 
                    output_schema.type == '__interaction__'):
                    # Extract InteractionOutput.id from payload
                    if (hasattr(output_schema, 'payload') and 
                        hasattr(output_schema.payload, 'id')):
                        component_id = output_schema.payload.id
                        logger.info(
                            f"Extracted component_id from interaction_data: "
                            f"{component_id}"
                        )
                        return component_id
        except Exception as e:
            logger.warning(
                f"Failed to extract component_id from interaction_data: {e}"
            )
        
        logger.warning("No component_id found in interaction_data, using default")
        return "questioner"  # Default value

    def _find_workflow_from_agent(self, workflow_id: str):
        """Find workflow object from runtime
        
        Args:
            workflow_id: workflow ID (format: {id}_{version})
            
        Returns:
            Workflow object, None if not found
        """
        # First try to find from Runner's global resource_mgr
        try:
            logger.info(f"Trying to find workflow from resource_mgr: {workflow_id}")
            # List all available workflows
            all_workflows = resource_mgr.workflow()._resources
            logger.info(f"Available workflows in resource_mgr: {list(all_workflows.keys())}")

            workflow = resource_mgr.workflow().get_workflow(workflow_id)
            logger.info(f"Found workflow from resource_mgr: {workflow is not None}")
            if workflow:
                return workflow
        except Exception as e:
            logger.warning(f"Failed to find workflow from resource_mgr {workflow_id}: {e}")

        # Then try to get from controller's _runtime
        try:
            logger.info(f"Trying to find workflow from controller._runtime: {workflow_id}")
            workflow = self._runtime.get_workflow(workflow_id)
            logger.info(f"Found workflow from controller._runtime: {workflow is not None}")
            return workflow
        except Exception as e:
            logger.error(f"Failed to find workflow from controller._runtime {workflow_id}: {e}")

        logger.error(f"Workflow not found: {workflow_id}")
        return None

    def _find_workflow_by_id(self, workflow_id: str, runtime: Runtime):
        """Find workflow object from runtime
        
        Args:
            workflow_id: workflow ID (format: {id}_{version})
            runtime: Runtime context
            
        Returns:
            Workflow object, None if not found
        """
        try:
            workflow = runtime.get_workflow(workflow_id)
            return workflow
        except Exception as e:
            logger.error(f"Failed to find workflow {workflow_id}: {e}")
            return None

    def _is_workflow_interrupted(self, result) -> bool:
        """Check if workflow is interrupted
        
        Args:
            result: WorkflowOutput object
            
        Returns:
            True if interrupted, False otherwise
        """
        if not result:
            return False

        # Check state (compare enum value)
        if hasattr(result, 'state'):
            return result.state == WorkflowExecutionState.INPUT_REQUIRED

        return False
