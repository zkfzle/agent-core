#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Workflow Controller - Workflow-specific execution logic"""

import asyncio
from typing import Dict, Optional

from openjiuwen.agent.common.enum import TaskStatus, TaskType
from openjiuwen.agent.common.schema import WorkflowSchema
from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.agent.utils import MessageUtils
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
from openjiuwen.core.common.constants.constant import INTERACTION
from openjiuwen.core.common.logging import logger
from openjiuwen.core.runner.runner import Runner, resource_mgr
from openjiuwen.core.runtime.interaction.interaction import InteractionOutput
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.utils.llm.messages import AIMessage
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
            config: AgentConfig = None,
            context_engine=None,
            runtime=None
    ):
        """Initialize WorkflowController
        
        Args:
            config: Agent configuration (optional, can be injected later)
            context_engine: Context engine (optional, can be injected later)
            runtime: Agent-level Runtime (optional, can be injected later)
            
        Note:
            If parameters are not provided, they will be injected by
            ControllerAgent via setup_from_agent()
        """
        super().__init__(config, context_engine, runtime)
        
        # Maintain backward compatible attribute name
        self.agent_config = config
        
        # Initialize reasoner (only if config and context_engine are available)
        self.reasoner = None
        if config is not None and context_engine is not None:
            self._init_reasoner()
    
    def _init_reasoner(self):
        """Initialize reasoner - can be called after setup_from_agent"""
        if self._config is not None and self._context_engine is not None:
            self.reasoner = AgentReasoner(
                self._config,
                self._context_engine,
                None  # runtime is dynamically passed when used
            )
            # Update backward compatible reference
            self.agent_config = self._config
    
    def setup_from_agent(self, agent):
        """Override to also initialize reasoner after setup"""
        super().setup_from_agent(agent)
        # Initialize reasoner after base setup
        self._init_reasoner()

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
        4. Check if user wants to switch workflow (if detected workflow != interrupted workflow)
        5. Return Intent (ExecNewTask or ResumeTask)
        
        Args:
            message: Message object
            runtime: Runtime context
            
        Returns:
            Intent: Intent object
        """
        workflows = self.agent_config.workflows or []

        if not workflows:
            raise ValueError("No workflows configured for agent")

        # 1. Select workflow based on user's current query
        if len(workflows) == 1:
            # Single workflow: Use directly
            detected_workflow = workflows[0]
            logger.info(f"Single workflow mode: using {detected_workflow.name}")
        else:
            # Multiple workflows: LLM recognition
            detected_workflow = await self._detect_workflow_via_llm(
                message, runtime
            )
            logger.info(
                f"Multi workflow mode: detected {detected_workflow.name}"
            )

        # 2. Check if detected workflow has an interrupted task
        interrupted_task = self._find_interrupted_task(
            detected_workflow, runtime
        )

        if interrupted_task:
            # Found interrupted task for this workflow: Resume
            logger.info(
                f"Found interrupted task for workflow "
                f"{detected_workflow.name}, resuming"
            )
            return Intent(
                intent_type=IntentType.ResumeTask,
                task=interrupted_task,
                workflow=detected_workflow
            )
        else:
            # No interrupted task for this workflow: Create new task
            # Note: Other workflows' interrupted states are preserved
            logger.info(
                f"No interrupted task for workflow {detected_workflow.name}, "
                f"creating new task"
            )
            new_task = self._create_new_task(message, detected_workflow)
            return Intent(
                intent_type=IntentType.ExecNewTask,
                task=new_task,
                workflow=detected_workflow
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
        conversation_id = runtime.session_id()

        try:
            # 1. Check if there's a running task for this conversation
            if self.task_queue.has_running_task(conversation_id):
                # Cancel old task
                cancelled = await self.task_queue.cancel_running_task(
                    conversation_id
                )
                if cancelled:
                    logger.info(
                        f"Cancelled previous running task for "
                        f"conversation: {conversation_id}"
                    )
                    # Clear old task's interrupted state
                    old_info = self.task_queue.find_task(conversation_id)
                    if old_info:
                        # Create a temporary task object for cleanup
                        temp_task = Task(
                            task_id=old_info.task.task_id,
                            task_type=old_info.task.task_type,
                            status=TaskStatus.CANCELLED,
                            input=TaskInput(
                                target_id=old_info.target_id,
                                target_name="",
                                arguments={}
                            )
                        )
                        self._clear_interrupted_state(temp_task, runtime)
            
            # 2. Prepare execution (existing code)
            task.status = TaskStatus.RUNNING

            # Get workflow object (from controller's agent)
            workflow = self._find_workflow_from_agent(workflow_id, runtime)
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")

            # Create workflow runtime
            workflow_runtime = runtime.create_workflow_runtime()

            # Prepare input parameters
            inputs = task.input.arguments

            # If resuming task, parameters should already be InteractiveInput
            if task.status == TaskStatus.INTERRUPTED:
                logger.info(
                    f"Resuming workflow: {workflow_id}, "
                    f"inputs type={type(inputs)}"
                )
            else:
                logger.info(f"Starting workflow: {workflow_id}")

            # 3. 使用流式调用 workflow，这样 workflow 层可以 yield:
            #    - tracer_workflow (执行跟踪)
            #    - __interaction__ (中断请求)
            #    - workflow_final (完成结果)
            # 流式数据会写入 runtime，agent 层的 stream_iterator 可以读取
            async def run_workflow_streaming():
                from openjiuwen.core.workflow.base import (
                    WorkflowOutput, WorkflowExecutionState
                )
                from openjiuwen.core.stream.base import OutputSchema
                workflow_stream = await Runner.run_workflow_streaming(
                    workflow,
                    inputs=inputs,
                    runtime=workflow_runtime,
                    context=self._context_engine.get_workflow_context(
                        session_id=conversation_id, workflow_id=workflow_id
                    )
                )
                chunks = []
                has_interaction = False
                final_result = None
                async for chunk in workflow_stream:
                    # 检查 chunk 类型
                    if isinstance(chunk, OutputSchema):
                        if chunk.type == INTERACTION:
                            has_interaction = True
                            # 不在这里透传 __interaction__
                            # 由上层 ControllerAgent.stream 在 controller.invoke 完成后统一写入
                            # 确保 __interaction__ 在所有 tracer 事件之后
                        elif chunk.type == "workflow_final":
                            # 不透传原始的 workflow_final
                            # 后面会构造正确格式的 workflow_final
                            final_result = chunk.payload
                        else:
                            # 透传其他流式数据（tracer 等）
                            await runtime.write_stream(chunk)
                    else:
                        await runtime.write_stream(chunk)
                    chunks.append(chunk)

                # add messages to context
                if chunks:
                    content_parts = []
                    for chunk in chunks:
                        if isinstance(chunk, OutputSchema):
                            if isinstance(chunk.payload, dict):
                                answer = chunk.payload.get("answer", "")
                                if answer is not None:
                                    content_parts.append(str(answer))
                            elif isinstance(chunk.payload, InteractionOutput):
                                # 保留交互中断的输出内容
                                content_parts.append(str(chunk.payload.value) if chunk.payload.value else "")
                    workflow_content = "".join(content_parts)
                    MessageUtils.add_ai_message(AIMessage(content=workflow_content), self._context_engine, runtime)

                # 构造 WorkflowOutput
                if has_interaction:
                    return WorkflowOutput(
                        result=chunks,
                        state=WorkflowExecutionState.INPUT_REQUIRED
                    )
                else:
                    return WorkflowOutput(
                        result=final_result,
                        state=WorkflowExecutionState.COMPLETED
                    )

            workflow_task = asyncio.create_task(run_workflow_streaming())

            # 4. Register task to queue
            await self.task_queue.register_task(
                conversation_id, task, workflow_task, target_id=workflow_id
            )
            
            # 5. Wait for task completion (may be cancelled)
            try:
                result = await workflow_task
            except asyncio.CancelledError:
                logger.info(f"Workflow cancelled: {workflow_id}")
                task.status = TaskStatus.CANCELLED
                return {
                    "status": "cancelled",
                    "task_id": task.task_id,
                    "workflow_id": workflow_id
                }
            finally:
                # 6. Unregister task
                await self.task_queue.unregister_task(conversation_id)

            # 7. Process result (existing code)
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
                interaction_data = (
                    result.result if hasattr(result, 'result') else None
                )
                await self.interrupt_task(task, runtime, interaction_data)

                # Return interruption response (interaction request)
                return result.result  # Return interaction list
            else:
                # Workflow completed
                logger.info(f"Workflow completed: {workflow_id}")
                task.status = TaskStatus.SUCCESS

                # Clean up interruption state (if any)
                self._clear_interrupted_state(task, runtime)

                # 写入 workflow_final 到流
                from openjiuwen.core.stream.base import OutputSchema
                payload = {"output": result, "result_type": "answer"}
                final_output = OutputSchema(
                    type="workflow_final",
                    index=0,
                    payload=payload
                )
                await runtime.write_stream(final_output)

                # Return completion response
                return payload

        except asyncio.CancelledError:
            # Task was cancelled
            logger.info(f"Task cancelled during execution: {workflow_id}")
            task.status = TaskStatus.CANCELLED
            await self.task_queue.unregister_task(conversation_id)
            return {
                "status": "cancelled",
                "task_id": task.task_id,
                "workflow_id": workflow_id
            }
        except Exception as e:
            # Execution failed
            logger.error(
                f"Workflow execution failed: {workflow_id}, error: {e}"
            )
            task.status = TaskStatus.FAILED
            await self.task_queue.unregister_task(conversation_id)
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

        # 优先使用 description 作为分类，语义更丰富；如果没有配置则回退到 name
        category_list = [
            workflow.description if workflow.description else workflow.name
            for workflow in self.agent_config.workflows
        ]
        intent_config = IntentDetectionConfig(
            category_list=category_list,
            category_info="\n".join(
                f"- {w.description if w.description else w.name}"
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

    def _find_workflow_from_agent(self, workflow_id: str, runtime: Runtime):
        """Find workflow object from runtime
        
        Args:
            workflow_id: workflow ID (format: {id}_{version})
            runtime: Task Runtime context
            
        Returns:
            Workflow object, None if not found
        """
        # First try to find from Runner's global resource_mgr
        try:
            logger.info(f"Trying to find workflow from resource_mgr: {workflow_id}")
            # List all available workflows
            all_workflows = resource_mgr.workflow()._resources
            logger.info(f"Available workflows in resource_mgr: {list(all_workflows.keys())}")

            workflow = resource_mgr.workflow().get_workflow(workflow_id, runtime.base())
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
