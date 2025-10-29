#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from jiuwen.agent.config.base import AgentConfig
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.core.agent.message.message import Message
from jiuwen.core.agent.task.task import Task, TaskInput
from jiuwen.core.agent.controller.scheduler.message_handler import MessageHandler, MessageHandlerResult
from jiuwen.agent.common.enum import TaskType
from jiuwen.core.common.logging import logger
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.utils.llm.messages import HumanMessage
from jiuwen.core.agent.message.message import MessageType
from jiuwen.core.agent.controller.config.reasoner_config import IntentDetectionConfig
from jiuwen.core.agent.controller.reasoner.intent_detection import IntentDetection


_RUNTIME_STATE_KEY = "workflow_controller_state"  # Runtime 中存储状态的 key
_STATE_INTERRUPTED_TASKS_KEY = "interrupted_tasks"  # 状态中存储中断任务列表的 key
_TASK_METADATA_COMPONENT_ID_KEY = "interrupted_component_id"  # Task metadata 中存储组件ID的 key


@dataclass
class WorkflowControllerState:
    """
    Workflow Controller 状态数据类 - 支持多个workflow的中断状态。
    状态结构：
    [
        task1,  # task1.input.target_id = "workflow_id_1"
                # task1.metadata["interrupted_component_id"] = "questioner"
        task2,  # task2.input.target_id = "workflow_id_2"
        ...
    ]
    """
    interrupted_tasks: List[Task] = field(default_factory=list)
    
    def is_interrupted(self) -> bool:
        """检查是否有任何中断的任务"""
        return len(self.interrupted_tasks) > 0
    
    def get_interrupted_task(self, workflow_id: str = None) -> Optional[Task]:
        """
        获取中断的任务。
        
        Args:
            workflow_id: 可选，指定workflow ID。如果不指定，返回第一个任务
        
        Returns:
            Optional[Task]: 中断的任务，如果不存在则返回None
        """
        if not self.interrupted_tasks:
            return None
        
        if workflow_id:
            # 遍历查找匹配的 workflow_id
            for task in self.interrupted_tasks:
                if task.input and task.input.target_id:
                    if task.input.target_id == workflow_id:
                        return task
            return None
        else:
            # 返回第一个任务 todo改成先进后出
            return self.interrupted_tasks[0] if self.interrupted_tasks else None
    
    def get_interrupted_component_id(self, workflow_id: str = None) -> Optional[str]:
        """
        从 Task 的 metadata 中获取中断时的组件ID。
        
        Args:
            workflow_id: 可选，指定workflow ID。如果不指定，使用第一个任务
        
        Returns:
            Optional[str]: 中断时的组件ID
        """
        task = self.get_interrupted_task(workflow_id)
        if task and task.metadata:
            return task.metadata.get(_TASK_METADATA_COMPONENT_ID_KEY)
        return None
    
    def add_interrupted_task(self, task: Task, component_id: Optional[str] = None):
        """
        添加中断的任务。
        
        将 component_id 存入 task.metadata，然后添加到列表。
        如果该 workflow 已有中断任务，则替换它。
        """
        if not task:
            return
        
        # 将 component_id 存入 Task 的 metadata
        if component_id:
            if not task.metadata:
                task.metadata = {}
            task.metadata[_TASK_METADATA_COMPONENT_ID_KEY] = component_id
        
        # 检查是否已存在该 workflow 的中断任务，如果有则替换
        workflow_id = task.input.target_id if task.input else None
        if workflow_id:
            # 移除旧的同 workflow 任务
            self.interrupted_tasks = [
                t for t in self.interrupted_tasks 
                if not (t.input and t.input.target_id == workflow_id)
            ]
        
        # 添加新任务
        self.interrupted_tasks.append(task)
    
    def clear_interrupted_task(self, workflow_id: str):
        """清除指定workflow的中断任务"""
        self.interrupted_tasks = [
            task for task in self.interrupted_tasks
            if not (task.input and task.input.target_id == workflow_id)
        ]
    
    def clear_all(self):
        """清除所有中断任务"""
        self.interrupted_tasks.clear()


class WorkflowMessageHandler(MessageHandler):
    """WorkflowMessageHandler - 工作流模式的消息处理器，包含Workflow状态管理"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        super().__init__(config, context_engine, runtime)
        self._state = WorkflowControllerState()  # 状态数据对象
        self.load_state()  # 从runtime加载状态

    async def _handle_task_completed(self, message: Message) -> MessageHandlerResult:
        """处理任务完成消息：写入流数据，清除该workflow的中断状态，返回停止信号"""
        # 写入任务完成的流数据
        await self._write_message_stream_data(message)

        # 清除该workflow的中断状态（如果有）
        if message.context.workflow_id:
            self._state.clear_interrupted_task(message.context.workflow_id)
            self.save_state()
            logger.info(f"Cleared interrupt state for workflow: {message.context.workflow_id}")

        # 返回停止信号，携带最终结果
        result = message.content.stream_data
        logger.info(f"Task completed, returning stop signal with final result")
        
        return MessageHandlerResult(
            tasks=[],
            should_continue=False,
            final_result=result
        )

    async def _handle_task_interrupted(self, message: Message) -> MessageHandlerResult:
        """处理任务中断消息：写入流数据，保存状态，返回停止信号"""
        # 写入任务中断的流数据
        await self._write_message_stream_data(message)

        # 从 extensions 中获取中断的 Task 对象（用于恢复）
        interrupted_task = message.content.extensions.get("interrupted_task")
        if interrupted_task:
            # 从 stream_data 中提取交互组件ID
            component_id = self._extract_component_id_from_stream_data(message.content.stream_data)
            
            workflow_id = interrupted_task.input.target_id if interrupted_task.input else "unknown"
            logger.info(f"Saving interrupt state: workflow={workflow_id}, task={interrupted_task.task_id}, component_id={component_id}")
            
            # 保存中断任务（component_id 会存入 task.metadata）
            self._state.add_interrupted_task(interrupted_task, component_id)
            self.save_state()
            logger.info(f"Task {message.context.task_id} interrupted and saved")

        # 返回停止信号，携带中断结果（交互请求列表）
        result = message.content.stream_data
        logger.info(f"Task interrupted, returning stop signal with interaction requests")
        
        return MessageHandlerResult(
            tasks=[],
            should_continue=False,
            final_result=result
        )

    @staticmethod
    def _extract_component_id_from_stream_data(stream_data: List) -> Optional[str]:
        """
        从 stream_data 中提取交互组件ID。

        Args:
            stream_data: 流数据列表 (List[OutputSchema])

        Returns:
            Optional[str]: 交互组件的ID，如果找不到则返回None
        """
        if not stream_data:
            return None

        try:
            # 遍历 stream_data，找到类型为 '__interaction__' 的输出
            for output_schema in stream_data:
                if hasattr(output_schema, 'type') and output_schema.type == '__interaction__':
                    # 从 payload 中提取 InteractionOutput.id
                    if hasattr(output_schema, 'payload') and hasattr(output_schema.payload, 'id'):
                        component_id = output_schema.payload.id
                        logger.debug(f"Extracted component_id from stream_data: {component_id}")
                        return component_id
        except Exception as e:
            logger.warning(f"Failed to extract component_id from stream_data: {e}")

        return None

    async def _handle_error(self, message: Message) -> MessageHandlerResult:
        """处理错误消息：发送错误流，返回停止信号"""
        error_msg = message.content.query if message.content.query else "Unknown error"
        logger.error(f"Received error message: {error_msg}")

        # 发送错误流式消息
        error_result = await self._send_error_stream(str(error_msg))
        
        # 返回停止信号，携带错误结果
        logger.info(f"Error occurred, returning stop signal with error result")
        
        return MessageHandlerResult(
            tasks=[],
            should_continue=False,
            final_result=error_result
        )

    async def _handle_user_input(self, message: Message) -> MessageHandlerResult:
        """
        处理用户输入消息。
        
        核心逻辑：
        - 单workflow：判断该workflow是否中断 -> 恢复或新建
        - 多workflow：通过意图识别选择workflow -> 判断是否中断 -> 恢复或新建
        """
        workflows = self.config.workflows or []
        if not workflows:
            raise ValueError("No workflows configured for agent")
        
        # 单workflow场景：简单判断是否中断
        if len(workflows) == 1:
            workflow = workflows[0]
            task_count = len(self._state.interrupted_tasks)
            logger.info(f"Single workflow mode: workflow={workflow.name}, "
                        f"is_interrupted={self._state.is_interrupted()}, "
                        f"interrupted_tasks_count={task_count}")
            interrupted_task = self._find_interrupted_task(workflow)
            
            if interrupted_task:
                # 该workflow处于中断状态，恢复它
                logger.info(f"Resuming interrupted workflow: {workflow.name}, task_id={interrupted_task.task_id}")
                return await self._create_resume_task(message, workflow, interrupted_task)
            else:
                # 该workflow不是中断状态，创建新任务
                logger.info(f"Creating new workflow task: {workflow.name} (no interrupted task found)")
                return await self._create_new_workflow_task(message, workflow)
        
        # 多workflow场景：进行意图识别选择workflow
        try:
            detected_workflow = await self._select_workflow_via_intent_detection(message)
            interrupted_task = self._find_interrupted_task(detected_workflow)
            
            if interrupted_task:
                # 意图识别的workflow处于中断状态，恢复它
                logger.info(f"Resuming interrupted workflow: {detected_workflow.name}")
                return await self._create_resume_task(message, detected_workflow, interrupted_task)
            else:
                # 创建新workflow任务
                logger.info(f"Creating new workflow task: {detected_workflow.name}")
                return await self._create_new_workflow_task(message, detected_workflow)
            
        except Exception as e:
            logger.error(f"Failed to detect intent: {e}, falling back to first workflow")
            return await self._create_new_workflow_task(message, workflows[0])
    
    def _find_interrupted_task(self, workflow: WorkflowSchema) -> Optional[Task]:
        """
        查找指定workflow的中断任务。
        
        尝试多种ID格式匹配，返回找到的第一个中断任务，如果没有则返回None。
        """
        if not self._state.is_interrupted():
            return None
        
        # 尝试多种可能的workflow_id格式
        possible_ids = [
            f"{workflow.id}_{workflow.version}",  # test_interrupt_workflow_1.0
            f"{workflow.id}_{workflow.version.replace('.', '_')}",  # test_interrupt_workflow_1_0
            workflow.id  # test_interrupt_workflow
        ]
        
        # 精确匹配
        for workflow_id in possible_ids:
            task = self._state.get_interrupted_task(workflow_id)
            if task:
                logger.debug(f"Found interrupted task with id={workflow_id}")
                return task
        
        # 模糊匹配（兜底） - 遍历所有中断任务
        for task in self._state.interrupted_tasks:
            if task.input and task.input.target_id and workflow.id in task.input.target_id:
                logger.debug(f"Fuzzy matched interrupted task: task_id={task.task_id}")
                return task
        
        return None

    async def _create_new_workflow_task(self, message: Message, workflow: WorkflowSchema) -> MessageHandlerResult:
        """
        创建新的workflow任务（使用query启动workflow）。
        
        Args:
            message: 用户消息
            workflow: 选中的workflow schema
            
        Returns:
            MessageHandlerResult: 包含任务的处理结果
        """
        query_text = message.content.get_query()
        
        # 过滤输入参数
        filtered_inputs = self._filter_inputs(
            schema=workflow.inputs or {},
            user_data={"query": query_text}
        )
        
        # 添加消息到聊天历史
        user_message = HumanMessage(content=query_text)
        self._add_msg_to_chat_history(user_message)
        if UserConfig.is_sensitive():
            logger.info(f"Added user message to chat history")
        else:
            logger.info(f"Added user message to chat history: {query_text}")
        
        # 创建任务
        task = Task(
            task_id=f"workflow_{message.msg_id}",
            task_type=TaskType.WORKFLOW,
            input=TaskInput(
                target_name=workflow.name,
                target_id=f"{workflow.id}_{workflow.version}",
                arguments=filtered_inputs,
            )
        )
        
        logger.info(f"Created new workflow task: {workflow.name}, task_id: {task.task_id}")
        
        # 返回任务，继续调度
        return MessageHandlerResult(
            tasks=[task],
            should_continue=True
        )
    
    async def _create_resume_task(self, message: Message, workflow: WorkflowSchema, interrupted_task: Task) -> MessageHandlerResult:
        """
        创建恢复任务（将query封装为InteractiveInput恢复中断的workflow）。
        
        Args:
            message: 用户消息
            workflow: 选中的workflow schema
            interrupted_task: 中断的任务对象
            
        Returns:
            MessageHandlerResult: 包含任务的处理结果
        """
        # 检查消息内容中是否已经有 InteractiveInput
        if message.content.interactive_input is not None:
            # 直接使用已有的 InteractiveInput（保留兼容性，但不推荐）
            interactive_input = message.content.interactive_input
            logger.warning(f"Received InteractiveInput directly (not recommended), using it for resuming workflow: "
                           f"{workflow.name}")
        else:
            # 从 query 文本创建 InteractiveInput（推荐方式）
            query_text = message.content.get_query()
            interactive_input = InteractiveInput()
            
            # 从状态中获取中断时的组件ID
            workflow_id = interrupted_task.input.target_id if interrupted_task.input else None
            component_id = self._extract_interrupted_component_id(workflow_id) or "questioner"
            
            interactive_input.update(component_id, query_text)
            logger.info(f"Created InteractiveInput for resuming workflow: {workflow.name}, "
                        f"component_id: {component_id}, query: {query_text}")
        
        # 更新中断任务的参数
        interrupted_task.input.arguments = interactive_input
        interrupted_task.task_id = f"workflow_{message.msg_id}"  # 更新task_id
        
        logger.info(f"Resuming workflow task: {workflow.name}, task_id: {interrupted_task.task_id}")
        
        # 返回恢复的任务，继续调度
        return MessageHandlerResult(
            tasks=[interrupted_task],
            should_continue=True
        )
    
    def _extract_interrupted_component_id(self, workflow_id: Optional[str] = None) -> Optional[str]:
        """
        从保存的状态中提取中断时的组件ID。
        
        Args:
            workflow_id: 可选，指定workflow ID。如果不指定，使用第一个中断的workflow
            
        Returns:
            Optional[str]: 中断时的组件ID，如果找不到则返回None
        """
        try:
            # 直接从状态中获取组件ID（状态已经在 load_state 时加载）
            component_id = self._state.get_interrupted_component_id(workflow_id)
            if component_id:
                logger.debug(f"Retrieved component_id from state: {component_id}")
            return component_id
        except Exception as e:
            logger.debug(f"Failed to extract component_id from state: {e}")
            return None
    
    async def _select_workflow_task(self, message: Message) -> Task:
        """选择工作流任务 - 返回Task而不是字符串"""
        workflows = self.config.workflows or []
        if not workflows:
            raise ValueError("No workflows configured for agent")

        if len(workflows) == 1:
            workflow = workflows[0]
        else:
            workflow = await self._select_workflow_via_intent_detection(message)

        filtered_inputs = self._filter_inputs(
            schema=workflow.inputs or {},
            user_data={"query": message.content.get_query()}
        )

        task = Task(
            task_id=f"workflow_{message.msg_id}",
            task_type=TaskType.WORKFLOW,
            input=TaskInput(
                target_name=workflow.name,
                target_id=f"{workflow.id}_{workflow.version}",
                arguments=filtered_inputs,
            )
        )

        return task

    async def _select_workflow_via_intent_detection(self, message: Message) -> WorkflowSchema:
        if not self.reasoner:
            logger.warning("Reasoner not set, fallback to first workflow")
            return self.config.workflows[0]

        try:
            self._ensure_intent_detection_initialized()
            detected_tasks = await self.reasoner.use_intent_detection(message)
        except Exception as exc:
            logger.error(f"Intent detection failed: {exc}")
            return self.config.workflows[0]

        detected_tasks = detected_tasks or []

        workflow = self._resolve_workflow_from_detected_tasks(detected_tasks)
        if workflow:
            return workflow

        logger.warning("Intent detection returned no matching workflow, fallback to first workflow")
        return self.config.workflows[0]

    def _ensure_intent_detection_initialized(self):
        if not self.config.workflows or len(self.config.workflows) < 2:
            return

        if not self.reasoner:
            logger.warning("Reasoner not ready, skip intent detection initialization")
            return

        if self.reasoner.intent_detection_module:
            return

        if not getattr(self.config, "model", None):
            logger.warning("Model config missing, cannot initialize intent detection")
            return

        category_names = [workflow.name for workflow in self.config.workflows]
        intent_config = IntentDetectionConfig(
            category_list=category_names,
            category_info="\n".join(category_names),
            enable_history=True,
            enable_input=True,
        )

        intent_detection = IntentDetection(
            intent_config=intent_config,
            agent_config=self.config,
            context_engine=self.context_engine,
            runtime=self.runtime
        )

        self.reasoner.set_intent_detection(intent_detection)

    def _resolve_workflow_from_detected_tasks(self, tasks: List[Task]) -> Optional[WorkflowSchema]:
        for task in tasks:
            workflow = self._match_workflow(task)
            if workflow:
                return workflow
        return None

    def _match_workflow(self, detected_task: Task) -> Optional[WorkflowSchema]:
        if not detected_task or not detected_task.input:
            return None

        target_id = detected_task.input.target_id
        target_name = detected_task.input.target_name

        for workflow in self.config.workflows:
            workflow_full_id = f"{workflow.id}_{workflow.version}" if workflow.version else workflow.id
            if target_id and target_id in {workflow.id, workflow_full_id}:
                return workflow
            if target_name and workflow.name == target_name:
                return workflow

        return None

    async def _handle_normal_message_from_message(self, message: Message) -> MessageHandlerResult:
        """处理正常消息，生成工作流任务"""
        task = await self._select_workflow_task(message)

        # 添加消息到聊天历史 - 使用统一的 get_query() 方法
        query_text = message.content.get_query()
        user_message = HumanMessage(content=query_text)
        self._add_msg_to_chat_history(user_message)

        if UserConfig.is_sensitive():
            logger.info(f"Added user message to chat history")
        else:
            logger.info(f"Added user message to chat history: {query_text}")

        # 返回任务，继续调度
        return MessageHandlerResult(
            tasks=[task],
            should_continue=True
        )

    async def _handle_interrupted_message_from_message(self, message: Message) -> MessageHandlerResult:
        """
        处理中断恢复的消息。
        
        支持多个workflow的中断恢复：
        1. 如果是InteractiveInput，从中提取workflow信息
        2. 如果是普通query，尝试匹配当前中断的workflow
        """
        # 检查是否有中断的workflows
        if not self._state.is_interrupted():
            return await self._handle_normal_message_from_message(message)
        
        # 获取第一个中断的任务（简化处理）
        task = self._state.get_interrupted_task()
        if not task:
            logger.warning(f"No interrupted task found")
            return await self._handle_normal_message_from_message(message)
        
        # 更新任务的输入参数
        if message.content.interactive_input:
            task.input.arguments = message.content.interactive_input
        else:
            task.input.arguments = message.content.get_query()
        
        workflow_id = task.input.target_id if task.input else "unknown"
        logger.info(f"Resuming interrupted workflow: {workflow_id}")
        
        # 返回恢复的任务，继续调度
        return MessageHandlerResult(
            tasks=[task],
            should_continue=True
        )

    def _validate_inputs(self, inputs: Dict):
        """验证输入参数"""
        if isinstance(inputs.get("query"), InteractiveInput):
            raise JiuWenBaseException(5000, "Non-interrupt status data format error.")

    @staticmethod
    def _filter_inputs(schema: dict, user_data: dict) -> dict:
        """过滤和验证用户输入"""
        if not schema:
            return {}

        required_fields = {
            k for k, v in schema.items()
            if isinstance(v, dict) and v.get("required") is True
        }

        filtered = {}
        for k in schema:
            if k not in user_data:
                if k in required_fields:
                    raise KeyError(f"missing required parameter: {k}")
                continue
            filtered[k] = user_data[k]

        return filtered

    async def _write_message_stream_data(self, message: Message):
        """写入消息携带的流数据 - 统一处理列表"""
        # stream_data 是 List[OutputSchema]
        stream_data = message.content.stream_data
        
        try:
            # 直接遍历列表，空列表会自动跳过
            for output_schema in stream_data:
                await self.runtime.write_stream(output_schema)
                logger.debug(f"Wrote stream data from message {message.msg_id}")

        except Exception as e:
            logger.warning(f"Failed to write message stream data: {e}")

    async def _send_error_stream(self, error_message: str):
        """发送错误流式消息并返回错误结果"""
        try:
            error_stream = OutputSchema(
                type="workflow_final",
                index=0,
                payload={
                    "error": True,
                    "message": error_message,
                    "status": "failed"
                }
            )
            await self.runtime.write_stream(error_stream)
            return error_stream
        except Exception as e:
            logger.error(f"Failed to send error stream: {e}")
            # 返回一个默认错误结果
            return OutputSchema(
                type="workflow_final",
                index=0,
                payload={
                    "error": True,
                    "message": str(e),
                    "status": "failed"
                }
            )

    def _add_msg_to_chat_history(self, message):
        """添加消息到聊天历史"""
        workflow_context = self.context_engine.get_workflow_context(
            workflow_id=self.config.workflows[0].id,
            session_id=self.runtime.session_id()
        )
        workflow_context.add_message(message)
