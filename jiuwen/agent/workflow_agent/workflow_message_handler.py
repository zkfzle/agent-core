#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

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
from jiuwen.core.agent.controller.controller import ControllerState
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.utils.llm.messages import HumanMessage
from jiuwen.core.agent.message.message import MessageType
from jiuwen.core.agent.controller.config.reasoner_config import IntentDetectionConfig
from jiuwen.core.agent.controller.reasoner.intent_detection import IntentDetection


_RUNTIME_STATE_KEY = "workflow_controller_state"  # Runtime 中存储状态的 key
_STATE_INTERRUPTED_TASKS_KEY = "interrupted_tasks"  # 状态中存储中断任务列表的 key
_TASK_METADATA_COMPONENT_ID_KEY = "interrupted_component_id"  # Task metadata 中存储组件ID的 key

class WorkflowMessageHandler(MessageHandler):
    """WorkflowMessageHandler - 工作流模式的消息处理器，包含Workflow状态管理"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        super().__init__(config, context_engine, runtime)
        self._state = ControllerState()  # 状态数据对象
        self.load_state()  # 从runtime加载状态

    def save_state(self):
        """将状态保存到runtime"""
        if not self._state.interrupted_tasks:
            self.runtime.update_state({_RUNTIME_STATE_KEY: None})
            logger.debug("Cleared workflow state")
            return

        # 使用 Pydantic 的序列化能力
        state_data = {
            _STATE_INTERRUPTED_TASKS_KEY: [task.model_dump() for task in self._state.interrupted_tasks]
        }
        self.runtime.update_state({_RUNTIME_STATE_KEY: state_data})
        logger.debug(f"Saved {len(self._state.interrupted_tasks)} interrupted tasks")

    def load_state(self):
        """从runtime加载状态"""
        state_data = self.runtime.get_state(_RUNTIME_STATE_KEY)
        if not state_data:
            self._state.interrupted_tasks = []
            logger.debug("No saved state found")
            return

        tasks_data = state_data.get(_STATE_INTERRUPTED_TASKS_KEY, [])

        # 使用 Pydantic 的反序列化能力
        deserialized_tasks = []
        for task_dict in tasks_data:
            try:
                task = Task.model_validate(task_dict)
                deserialized_tasks.append(task)
            except Exception as e:
                logger.error(f"Failed to deserialize task: {e}")

        self._state.interrupted_tasks = deserialized_tasks
        logger.info(f"Loaded {len(deserialized_tasks)} interrupted tasks")

    async def handle_message(self, message: Message) -> MessageHandlerResult:
        """处理工作流模式的消息，返回处理结果"""
        try:
            # 根据消息类型分发处理
            if message.msg_type == MessageType.TASK_COMPLETED:
                # 任务完成：写入流数据，决定是否停止
                return await self._handle_task_completed(message)

            elif message.msg_type == MessageType.TASK_INTERRUPTED:
                # 任务中断：写入流数据，保存状态，决定是否停止
                return await self._handle_task_interrupted(message)

            elif message.msg_type == MessageType.USER_INPUT:
                # 用户输入：根据当前状态决定是新任务还是恢复任务
                return await self._handle_user_input(message)

            elif message.msg_type == MessageType.ERROR:
                # 错误消息：发送错误流，停止工作流
                return await self._handle_error(message)

            else:
                logger.warning(f"Unsupported message type: {message.msg_type}")
                # 不支持的消息类型，继续处理下一个消息
                return MessageHandlerResult(tasks=[], should_continue=True)

        except Exception as e:
            logger.error(f"Error in WorkflowMessageHandler: {e}")
            # 发送错误流式消息
            error_result = await self._send_error_stream(str(e))
            # 返回停止信号
            return MessageHandlerResult(
                tasks=[],
                should_continue=False,
                final_result=error_result
            )

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
            self._state.add_interrupted_task(interrupted_task, component_id,
                                             component_id_key=_TASK_METADATA_COMPONENT_ID_KEY)
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
            component_id = self._state.get_interrupted_component_id(workflow_id,
                                                                    component_id_key=_TASK_METADATA_COMPONENT_ID_KEY)
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
        return self._resolve_workflow_from_tasks_generic(tasks, self.config.workflows)

    def _match_workflow(self, detected_task: Task) -> Optional[WorkflowSchema]:
        return self._match_workflow_generic(detected_task, self.config.workflows)

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

    def _add_msg_to_chat_history(self, message):
        """添加消息到聊天历史"""
        workflow_context = self.context_engine.get_workflow_context(
            workflow_id=self.config.workflows[0].id,
            session_id=self.runtime.session_id()
        )
        workflow_context.add_message(message)
