#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.agent.common.schema import WorkflowSchema
from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.agent.task.task import Task, TaskInput
from openjiuwen.core.agent.controller.scheduler.message_handler import MessageHandler, MessageHandlerResult
from openjiuwen.agent.common.enum import TaskType
from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.config.user_config import UserConfig
from openjiuwen.core.utils.llm.messages import HumanMessage
from openjiuwen.core.agent.message.message import MessageType
from openjiuwen.core.agent.controller.config.reasoner_config import IntentDetectionConfig
from openjiuwen.core.agent.controller.reasoner.intent_detection import IntentDetection
from openjiuwen.core.agent.controller.utils import MessageHandlerUtils
from openjiuwen.agent.utils import MessageUtils


class WorkflowMessageHandler(MessageHandler):
    """WorkflowMessageHandler - 工作流模式的消息处理器
    
    核心职责：通过意图识别选择 workflow 并执行
    状态管理和中断处理逻辑已提取到基类 MessageHandler
    """

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        super().__init__(config, context_engine, runtime)
        self.load_state()  # 从runtime加载状态（基类实现）

    def get_state_key(self) -> str:
        """返回workflow handler的状态存储key"""
        return "workflow_controller_state"

    async def handle_message(self, message: Message) -> MessageHandlerResult:
        """处理工作流模式的消息，返回处理结果
        
        消息分发逻辑：
        - TASK_COMPLETED: 简单完成，不做 reasoning
        - TASK_INTERRUPTED: 使用基类通用中断处理
        - USER_INPUT: 意图识别选择 workflow
        - ERROR: 使用基类通用错误处理
        """
        try:
            # 根据消息类型分发处理
            if message.msg_type == MessageType.TASK_COMPLETED:
                return await self._handle_task_completed(message)

            elif message.msg_type == MessageType.TASK_INTERRUPTED:
                # 使用基类通用中断处理
                return await self._handle_task_interrupted(message)

            elif message.msg_type == MessageType.USER_INPUT:
                return await self._handle_user_input(message)

            elif message.msg_type == MessageType.ERROR:
                # 使用基类通用错误处理
                return await self._handle_error(message)

            else:
                logger.warning(f"Unsupported message type: {message.msg_type}")
                return MessageHandlerResult(tasks=[], stop=False)

        except Exception as e:
            logger.error(f"Error in WorkflowMessageHandler: {e}")
            error_result = await self._send_error_stream(str(e))
            return MessageHandlerResult(
                tasks=[],
                stop=True,
                final_result=error_result
            )

    async def _handle_task_completed(self, message: Message) -> MessageHandlerResult:
        """处理任务完成消息 - Workflow 特有：简单完成，不做 reasoning
        
        与 React 的区别：
        - React: 任务完成后再次调用 LLM reasoning
        - Workflow: 任务完成就停止
        """
        # 写入任务完成的流数据
        await self._write_message_stream_data(message)

        # 清除该workflow的中断状态（如果有）- 使用基类状态管理
        if message.context.workflow_id:
            self._state.clear_interrupted_task(message.context.workflow_id)
            self.save_state()
            logger.info(f"Cleared interrupt state for workflow: {message.context.workflow_id}")

        # 返回停止信号，携带最终结果
        result = message.content.stream_data
        logger.info(f"Task completed, returning stop signal with final result")

        return MessageHandlerResult(
            tasks=[],
            stop=True,
            final_result=result
        )

    async def _handle_user_input(self, message: Message) -> MessageHandlerResult:
        """处理用户输入消息 - Workflow 核心：意图识别选择 workflow
        
        流程：
        - 单 workflow：判断是否中断 -> 恢复或新建
        - 多 workflow：意图识别选择 workflow -> 判断是否中断 -> 恢复或新建
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
                # 恢复中断的 workflow（使用基类方法）
                logger.info(f"Resuming interrupted workflow: {workflow.name}, task_id={interrupted_task.task_id}")
                return await self._create_resume_task(message, workflow, interrupted_task)
            else:
                # 创建新的 workflow 任务
                logger.info(f"Creating new workflow task: {workflow.name} (no interrupted task found)")
                return await self._create_new_workflow_task(message, workflow)

        # 多workflow场景：进行意图识别选择workflow
        try:
            detected_workflow = await self._select_workflow_via_intent_detection(message)
            interrupted_task = self._find_interrupted_task(detected_workflow)

            if interrupted_task:
                # 恢复中断的 workflow（使用基类方法）
                logger.info(f"Resuming interrupted workflow: {detected_workflow.name}")
                return await self._create_resume_task(message, detected_workflow, interrupted_task)
            else:
                # 创建新workflow任务
                logger.info(f"Creating new workflow task: {detected_workflow.name}")
                return await self._create_new_workflow_task(message, detected_workflow)

        except Exception as e:
            logger.error(f"Failed to detect intent: {e}, falling back to first workflow")
            return await self._create_new_workflow_task(message, workflows[0])

    async def _create_new_workflow_task(self, message: Message, workflow: WorkflowSchema) -> MessageHandlerResult:
        """创建新的workflow任务 - Workflow 特有方法
        
        使用query启动workflow，包括输入过滤和聊天历史记录。
        """
        query_text = message.content.get_query()

        # 过滤输入参数
        filtered_inputs = MessageHandlerUtils.filter_inputs(
            schema=workflow.inputs or {},
            user_data={"query": query_text}
        )

        # 添加消息到聊天历史
        user_message = HumanMessage(content=query_text)
        MessageUtils.add_workflow_message(
            user_message, workflow.id, self.context_engine, self.runtime
        )
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

        return MessageHandlerResult(
            tasks=[task],
            stop=False
        )

    async def _select_workflow_via_intent_detection(self, message: Message) -> WorkflowSchema:
        """使用意图识别选择workflow - Workflow 核心逻辑"""
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

        # 使用基类的通用方法
        workflow = self._resolve_workflow_from_tasks(detected_tasks)
        if workflow:
            return workflow

        logger.warning("Intent detection returned no matching workflow, fallback to first workflow")
        return self.config.workflows[0]

    def _ensure_intent_detection_initialized(self):
        """初始化意图识别模块 - Workflow 特有逻辑"""
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
