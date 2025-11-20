#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any
from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.core.agent.task.task import Task
from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.agent.controller.reasoner.agent_reasoner import AgentReasoner
from openjiuwen.core.agent.controller.state import ControllerState
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.utlis.hash_util import generate_key
from openjiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from openjiuwen.core.stream.base import OutputSchema
from openjiuwen.agent.common.schema import WorkflowSchema
from openjiuwen.core.runtime.interaction.interactive_input import InteractiveInput


@dataclass
class MessageHandlerResult:
    """消息处理结果 - 统一的返回结构
    
    Attributes:
        tasks: 生成的任务列表
        stop: 是否停止调度循环（True=停止，False=继续）
        final_result: 最终结果（当 stop=True 时使用）
    """
    tasks: List[Task]
    stop: bool = False
    final_result: Optional[Any] = None


class MessageHandler(ABC):
    """MessageHandler - 消息处理器基类，包含抽象方法，支持自定义实现
    
    包含通用的状态管理和中断处理逻辑，消除子类重复代码。
    """

    # 状态管理常量
    _STATE_INTERRUPTED_TASKS_KEY = "interrupted_tasks"
    _TASK_METADATA_COMPONENT_ID_KEY = "interrupted_component_id"

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        self.config = config
        self.context_engine = context_engine
        self.runtime = runtime
        self.reasoner: Optional[AgentReasoner] = None

        # 状态管理 - 所有 handler 共享相同的状态管理逻辑
        self._state = ControllerState()

    def set_reasoner(self, reasoner: AgentReasoner):
        """设置决策器引用"""
        self.reasoner = reasoner

    @abstractmethod
    def get_state_key(self) -> str:
        """返回此handler在runtime中存储状态的key
        
        子类必须实现此方法，返回唯一的状态存储key。
        
        Examples:
            - WorkflowMessageHandler: "workflow_controller_state"
            - ReActMessageHandler: "react_controller_state"
        
        Returns:
            str: 状态存储的唯一key
        """
        pass

    async def process_message(self, message: Message) -> MessageHandlerResult:
        """处理消息，返回处理结果 - 统一的消息处理入口"""
        try:
            # 1. 消息预处理
            processed_message = await self.preprocess_message(message)

            # 2. 自定义消息处理逻辑（子类实现）
            result = await self.handle_message(processed_message)

            # 3. 任务后处理
            processed_tasks = []
            for task in result.tasks:
                processed_task = await self.postprocess_task(task)
                processed_tasks.append(processed_task)

            return MessageHandlerResult(
                tasks=processed_tasks,
                stop=result.stop,
                final_result=result.final_result
            )

        except Exception as e:
            logger.error(f"Error processing message {message.msg_id}: {e}")
            # 出错时停止调度
            return MessageHandlerResult(tasks=[], stop=True, final_result=None)

    @abstractmethod
    async def handle_message(self, message: Message) -> MessageHandlerResult:
        """消息处理逻辑 - 子类必须实现此方法，返回 MessageHandlerResult"""
        pass

    async def preprocess_message(self, message: Message) -> Message:
        """消息预处理 - 子类可重写"""
        return message

    async def postprocess_task(self, task: Task) -> Task:
        """任务后处理 - 子类可重写"""
        return task

    async def _send_final_stream(self, msg):
        """发送最终结果流并返回 OutputSchema"""
        try:
            payload = {"output": msg, "result_type": "answer"}
            final_stream = OutputSchema(
                type="answer",
                index=0,
                payload=payload
            )
            await self.runtime.write_stream(final_stream)
            return final_stream
        except Exception as e:
            logger.error(f"Failed to send final stream: {e}")
            return OutputSchema(
                type="workflow_final",
                index=0,
                payload={
                    "error": True,
                    "message": str(e),
                    "status": "failed"
                }
            )

    async def _send_error_stream(self, error_message: str):
        """发送错误结果流并返回 OutputSchema"""
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
            return OutputSchema(
                type="workflow_final",
                index=0,
                payload={
                    "error": True,
                    "message": str(e),
                    "status": "failed"
                }
            )

    async def _write_message_stream_data(self, message: Message):
        """写入消息携带的流数据 List[OutputSchema]"""
        stream_data = message.content.stream_data
        try:
            for output_schema in stream_data:
                await self.runtime.write_stream(output_schema)
                logger.debug(f"Wrote stream data from message {message.msg_id}")
        except Exception as e:
            logger.warning(f"Failed to write message stream data: {e}")

    async def _handle_error(self, message: Message) -> "MessageHandlerResult":
        """处理错误消息：发送错误流并返回停止信号"""
        error_msg = message.content.query if message.content.query else "Unknown error"
        logger.error(f"Received error message: {error_msg}")
        error_result = await self._send_error_stream(str(error_msg))
        return MessageHandlerResult(
            tasks=[],
            stop=True,
            final_result=error_result
        )

    def _get_model(self):
        """根据模型配置获取或创建（并缓存到 runtime）模型实例"""
        model_id = generate_key(
            self.config.model.model_info.api_key,
            self.config.model.model_info.api_base,
            self.config.model.model_provider
        )

        model = self.runtime.get_model(model_id=model_id)
        if model is None:
            model = ModelFactory().get_model(
                model_provider=self.config.model.model_provider,
                api_base=self.config.model.model_info.api_base,
                api_key=self.config.model.model_info.api_key,
                timeout=self.config.model.model_info.timeout
            )
            self.runtime.add_model(model_id=model_id, model=model)

        return self.runtime.get_model(model_id=model_id)

    @staticmethod
    def extract_component_id_from_stream_data(stream_data: List) -> Optional[str]:
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

    @staticmethod
    def _match_workflow_generic(detected_task: Task, workflows: List[WorkflowSchema]) -> Optional[WorkflowSchema]:
        """通用的 workflow 匹配：根据 task.input 的 id/name 在给定 workflows 中找匹配项"""
        if not detected_task or not detected_task.input:
            return None
        target_id = detected_task.input.target_id
        target_name = detected_task.input.target_name
        for workflow in workflows:
            workflow_full_id = f"{workflow.id}_{workflow.version}" if workflow.version else workflow.id
            if target_id and target_id in {workflow.id, workflow_full_id}:
                return workflow
            if target_name and workflow.name == target_name:
                return workflow
        return None

    @classmethod
    def _resolve_workflow_from_tasks_generic(cls, tasks: List[Task], workflows: List[WorkflowSchema]) -> Optional[
        WorkflowSchema]:
        """遍历 tasks，使用通用匹配逻辑返回第一个匹配的 workflow"""
        for task in tasks or []:
            workflow = cls._match_workflow_generic(task, workflows)
            if workflow:
                return workflow
        return None

    # ==================== 状态管理通用方法 ====================

    def save_state(self):
        """将状态保存到runtime"""
        if not self._state.interrupted_tasks:
            self.runtime.update_state({self.get_state_key(): None})
            logger.debug(f"Cleared state for {self.__class__.__name__}")
            return

        # 使用 Pydantic 的序列化能力
        state_data = {
            self._STATE_INTERRUPTED_TASKS_KEY: [task.model_dump() for task in self._state.interrupted_tasks]
        }
        self.runtime.update_state({self.get_state_key(): state_data})
        logger.debug(f"Saved {len(self._state.interrupted_tasks)} interrupted tasks for {self.__class__.__name__}")

    def load_state(self):
        """从runtime加载状态 - 通用实现，消除子类重复"""
        state_data = self.runtime.get_state(self.get_state_key())
        if not state_data:
            self._state.interrupted_tasks = []
            logger.debug(f"No saved state found for {self.__class__.__name__}")
            return

        tasks_data = state_data.get(self._STATE_INTERRUPTED_TASKS_KEY, [])

        # 使用 Pydantic 的反序列化能力
        deserialized_tasks = []
        for task_dict in tasks_data:
            try:
                task = Task.model_validate(task_dict)
                deserialized_tasks.append(task)
            except Exception as e:
                logger.error(f"Failed to deserialize task: {e}")

        self._state.interrupted_tasks = deserialized_tasks
        logger.info(f"Loaded {len(deserialized_tasks)} interrupted tasks for {self.__class__.__name__}")

    # ==================== 中断处理通用方法 ====================

    async def _handle_task_interrupted(self, message: Message) -> MessageHandlerResult:
        """处理任务中断消息 - 通用实现，消除子类重复
        
        工作流程：
        1. 写入任务中断的流数据
        2. 从 extensions 中获取中断的 Task 对象
        3. 提取交互组件ID并保存到状态
        4. 返回停止信号，携带中断结果
        """
        # 写入任务中断的流数据
        await self._write_message_stream_data(message)

        # 从 extensions 中获取中断的 Task 对象（用于恢复）
        interrupted_task = message.content.extensions.get("interrupted_task")
        if interrupted_task:
            # 从 stream_data 中提取交互组件ID
            component_id = self.extract_component_id_from_stream_data(message.content.stream_data)

            workflow_id = interrupted_task.input.target_id if interrupted_task.input else "unknown"
            logger.info(
                f"Saving interrupt state: workflow={workflow_id}, task={interrupted_task.task_id}, "
                f"component_id={component_id}")

            # 保存中断任务（component_id 会存入 task.metadata）
            self._state.add_interrupted_task(interrupted_task, component_id,
                                             component_id_key=self._TASK_METADATA_COMPONENT_ID_KEY)
            self.save_state()
            logger.info(f"Task {message.context.task_id} interrupted and saved")

        # 返回停止信号，携带中断结果（交互请求列表）
        result = message.content.stream_data
        logger.info(f"Task interrupted, returning stop signal with interaction requests")

        return MessageHandlerResult(
            tasks=[],
            stop=True,
            final_result=result
        )

    def _find_interrupted_task(self, workflow: WorkflowSchema) -> Optional[Task]:
        """查找指定workflow的中断任务 - 通用实现，消除子类重复
        
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

    async def _create_resume_task(self, message: Message, workflow: WorkflowSchema,
                                  interrupted_task: Task) -> MessageHandlerResult:
        """创建恢复任务 - 通用实现，消除子类重复
        
        将query封装为InteractiveInput恢复中断的workflow。
        
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
            stop=False
        )

    def _extract_interrupted_component_id(self, workflow_id: Optional[str] = None) -> Optional[str]:
        """从保存的状态中提取中断时的组件ID - 通用实现，消除子类重复
        
        Args:
            workflow_id: 可选，指定workflow ID。如果不指定，使用第一个中断的workflow
            
        Returns:
            Optional[str]: 中断时的组件ID，如果找不到则返回None
        """
        try:
            # 直接从状态中获取组件ID（状态已经在 load_state 时加载）
            component_id = self._state.get_interrupted_component_id(
                workflow_id,
                component_id_key=self._TASK_METADATA_COMPONENT_ID_KEY
            )
            if component_id:
                logger.debug(f"Retrieved component_id from state: {component_id}")
            return component_id
        except Exception as e:
            logger.debug(f"Failed to extract component_id from state: {e}")
            return None

    # ==================== Workflow 查找简化方法 ====================

    def _resolve_workflow_from_tasks(self, tasks: List[Task]) -> Optional[WorkflowSchema]:
        """从任务列表中解析workflow - 简化子类调用"""
        return self._resolve_workflow_from_tasks_generic(tasks, self.config.workflows)

    def _match_workflow(self, detected_task: Task) -> Optional[WorkflowSchema]:
        """匹配单个任务到workflow - 简化子类调用"""
        return self._match_workflow_generic(detected_task, self.config.workflows)
