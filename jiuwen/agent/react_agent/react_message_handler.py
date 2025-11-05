#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import List, Optional
from jiuwen.agent.config.base import AgentConfig
from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.core.agent.message.message import Message
from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.controller.scheduler.message_handler import MessageHandler, MessageHandlerResult
from jiuwen.core.common.logging import logger
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.agent.message.message import MessageType
from jiuwen.core.agent.controller.utils import ReActControllerUtils, ReActControllerInput
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.agent.controller.controller import ControllerState

_RUNTIME_STATE_KEY = "react_controller_state"  # Runtime 中存储状态的 key
_STATE_INTERRUPTED_TASKS_KEY = "interrupted_tasks"  # 状态中存储中断任务列表的 key
_TASK_METADATA_COMPONENT_ID_KEY = "interrupted_component_id"  # Task metadata 中存储组件ID的 key

class ReActMessageHandler(MessageHandler):
    """ReActMessageHandler - ReAct模式的消息处理器，包含ReAct状态管理"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        super().__init__(config, context_engine, runtime)
        self._state = ControllerState()  # 状态数据对象
        self.load_state()  # 从runtime加载状态
        self.iteration = 0  # reasoning的迭代次数

    def save_state(self):
        """将状态保存到runtime"""
        if not self._state.interrupted_tasks:
            self.runtime.update_state({_RUNTIME_STATE_KEY: None})
            logger.debug("Cleared react state")
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
        """处理react模式的消息，返回处理结果"""
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
            logger.error(f"Error in ReActMessageHandler: {e}")
            # 发送错误流式消息
            error_result = await self._send_error_stream(str(e))
            # 返回停止信号
            return MessageHandlerResult(
                tasks=[],
                should_continue=False,
                final_result=error_result
            )


    async def _handle_user_input(self, message: Message) -> MessageHandlerResult:
        """
        处理用户输入消息。

        核心逻辑：
        返回基于大模型的规划，包含任务列表和是否继续信号。
        """
        if not self.config.model:
            logger.warning("Model config missing, cannot generate plan")
            return MessageHandlerResult(tasks=[], should_continue=False)

        # 添加user_message到对话历史
        ReActControllerUtils.add_user_message(message.get_display_content(), self.context_engine, self.runtime)

        # 调用大模型 reasoning 生成计划
        plan_result = await self._generate_plan_from_llm(message)
        if not plan_result.tasks:
            logger.info("No task is generated")
            final_result = await self._send_final_stream(plan_result.llm_output.content)
            return MessageHandlerResult(tasks=[], should_continue=False, final_result=final_result)
        
        # 判断规划任务是否为中断任务 恢复中断任务
        workflow_task = self._resolve_workflow_from_tasks(plan_result.tasks)
        if not workflow_task:
            # plugin任务 直接返回
            logger.info("Created plugin task: %s", plan_result.tasks)
            return MessageHandlerResult(tasks=plan_result.tasks, should_continue=plan_result.should_continue)

        # workflow任务 需要判断是否是中断的任务
        interrupted_task = self._find_interrupted_task(workflow_task)
        if interrupted_task:
            # reasoning返回的workflow处于中断状态，恢复它
            logger.info(f"Resuming interrupted workflow: {workflow_task.name}")
            return await self._create_resume_task(message, workflow_task, interrupted_task)

        logger.info(f"Created new workflow task: {workflow_task.name}")
        return MessageHandlerResult(tasks=plan_result.tasks, should_continue=plan_result.should_continue)

    async def _handle_task_completed(self, message: Message) -> MessageHandlerResult:
        """处理任务完成消息：写入流数据，清除workflow的状态，重新调用大模型reasoning生成计划，如果任务已经完成，返回停止信号"""
        # 写入任务完成的流数据
        await self._write_message_stream_data(message)

        # 添加工具调用结果到历史
        if message.content.stream_data[0].type == "plugin_final":
            ReActControllerUtils.add_tool_result(message, self.context_engine, self.runtime)

        # 清除workflow的中断状态（如果有）
        if message.context.workflow_id:
            self._state.clear_interrupted_task(message.context.workflow_id)
            self.save_state()
            logger.info(f"Cleared interrupt state for workflow: {message.context.workflow_id}")

        # 再次调用大模型 reasoning 生成计划 判断问题是否回答完成
        if self.iteration < self.config.constrain.max_iteration:
            plan_result = await self._generate_plan_from_llm(message)
            # 如果没有新的任务，返回停止信号，写入流数据, 携带最终结果
            if not plan_result.tasks:
                logger.info("No new tasks generated, task completed, returning stop signal with final result")
                final_result = await self._send_final_stream(plan_result.llm_output.content)
                return MessageHandlerResult(tasks=[], should_continue=False, final_result=final_result)
            return MessageHandlerResult(tasks=plan_result.tasks, should_continue=plan_result.should_continue)
            
        # 超过最大迭代次数，直接返回停止信号，携带最终结果
        result = message.content.stream_data
        logger.info(f"Exceed max iteration {self.config.constrain.max_iteration}, "
                    "task completed, returning stop signal with final result")

        return MessageHandlerResult(
            tasks=[],
            should_continue=False,
            final_result=result
        )

    async def _generate_plan_from_llm(self, message: Message) -> "ReActControllerOutput":
        """调用大模型生成计划, 并返回 ReActControllerOutput, 迭代次数+1
        
        核心逻辑：
        - 调用大语言模型生成任务执行计划
        - 如果是中断workflow任务, 从中断任务处恢复执行
        - 返回格式化的消息处理结果
        - reasoning迭代次数+1
        """
        logger.info(f"ReAct iteration {self.iteration + 1}")
        inputs = message.get_display_content()
        controller_input = ReActControllerInput(query=inputs)
        tools = self.runtime.get_tool_info()
        chat_history = ReActControllerUtils.get_chat_history(self.context_engine, self.runtime, self.config)
        llm_inputs = ReActControllerUtils.format_llm_inputs(controller_input, chat_history, self.config)
        if UserConfig.is_sensitive():
            logger.info(f"React llm inputs")
        else:
            logger.info(f"React llm inputs: {llm_inputs}")

        try:
            model = self._get_model()
            response = await model.ainvoke(
                self.config.model.model_info.model_name,
                llm_inputs,
                tools
            )

            result = ReActControllerUtils.parse_llm_output(response, self.config)
            # 大模型输出信息添加到CE对话历史中
            ReActControllerUtils.add_ai_message(result.llm_output, self.context_engine, self.runtime)
            if UserConfig.is_sensitive():
                logger.info(f"React llm output")
            else:
                logger.info(f"React llm output: {result.llm_output}")
        except Exception as e:
            self.iteration += 1
            logger.error(f"Failed to invoke model, {e}")
            import traceback
            logger.info(traceback.format_exc())
            raise JiuWenBaseException(-1, "Failed to invoke model")

        self.iteration += 1
        return result

    async def _handle_task_interrupted(self, message: Message) -> MessageHandlerResult:
        """处理任务中断消息：写入流数据，保存状态，返回停止信号 中断信息来源只有workflow任务"""
        # 写入任务中断的流数据
        await self._write_message_stream_data(message)

        # 从 extensions 中获取中断的 Task 对象（用于恢复）
        interrupted_task = message.content.extensions.get("interrupted_task")
        if interrupted_task:
            # 从 stream_data 中提取交互组件ID
            component_id = self._extract_component_id_from_stream_data(message.content.stream_data)

            workflow_id = interrupted_task.input.target_id if interrupted_task.input else "unknown"
            logger.info(
                f"Saving interrupt state: workflow={workflow_id}, task={interrupted_task.task_id}, component_id={component_id}")

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

    async def _create_resume_task(self, message: Message, workflow: WorkflowSchema,
                                  interrupted_task: Task) -> MessageHandlerResult:
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

    def _resolve_workflow_from_tasks(self, tasks: List[Task]) -> Optional[WorkflowSchema]:
        return self._resolve_workflow_from_tasks_generic(tasks, self.config.workflows)

    def _match_workflow(self, detected_task: Task) -> Optional[WorkflowSchema]:
        return self._match_workflow_generic(detected_task, self.config.workflows)

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
        