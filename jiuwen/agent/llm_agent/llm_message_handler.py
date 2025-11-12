#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.message.message import Message
from jiuwen.core.agent.controller.scheduler.message_handler import MessageHandler, MessageHandlerResult
from jiuwen.core.common.logging import logger
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.agent.message.message import MessageType
from jiuwen.core.agent.controller.utils import MessageHandlerUtils
from jiuwen.agent.utils import MessageUtils
from jiuwen.core.utils.config.user_config import UserConfig


class ReActMessageHandler(MessageHandler):
    """ReActMessageHandler - ReAct模式的消息处理器，包含ReAct状态管理"""

    def __init__(self, config: AgentConfig, context_engine=None, runtime=None):
        super().__init__(config, context_engine, runtime)
        self.load_state()  # 从runtime加载状态（基类实现）
        self.iteration = 0  # reasoning的迭代次数

    def get_state_key(self) -> str:
        """返回react handler的状态存储key"""
        return "react_controller_state"

    async def handle_message(self, message: Message) -> MessageHandlerResult:
        """处理react模式的消息，返回处理结果
        
        消息分发逻辑：
        - TASK_COMPLETED: 调用 LLM 再次 reasoning，决定是否继续
        - TASK_INTERRUPTED: 使用基类通用中断处理
        - USER_INPUT: 调用 LLM reasoning 生成计划
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
            logger.error(f"Error in ReActMessageHandler: {e}")
            error_result = await self._send_error_stream(str(e))
            return MessageHandlerResult(
                tasks=[],
                stop=True,
                final_result=error_result
            )

    async def _handle_user_input(self, message: Message) -> MessageHandlerResult:
        """处理用户输入消息 - React 核心：LLM reasoning 生成计划
        
        流程：
        1. 添加用户消息到对话历史
        2. 调用 LLM reasoning 生成任务计划
        3. 如果是 workflow 任务，判断是否需要恢复中断任务（使用基类方法）
        4. 返回任务列表
        """
        if not self.config.model:
            logger.warning("Model config missing, cannot generate plan")
            return MessageHandlerResult(tasks=[], stop=True)

        # 添加user_message到对话历史
        MessageUtils.add_user_message(message.get_display_content(), self.context_engine, self.runtime)

        # 调用大模型 reasoning 生成计划
        tasks, llm_output = await self._generate_plan_from_llm(message)
        if not tasks:
            logger.info("No task is generated")
            final_result = await self._send_final_stream(llm_output.content)
            return MessageHandlerResult(tasks=[], stop=True, final_result=final_result)

        # 判断规划任务是否为 workflow 任务
        workflow_task = self._resolve_workflow_from_tasks(tasks)
        if not workflow_task:
            # plugin任务 直接返回
            logger.info("Created plugin task: %s", tasks)
            return MessageHandlerResult(tasks=tasks, stop=False)

        # workflow任务 - 判断是否需要恢复中断任务（使用基类方法）
        interrupted_task = self._find_interrupted_task(workflow_task)
        if interrupted_task:
            # reasoning返回的workflow处于中断状态，恢复它（使用基类方法）
            logger.info(f"Resuming interrupted workflow: {workflow_task.name}")
            return await self._create_resume_task(message, workflow_task, interrupted_task)

        logger.info(f"Created new workflow task: {workflow_task.name}")
        return MessageHandlerResult(tasks=tasks, stop=False)

    async def _handle_task_completed(self, message: Message) -> MessageHandlerResult:
        """处理任务完成消息 - React 特有：继续 LLM reasoning 直到任务完成
        
        与 Workflow 的区别：
        - Workflow: 任务完成就停止
        - React: 任务完成后再次调用 LLM reasoning，判断问题是否已解决
        """
        # 写入任务完成的流数据
        await self._write_message_stream_data(message)

        # 添加工具调用结果到历史
        if message.content.stream_data[0].type in ("plugin_final", "workflow_final"):
            MessageHandlerUtils.add_tool_result(message, self.context_engine, self.runtime)

        # 清除workflow的中断状态（如果有）- 使用基类状态管理
        if message.context.workflow_id:
            self._state.clear_interrupted_task(message.context.workflow_id)
            self.save_state()
            logger.info(f"Cleared interrupt state for workflow: {message.context.workflow_id}")

        # 再次调用大模型 reasoning 生成计划 判断问题是否回答完成
        if self.iteration < self.config.constrain.max_iteration:
            tasks, llm_output = await self._generate_plan_from_llm(message)
            # 如果没有新的任务，返回停止信号，写入流数据, 携带最终结果
            if not tasks:
                logger.info("No new tasks generated, task completed, returning stop signal with final result")
                final_result = await self._send_final_stream(llm_output.content)
                return MessageHandlerResult(tasks=[], stop=True, final_result=final_result)
            return MessageHandlerResult(tasks=tasks, stop=False)

        # 超过最大迭代次数，直接返回停止信号，携带最终结果
        result = message.content.stream_data
        logger.info(f"Exceed max iteration {self.config.constrain.max_iteration}, "
                    "task completed, returning stop signal with final result")

        return MessageHandlerResult(
            tasks=[],
            stop=True,
            final_result=result
        )

    async def _generate_plan_from_llm(self, message: Message):
        """调用大模型生成计划 - React 核心方法
        
        这是 ReAct 模式的唯一差异化逻辑：
        - 使用 LLM reasoning 生成任务执行计划
        - 解析 LLM 输出为任务列表
        - 迭代次数+1
        
        Returns:
            Tuple[List[Task], BaseMessage]: (tasks, llm_output)
        """
        logger.info(f"ReAct iteration {self.iteration + 1}")
        inputs = message.get_display_content()
        tools = self.runtime.get_tool_info()
        chat_history = MessageUtils.get_chat_history(self.context_engine, self.runtime, self.config)
        llm_inputs = MessageHandlerUtils.format_llm_inputs(inputs, chat_history, self.config)
        if UserConfig.is_sensitive():
            logger.info(f"React llm inputs")
        else:
            logger.info(f"React llm inputs: {llm_inputs}")

        try:
            model = self._get_model()
            llm_output = await model.ainvoke(
                self.config.model.model_info.model_name,
                llm_inputs,
                tools
            )

            tasks = MessageHandlerUtils.parse_llm_output(llm_output, self.config)
            # 大模型输出信息添加到CE对话历史中
            MessageUtils.add_ai_message(llm_output, self.context_engine, self.runtime)
            if UserConfig.is_sensitive():
                logger.info(f"React llm output")
            else:
                logger.info(f"React llm output: {llm_output}")
        except Exception as e:
            self.iteration += 1
            logger.error(f"Failed to invoke model, {e}")
            raise JiuWenBaseException(-1, "Failed to invoke model")

        self.iteration += 1
        return tasks, llm_output
