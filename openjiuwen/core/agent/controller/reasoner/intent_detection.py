#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import json
import secrets
from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.core.agent.controller.config.reasoner_config import IntentDetectionConfig
from openjiuwen.core.agent.controller.constants import IntentDetectionConstants
from openjiuwen.core.agent.controller.utils import ReasonerUtils
from openjiuwen.core.agent.message.message import Message
from openjiuwen.core.agent.task.task import Task, TaskType, TaskInput
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.context_engine.engine import ContextEngine
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.core.utils.llm.messages import BaseMessage
from typing import List, Union

class IntentDetection:
    """IntentDetection - 意图识别模块，负责识别消息意图并生成简单任务"""

    def __init__(self, intent_config: IntentDetectionConfig, agent_config: AgentConfig,
                context_engine: ContextEngine, runtime: Runtime):
        """
        初始化IntentDetection
        
        Args:
            intent_config: IntentDetection配置
            agent_config: Agent配置
            context_engine: 上下文引擎
            runtime: 运行时环境
        """
        self.intent_config = intent_config
        self.agent_config = agent_config
        self.context_engine = context_engine
        self.runtime = runtime

    async def process_message(self, message: Message) -> List[Task]:
        """
        处理消息，识别意图并生成任务
        
        Args:
            message: 输入消息
            
        Returns:
            List[Task]: 生成的任务列表
        """
        # 1. 识别意图
        llm_inputs = self._prepare_detection_input(message)
        session_id = self.runtime.session_id()
        if UserConfig.is_sensitive():
            logger.info(f"[%s] <LLM Input>", session_id)
        else:
            logger.info(f"[%s] <LLM Input>: %s", session_id, llm_inputs)
        
        # 2. 调用大模型识别意图
        llm_output = await self._invoke_llm_get_output(llm_inputs)
        if UserConfig.is_sensitive():
            logger.info(f"[%s] <LLM Output>", session_id)
        else:
            logger.info(f"[%s] <LLM Output>: %s", session_id, llm_output)
        detected_intent_id = self._parse_intent_from_output(llm_output)
        
        # 3. 根据意图创建任务
        tasks = self._generate_tasks_from_intent(detected_intent_id, message)
        return tasks

    def _generate_tasks_from_intent(
        self, intent_id: str, message: Message
    ) -> List[Task]:
        """
        创建任务对象
        
        根据识别到的意图创建对应的任务
        1. 映射意图到任务类型
        2. 创建任务实例
        3. 返回任务列表
        
        Note: 如果 agent_config 没有 workflows，直接用 intent_id 作为 target_name
        """
        tasks = []
        session_id = self.runtime.session_id()
        task_unique_id = f"{session_id}_intent_{intent_id}_{secrets.token_hex(4)}"
        
        if intent_id == IntentDetectionConstants.DEFAULT_CLASS:
            # 意图识别没有匹配结果，返回空任务列表
            return tasks
        
        # 如果没有 workflows，直接用 intent_id 作为 target
        workflows = getattr(self.agent_config, 'workflows', None) or []
        if not workflows:
            task_input = TaskInput(target_id=intent_id, target_name=intent_id, arguments=message.content)
            task = Task(
                agent_id=self.agent_config.id,
                task_id=task_unique_id,
                task_type=TaskType.WORKFLOW,
                input=task_input
            )
            tasks.append(task)
            logger.info(
                f"[%s] success to create task for intent (direct): %s",
                session_id, intent_id
            )
            return tasks
        
        # 有 workflows 时，匹配 workflow
        for workflow in workflows:
            if workflow.id == intent_id:
                task_input = TaskInput(target_id=workflow.id, target_name=workflow.name, arguments=message.content)
                task = Task(
                    agent_id=self.agent_config.id,
                    task_id=task_unique_id,
                    task_type=TaskType.WORKFLOW,
                    input=task_input
                )
                tasks.append(task)
                logger.info(
                    f"[%s] success to create task for intent: %s",
                    session_id, intent_id
                )
                break
        return tasks

    def _parse_intent_from_output(self, llm_output: str) -> str:
        """
        从大模型输出中解析意图
        
        从大模型的输出中提取意图
        1. 解析输出中的意图标签
        2. 返回意图工作流id或category名称
        
        Note: 如果 agent_config 没有 workflows，直接返回 category 名称
        """
        detected_intent_id = ""
        session_id = self.runtime.session_id()
        try:
            output_data = json.loads(llm_output)
            detected_class_number = int(output_data.get('result', ''))
            if (detected_class_number <= 0 or
                    detected_class_number > len(self.intent_config.category_list)):
                # 意图不明
                logger.warning("get unknown class")
            else:
                detected_intent_name = (
                    self.intent_config.category_list[detected_class_number - 1]
                )
                
                # 如果没有 workflows，直接返回 category 名称
                workflows = getattr(self.agent_config, 'workflows', None) or []
                if not workflows:
                    logger.info(
                        f"[%s] get intent (direct category): %s",
                        session_id, detected_intent_name
                    )
                    return detected_intent_name
                
                # 有 workflows 时，匹配 workflow
                # 优先通过 description 匹配，如果匹配不到再通过 name 匹配
                for workflow in workflows:
                    workflow_label = (
                        workflow.description if workflow.description else workflow.name
                    )
                    if workflow_label == detected_intent_name:
                        detected_intent_id = workflow.id
                        logger.info(
                            f"[%s] get intent: %s", session_id, detected_intent_id
                        )
                        break
                return detected_intent_id
        except Exception as e:
            if UserConfig.is_sensitive():
                logger.error("failed to parse JSON from LLM output")
            else:
                logger.error(
                    "failed to parse JSON from LLM output, error: %s", str(e)
                )
            raise

        return IntentDetectionConstants.DEFAULT_CLASS

    async def _invoke_llm_get_output(self, llm_inputs: Union[List[BaseMessage], str]) -> str:
        """调用大模型invoke方法获取输出"""
        try:
            # 调用LLM
            model = ReasonerUtils.get_model(self.agent_config.model, self.runtime)
            llm_output = await model.ainvoke(self.agent_config.model.model_info.model_name, llm_inputs)
            llm_output_content = llm_output.content.strip()
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.INVOKE_LLM_FAILED.code,
                message=StatusCode.INVOKE_LLM_FAILED.errmsg
            ) from e

        return llm_output_content

    def _prepare_detection_input(self, message: Message) -> str:
        """
        准备意图识别输入
        
        将当前消息和历史记录拼接为大模型的输入
        1. 格式化当前消息
        2. 格式化历史记录
        3. 拼接为大模型输入
        """
        category_list = "分类0：意图不明\n" + "\n".join(f"分类{i+1}：{c}" for i, c in enumerate(self.intent_config.category_list))
        current_inputs = {}
        current_inputs.update({
            IntentDetectionConstants.USER_PROMPT: self.intent_config.user_prompt,
            IntentDetectionConstants.CATEGORY_LIST: category_list,
            IntentDetectionConstants.DEFAULT_CLASS: self.intent_config.default_class,
            IntentDetectionConstants.ENABLE_HISTORY: self.intent_config.enable_history,
            IntentDetectionConstants.ENABLE_INPUT: self.intent_config.enable_input,
            IntentDetectionConstants.EXAMPLE_CONTENT: "\n\n".join(self.intent_config.example_content),
            IntentDetectionConstants.CHAT_HISTORY_MAX_TURN: self.intent_config.chat_history_max_turn,
            IntentDetectionConstants.CHAT_HISTORY: ""
        })

        # 更新对话历史
        if self.intent_config.enable_history:
            chat_history = ReasonerUtils.get_chat_history(self.context_engine, self.runtime,
                                                          self.intent_config.chat_history_max_turn)
            chat_history_str = ""
            for history in chat_history:
                chat_history_str += "{}: {}\n".format(
                    IntentDetectionConstants.ROLE_MAP.get(history.role, "用户"),
                    history.content
                )
            current_inputs.update({IntentDetectionConstants.CHAT_HISTORY: chat_history_str})

        # 处理当前输入
        if self.intent_config.enable_input:
            current_inputs.update({IntentDetectionConstants.INPUT: message.content.get_query() or ""})
        llm_inputs = self.intent_config.intent_detection_template.format(current_inputs).to_messages()
        return llm_inputs
