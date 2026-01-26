# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import uuid
from typing import List, Dict, Tuple

from openjiuwen.core.controller.schema import Intent, IntentType


class IntentToolkits:
    def __init__(self, event, confidence_threshold: float):
        self.event = event
        self.confidence_threshold = confidence_threshold

    def _low_confidence_intent(self, confidence: float) -> Tuple[Intent, str]:
        return Intent(
            intent_type=IntentType.UNKNOWN_TASK,
            event=self.event,
            target_task_id="",
            target_task_description=None,
            depend_task_id="",
            supplementary_info=None,
            modification_details=None,
            confidence=confidence,
            clarification_prompt="不好意思，未能理解您的意思，请明确当前需要创建新任务还是修改某一任务",
        ), f"由于当前confidence过低，自动转换为unknown_task"

    def create_task(self, confidence: float, task_description: str, dependent_task_id: str = "") -> Tuple[Intent, str]:
        if confidence < self.confidence_threshold:
            return self._low_confidence_intent(confidence)
        target_task_id = str(uuid.uuid4())
        return Intent(
            intent_type=IntentType.CREATE_TASK,
            event=self.event,
            target_task_id=target_task_id,
            target_task_description=task_description,
            depend_task_id=dependent_task_id,
            supplementary_info=None,
            modification_details=None,
            confidence=confidence,
            clarification_prompt=None
        ), f"任务id: {target_task_id}，任务描述:{task_description}，当前状态: 已创建，并提交执行"

    def pause_task(self, confidence: float, task_id: str) -> Tuple[Intent, str]:
        if confidence < self.confidence_threshold:
            return self._low_confidence_intent(confidence)
        return Intent(
            intent_type=IntentType.PAUSE_TASK,
            event=self.event,
            target_task_id=task_id,
            target_task_description=None,
            depend_task_id="",
            supplementary_info=None,
            modification_details=None,
            confidence=confidence,
            clarification_prompt=None
        ), f"任务id: {task_id}，当前状态: 已暂停"

    def cancel_task(self, confidence: float, task_id: str) -> Tuple[Intent, str]:
        if confidence < self.confidence_threshold:
            return self._low_confidence_intent(confidence)
        return Intent(
            intent_type=IntentType.CANCEL_TASK,
            event=self.event,
            target_task_id=task_id,
            target_task_description=None,
            depend_task_id="",
            supplementary_info=None,
            modification_details=None,
            confidence=confidence,
            clarification_prompt=None
        ), f"任务id: {task_id}，当前状态: 已取消"

    def resume_task(self, confidence: float, task_id: str) -> Tuple[Intent, str]:
        if confidence < self.confidence_threshold:
            return self._low_confidence_intent(confidence)
        return Intent(
            intent_type=IntentType.CONTINUE_TASK,
            event=self.event,
            target_task_id=task_id,
            target_task_description=None,
            depend_task_id="",
            supplementary_info=None,
            modification_details=None,
            confidence=confidence,
            clarification_prompt=None
        ), f"任务id: {task_id}，当前状态: 已恢复"

    def unknown_task(self, confidence: float, question_for_user: str) -> Tuple[Intent, str]:
        if confidence < self.confidence_threshold:
            return self._low_confidence_intent(confidence)
        return Intent(
            intent_type=IntentType.UNKNOWN_TASK,
            event=self.event,
            target_task_id="",
            target_task_description=None,
            depend_task_id="",
            supplementary_info=None,
            modification_details=None,
            confidence=confidence,
            clarification_prompt=question_for_user,
        ), f"已发送请求，等待用户回复中。"

    @staticmethod
    def get_openai_tool_schemas() -> List[Dict]:
        """
        list OpenAI Tool Schemas

        Returns:
            List[Dict]: OpenAI tool schemas
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_task",
                    "description": "创建新任务。当用户想要开始一个新的任务或活动时使用此方法。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "模型对此意图的置信度分数（0-1.0），通常较低时使用此方法"
                            },
                            "task_description": {
                                "type": "string",
                                "description": "任务的详细描述，说明用户想要完成的具体内容"
                            },
                            "dependent_task_id": {
                                "type": "string",
                                "description": "可选参数，指定此任务依赖的前置任务ID，用于任务间依赖关系"
                            }
                        },
                        "required": ["confidence", "task_description"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "pause_task",
                    "description": "暂停指定任务。当用户想要暂时中断或挂起一个正在进行的任务时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "模型对此意图的置信度分数（0-1.0），通常较低时使用此方法"
                            },
                            "task_id": {
                                "type": "string",
                                "description": "需要暂停的任务的唯一标识符"
                            }
                        },
                        "required": ["confidence", "task_id"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_task",
                    "description": "取消指定任务。当用户想要完全终止或放弃一个任务时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "模型对此意图的置信度分数（0-1.0），通常较低时使用此方法"
                            },
                            "task_id": {
                                "type": "string",
                                "description": "需要取消的任务的唯一标识符"
                            }
                        },
                        "required": ["confidence", "task_id"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "resume_task",
                    "description": "继续指定任务。当用户想要恢复一个之前被暂停或中断的任务时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "模型对此意图的置信度分数（0-1.0），通常较低时使用此方法"
                            },
                            "task_id": {
                                "type": "string",
                                "description": "需要继续的任务的唯一标识符"
                            }
                        },
                        "required": ["confidence", "task_id"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "unknown_task",
                    "description": "处理未知或模糊的用户意图。当无法确定用户确切意图时，使用此方法创建需要用户澄清的问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "模型对此意图的置信度分数（0-1.0），通常较低时使用此方法"
                            },
                            "question_for_user": {
                                "type": "string",
                                "description": "向用户提出的澄清问题，用于获取更多信息以确定确切意图"
                            }
                        },
                        "required": ["confidence", "question_for_user"],
                        "additionalProperties": False
                    }
                }
            }
        ]