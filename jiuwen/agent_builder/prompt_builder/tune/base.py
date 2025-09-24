#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Optional, List, Dict

from pydantic import BaseModel, Field

from jiuwen.core.utils.llm.messages import BaseMessage, ToolInfo, AIMessage


class TuneConstant:
    """prompt tuning constants"""

    """optimizer parameters default value constant"""
    DEFAULT_EXAMPLE_NUM: int = 0
    DEFAULT_ITERATION_NUM: int = 3
    DEFAULT_MAX_SAMPLED_EXAMPLE_NUM: int = 10
    DEFAULT_NUM_PARALLEL: int = 1
    DEFAULT_LLM_CALL_RETRY_NUM: int = 5
    DEFAULT_MAX_NUM_SAMPLE_ERROR_CASES: int = 20,
    DEFAULT_EARLY_STOP_ACCURACY: int = 1.0

    """optimizer parameters threshold constant"""
    MIN_ITERATION_NUM: int = 1
    MAX_ITERATION_NUM: int = 20
    MIN_LLM_CALL_RETRY_NUM: int = 1
    MAX_LLM_CALL_RETRY_NUM: int = 10
    MIN_LLM_PARALLEL_DEGREE: int = 1
    MAX_LLM_PARALLEL_DEGREE: int = 10
    MIN_EXAMPLE_NUM: int = 0
    MAX_EXAMPLE_NUM: int = 10

    """training status"""
    TASK_RUNNING = "running"
    TASK_FINISHED = "finished"
    TASK_FAILED = "failed"


class Case(BaseModel):
    """definition of case"""
    messages: List[BaseMessage] = Field(default=[])
    variables: Dict[str, str] = Field(default={})
    label: AIMessage = Field(...)
    tools: Optional[List[ToolInfo]] = Field(default=None)


class EvaluatedCase(Case):
    """definition of evaluated case"""
    answer: Optional[AIMessage] = Field(default=None)
    score: int = Field(default=0)
    reason: str = Field(default="")


class PromptOptimizeHistory(BaseModel):
    iteration_round: int = Field(default=0)
    optimized_prompt: str = Field(default="")
    accuracy: float = Field(default=0.0)


class PromptOptimizeProgress(BaseModel):
    instruction: str = Field(default="")
    examples: List[Case] = Field(default=[])
    best_prompt: str = Field(default="")
    best_accuracy: float = Field(default=0.0)
    iteration_round: int = Field(default=0)
    status: str = Field(default="running")
    exception_info: str = Field(default="")
    history: List[PromptOptimizeHistory] = Field(default=[])