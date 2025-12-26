#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Config of Agent"""
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field

from openjiuwen.core.single_agent.schema.schema import WorkflowSchema
from openjiuwen.core.common.constants.enums import ControllerType
from openjiuwen.core.foundation.llm.schema.model_config import ModelConfig


class AgentConfig(BaseModel):
    id: str = Field(default="")
    version: str = Field(default="")
    description: str = Field(default="")
    controller_type: ControllerType = Field(default=ControllerType.Undefined)
    workflows: List[WorkflowSchema] = Field(default_factory=list)
    model: Optional[ModelConfig] = Field(default=None)
    tools: List[str] = Field(default_factory=list)


class LLMCallConfig(BaseModel):
    model: Optional[ModelConfig] = Field(default=None)
    system_prompt: List[Dict] = Field(default_factory=list)
    user_prompt: List[Dict] = Field(default_factory=list)
    freeze_system_prompt: bool = Field(default=False)
    freeze_user_prompt: bool = Field(default=True)


class IntentDetectionConfig(BaseModel):
    """Intent detection configuration"""
    intent_detection_template: List[Dict] = Field(default_factory=list)
    default_class: str = Field(default="分类1")
    enable_input: bool = Field(default=True)
    enable_history: bool = Field(default=False)
    chat_history_max_turn: int = Field(default=5)
    category_list: List[str] = Field(default_factory=list)
    user_prompt: str = Field(default="")
    example_content: List[str] = Field(default_factory=list)


class ConstrainConfig(BaseModel):
    """Constraint configuration for agent behavior"""
    reserved_max_chat_rounds: int = Field(default=10, gt=0)
    max_iteration: int = Field(default=5, gt=0)


class DefaultResponse(BaseModel):
    """Default response configuration for workflow agent"""
    type: Literal["text", "workflow"] = "text"
    text: str = None


class WorkflowAgentConfig(AgentConfig):
    """Configuration for workflow agent"""
    controller_type: ControllerType = Field(default=ControllerType.WorkflowController)
    start_workflow: WorkflowSchema = Field(default_factory=WorkflowSchema)
    end_workflow: WorkflowSchema = Field(default_factory=WorkflowSchema)
    global_variables: List[dict] = Field(default_factory=list)
    global_params: Dict[str, Any] = Field(default_factory=dict)
    constrain: ConstrainConfig = Field(default=ConstrainConfig())
    default_response: DefaultResponse = Field(default_factory=DefaultResponse)
