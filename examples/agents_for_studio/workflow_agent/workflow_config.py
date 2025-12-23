#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import List, Dict, Any, Literal

from pydantic import Field, BaseModel

from openjiuwen.core.common.constants.enums import ControllerType
from openjiuwen.core.single_agent.schema.schema import WorkflowSchema
from openjiuwen.core.single_agent.config.base import AgentConfig
from examples.agents_for_studio.llm_agent.react_config import ConstrainConfig


class DefaultResponse(BaseModel):
    type: Literal["text", "workflow"] = "text"
    text: str = None


class WorkflowAgentConfig(AgentConfig):
    controller_type: ControllerType = Field(default=ControllerType.WorkflowController)
    start_workflow: WorkflowSchema = Field(default_factory=WorkflowSchema)
    end_workflow: WorkflowSchema = Field(default_factory=WorkflowSchema)
    global_variables: List[dict] = Field(default_factory=list)
    global_params: Dict[str, Any] = Field(default_factory=dict)
    constrain: ConstrainConfig = Field(default=ConstrainConfig())
    default_response: DefaultResponse = Field(default_factory=DefaultResponse)
