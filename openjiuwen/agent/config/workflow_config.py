#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import List, Dict, Any

from pydantic import Field

from openjiuwen.agent.common.enum import ControllerType
from openjiuwen.agent.common.schema import WorkflowSchema
from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.agent.config.react_config import ConstrainConfig


class WorkflowAgentConfig(AgentConfig):
    controller_type: ControllerType = Field(default=ControllerType.WorkflowController)
    start_workflow: WorkflowSchema = Field(default_factory=WorkflowSchema)
    end_workflow: WorkflowSchema = Field(default_factory=WorkflowSchema)
    global_variables: List[dict] = Field(default_factory=list)
    global_params: Dict[str, Any] = Field(default_factory=dict)

    constrain: ConstrainConfig = Field(default=ConstrainConfig())
