#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from openjiuwen.core.workflow.base import (
    Workflow,
    WorkflowExecutionState,
    WorkflowOutput,
    WorkflowChunk,
    generate_workflow_key,
)

from openjiuwen.core.workflow.workflow_config import (
    WorkflowMetadata,
    WorkflowInputsSchema,
    WorkflowConfig,
)

from openjiuwen.core.workflow.components.base import (
    WorkflowComponent,
    WorkflowComponentMetadata,
    ComponentConfig,
    ComponentState, ComponentExecutable, Input, Output,
)

from openjiuwen.core.workflow.components.basic_components.llm_comp import LLMComponent, LLMCompConfig
from openjiuwen.core.workflow.components.basic_components.tool_comp import ToolComponent, ToolComponentConfig
from openjiuwen.core.workflow.components.basic_components.workflow_comp import SubWorkflowComponent
from openjiuwen.core.workflow.components.basic_components.intent_detection_comp import (
    IntentDetectionComponent,
    IntentDetectionCompConfig
)

from openjiuwen.core.workflow.components.flow_components.start_comp import Start
from openjiuwen.core.workflow.components.flow_components.end_comp import End, EndConfig
from openjiuwen.core.workflow.components.flow_components.branch_comp import BranchComponent
from openjiuwen.core.workflow.components.flow_components.loop.loop_comp import LoopComponent, LoopGroup
from openjiuwen.core.workflow.components.flow_components.loop.break_comp import BreakComponent
from openjiuwen.core.workflow.components.flow_components.loop.set_variable_comp import SetVariableComponent
from openjiuwen.core.workflow.components.interact_components.questioner_comp import QuestionerComponent, \
    QuestionerConfig, FieldInfo
from openjiuwen.core.workflow.components.branch_router import BranchRouter, Branch
from openjiuwen.core.workflow.components.condition.condition import Condition, FuncCondition, AlwaysTrue
from openjiuwen.core.workflow.components.condition.expression import ExpressionCondition
from openjiuwen.core.workflow.components.condition.array import ArrayCondition
from openjiuwen.core.workflow.components.condition.number import NumberCondition


_WORKFLOW_CLASSES = [
    "Workflow",
    "WorkflowConfig",
    "WorkflowMetadata",
    "ComponentExecutable"
]

_WORKFLOW_METHODS = [
    "generate_workflow_key"
]

_WORKFLOW_INPUTS_AND_OUTPUTS_CLASSES = [
    "Input",
    "Output",
    "WorkflowInputsSchema",
    "WorkflowOutput",
    "WorkflowChunk",
    "WorkflowExecutionState",
]

_COMPONENTS_CLASSES = [
    "WorkflowComponent",
    "WorkflowComponentMetadata",
    "ComponentConfig",
    "ComponentState",
]

_BASIC_COMPONENTS = [
    "LLMComponent",
    "LLMCompConfig",
    "ToolComponent",
    "ToolComponentConfig",
    "SubWorkflowComponent",
    "IntentDetectionComponent",
    "IntentDetectionCompConfig"
]

_FLOW_COMPONENTS = [
    "Start",
    "End",
    "EndConfig",
    "BranchComponent",
    "LoopComponent",
    "LoopGroup",
    "BreakComponent",
    "SetVariableComponent",
    "BranchRouter",
    "Branch",
]

_INTERACT_COMPONENTS = [
    "QuestionerComponent",
    "QuestionerConfig",
    "FieldInfo"
]

_CONDITION_CLASSES = [
    "Condition",
    "FuncCondition",
    "ExpressionCondition",
    "ArrayCondition",
    "NumberCondition",
    "AlwaysTrue",
]

__all__ = (
        _WORKFLOW_CLASSES +
        _WORKFLOW_INPUTS_AND_OUTPUTS_CLASSES +
        _COMPONENTS_CLASSES +
        _BASIC_COMPONENTS +
        _FLOW_COMPONENTS +
        _INTERACT_COMPONENTS +
        _CONDITION_CLASSES +
        _WORKFLOW_METHODS
)
