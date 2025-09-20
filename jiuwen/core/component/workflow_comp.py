#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from jiuwen.core.common.constants.component import SUB_WORKFLOW_COMPONENT
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.context_engine.base import Context
from jiuwen.core.graph.base import INPUTS_KEY, CONFIG_KEY
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.workflow.base import Workflow


class SubWorkflowComponent(WorkflowComponent, ComponentExecutable):
    def __init__(self, sub_workflow: Workflow):
        super().__init__()
        self._sub_workflow = sub_workflow

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        return await self._sub_workflow.sub_invoke(inputs.get(INPUTS_KEY), runtime.base(), inputs.get(CONFIG_KEY))

    def graph_invoker(self) -> bool:
        return True

    def component_type(self) -> str:
        return SUB_WORKFLOW_COMPONENT
