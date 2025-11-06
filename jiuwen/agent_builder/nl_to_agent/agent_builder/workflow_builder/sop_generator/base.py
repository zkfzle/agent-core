#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
from typing import Optional

from jiuwen.core.utils.llm.messages import SystemMessage, HumanMessage
from .prompt import transform_system_prompt, generate_system_prompt


SOP_RESPONSE_CONTENT = "SOP内容如下：\n"


class SopGenerator:
    def __init__(self, llm, context_manager):
        self.llm = llm
        self.context_manager = context_manager
    
    def transform(self, query: str) -> str:
        pass

    def generate(self, query: str, resource: dict) -> str:
        pass

    def _execute(self, query: str, system_prompt: str) -> str:
        prompts = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        sop_result = self.llm.chat(prompts)
        self.context_manager.add_assistant_message(SOP_RESPONSE_CONTENT + sop_result, intent_label='工作流')
        return sop_result
