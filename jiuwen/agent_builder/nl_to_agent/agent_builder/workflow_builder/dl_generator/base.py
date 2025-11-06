#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
from jiuwen.core.utils.llm.messages import SystemMessage, HumanMessage
from .prompt import generate_system_prompt, refine_system_prompt


class DLGenerator:
    def __init__(self, llm, context_manager):
        self.llm = llm
        self.context_manager = context_manager
        self._reflect_prompts = []

    def generate(self, query: str, resource: dict) -> str:
        pass

    def refine(self, query: str, resource: dict, exist_dl: str, exist_mermaid: str) -> str:
        pass

    def add_reflect_prompt(self, prompt: str):
        self._reflect_prompts.append(HumanMessage(content=prompt))

    def reset(self):
        self._reflect_prompts = []

    def _execute(self, query, system_prompt) -> str:
        prompts = [SystemMessage(content=system_prompt), HumanMessage(content=query)] + self._reflect_prompts
        dl_result = self.llm.chat(prompts)
        return dl_result
