#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Dict, List

from openjiuwen.core.utils.llm.messages import SystemMessage, HumanMessage
from openjiuwen.agent_builder.nl_to_agent.workflow_builder.sop_generator.prompt import transform_system_prompt, generate_system_prompt


SOP_GENERATE_PROMPT = "请根据以下对话历史设计工作流程：\n"


class SopGenerator:
    def __init__(self, llm):
        self.llm = llm
    
    def transform(self, query: str) -> str:
        return self._execute(query, transform_system_prompt)

    def generate(self, query, resource: Dict[str, List[dict]]) -> str:
        query = SOP_GENERATE_PROMPT + query
        system_prompt = generate_system_prompt.replace("{{resource_info}}", self._format_resource_info(resource))
        return self._execute(query, system_prompt)

    def _execute(self, query: str, system_prompt: str) -> str:
        prompts = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        sop_result = self.llm.chat(prompts)
        return sop_result
    
    def _format_resource_info(self, resource: Dict[str, List[dict]]) -> str:
        if not resource:
            return "无可用工具/资源/外部接口。"
        info_lines = []
        for key, value in resource.items():
            info_lines.append(f"{key}:")
            for item in value:
                name = item.get('name')
                desc = item.get('description', '无描述')
                if name:
                    info_lines.append(f"- {name}: {desc}")
        return "\n".join(info_lines)
    
