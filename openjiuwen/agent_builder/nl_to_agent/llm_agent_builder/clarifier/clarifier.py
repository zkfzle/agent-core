#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import List, Dict, Any

from openjiuwen.agent_builder.nl_to_agent.llm_agent_builder.clarifier.prompt import PROMPT


class Clarifier:
    def __init__(self, llm):
        self.llm = llm

    def clarify(self, messages: List[Dict[str, Any]], resource: Dict[Any, Any] )-> str:
        system_messages = {
            "role": "system",
            "content": PROMPT
        }

        clarifier_messages = [system_messages] + messages
        llm_output = self.llm.chat(messages=clarifier_messages, method="invoke", add_prefix=False)
        resource_keys = ", ".join(resource.keys())
        clarifier_output = f"{llm_output}\n\n<可用外部资源>\n【可用插件】\n{resource_keys}\n</可用外部资源>"

        return clarifier_output
