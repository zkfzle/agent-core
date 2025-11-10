#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
import os
from typing import Dict, List

from jiuwen.core.utils.llm.messages import SystemMessage, HumanMessage
from jiuwen.agent_builder.nl_to_agent.utils.utils import load_yaml_file
from .prompt import generate_system_prompt, refine_user_prompt


class DLGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.reflect_prompts = []

        current_dir = os.path.dirname(__file__)
        components_schema_yml = os.path.join(current_dir, "schema.yaml")
        self.components_info, self.schema_info = self.load_components_schema_from_yaml(components_schema_yml)
    
    @staticmethod
    def load_components_schema_from_yaml(schema_file_path):
        yaml_data = load_yaml_file(schema_file_path)
        component_info = yaml_data.get("components")
        schema_info = "\n".join([value for key, value in yaml_data.items() if key != "components"])
        return component_info, schema_info

    def generate(self, query: str, resource: Dict[str, List[dict]]) -> str:
        system_prompt = self._update_prompt(resource)
        return self._execute(query, system_prompt)

    def refine(self, query: str, resource: Dict[str, List[dict]], exist_dl: str, exist_mermaid: str) -> str:
        system_prompt = self._update_prompt(resource)
        user_prompt = (refine_user_prompt.replace("{{user_input}}", query)
                                         .replace("{{exist_dl}}", exist_dl)
                                         .replace("{{exist_mermaid}}", exist_mermaid))
        return self._execute(user_prompt, system_prompt)

    def _execute(self, query, system_prompt) -> str:
        prompts = [SystemMessage(content=system_prompt), HumanMessage(content=query)] + self.reflect_prompts
        generated_dl = self.llm.chat(prompts)
        return generated_dl

    def _update_prompt(self, resource: Dict[str, List[dict]]):
        prompt = (generate_system_prompt.replace("{{components}}", self.components_info)
                                         .replace("{{schema}}", self.schema_info))
        for key, value in resource.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", f"{value}")
        return prompt
