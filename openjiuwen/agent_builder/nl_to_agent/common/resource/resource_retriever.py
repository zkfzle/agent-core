#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import json
from typing import List, Dict

from openjiuwen.core.utils.llm.messages import SystemMessage
from openjiuwen.agent_builder.nl_to_agent.utils.utils import load_yaml_file
from openjiuwen.agent_builder.nl_to_agent.common.resource.prompt import retrieve_system_prompt
from openjiuwen.agent_builder.nl_to_agent.common.resource.plugin_processor import PluginProcessor


class ResourceRetriever:
    def __init__(self, llm):
        self.llm = llm

        raw_plugins = self.load_resources()
        self.plugin_dict, self.tool_plugin_id_map = PluginProcessor.preprocess(raw_plugins)
        self.retrieved_plugin = []

    @staticmethod
    def load_resources():
        current_dir = os.path.dirname(__file__)
        plugin_yml = os.path.join(current_dir, "plugins.yaml")
        raw_plugins = load_yaml_file(plugin_yml).get("plugins", [])
        return raw_plugins
    
    def collect_resource(self, id_list: List[str], resource_type: str) -> List[dict]:
        if resource_type == "plugin":
            return PluginProcessor.collect_plugin(id_list, self.plugin_dict, self.tool_plugin_id_map)
        return []

    def retrieve(self, dialog_history: List[Dict[str, str]], for_workflow: bool = True) -> Dict[str, List[dict]]:
        dialog_history_query = "\n".join(f"{msg['role']}: {msg['content']}" for msg in dialog_history)
        plugin_info_list = PluginProcessor.format_for_prompt(self.plugin_dict)

        system_prompt = (retrieve_system_prompt.replace("{{dialog_history}}", dialog_history_query)
                                               .replace("{{plugin_info_list}}", str(plugin_info_list)))
        data = self._llm_retrieve(system_prompt)
        data = json.loads(data)
        retrieved_plugin = PluginProcessor.get_retrieved_info(
            data.get("plugin_id_list", []),
            self.plugin_dict,
            self.tool_plugin_id_map,
            need_inputs_outputs=for_workflow
        )
        return dict(plugins=retrieved_plugin)
    
    def _llm_retrieve(self, system_prompt: str) -> List[str]:
        prompts = [SystemMessage(content=system_prompt)]
        response = self.llm.chat(prompts)
        return json.loads(response)