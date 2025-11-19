#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import json
from typing import List, Dict

from openjiuwen.core.utils.llm.messages import SystemMessage
from openjiuwen.agent_builder.nl_to_agent.utils.utils import load_yaml_file
from openjiuwen.agent_builder.nl_to_agent.common.resource.prompt import retrieve_system_prompt

class ResourceRetriever:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _load_resources():
        current_dir = os.path.dirname(__file__)
        plugin_yml = os.path.join(current_dir, "plugins.yaml")
        raw_plugins = load_yaml_file(plugin_yml)
        return dict(raw_plugins=raw_plugins.get("plugins", []))

    def retrieve(self, query: str, dialog_history: str):
        raw = self._load_resources()
        plugin_dict, tool_plugin_id_map = self._preprocess_plugin_info(raw.get("raw_plugins", []))
        plugin_info_list = self._format_plugin_info_list(plugin_dict)

        system_prompt = (retrieve_system_prompt.replace("{{dialog_history}}", dialog_history)
                                               .replace("{{user_input}}", query)
                                               .replace("{{plugin_info_list}}", str(plugin_info_list)))
        data = self._llm_retrieve(system_prompt)
        return self._get_retrieved_info(data, plugin_dict, tool_plugin_id_map)

    def _preprocess_plugin_info(self, raw_plugins: List[dict]):
        def format_params(params):
            return [
                {"name": p.get("name", ""), "desc": p.get("description", "")}
                for p in (params or [])
            ]

        plugin_dict = {}
        tool_plugin_id_map = {}
        for plugin in raw_plugins or []:
            plugin_id = plugin.get("plugin_id", "")
            if not plugin_id:
                continue

            formatted_tools = {}
            for tool in plugin.get("tools", []):
                tool_id = tool.get("tool_id")
                if not tool_id:
                    continue

                tool_plugin_id_map[tool_id] = plugin_id
                input_params = tool.get("input_parameters", [])
                output_params = tool.get("output_parameters", [])
                tool_dict = {
                    "tool_id": tool_id,
                    "tool_name": tool.get("tool_name", ""),
                    "tool_desc": tool.get("desc", ""),
                    "ori_inputs": input_params,
                    "ori_outputs": output_params,
                    "inputs_for_dl_gen": format_params(input_params),
                    "outputs_for_dl_gen": format_params(output_params)
                }
                formatted_tools[tool_id] = tool_dict

            plugin_dict[plugin_id] = {
                "plugin_id": plugin_id,
                "plugin_name": plugin.get("plugin_name", ""),
                "plugin_desc": plugin.get("plugin_desc", ""),
                "tools": formatted_tools,
            }

        return plugin_dict, tool_plugin_id_map
    
    def _format_plugin_info_list(self, plugin_dict: Dict[str, dict]):
        result = []
        for plugin in plugin_dict.values():
            tools_brief = [
                {"tool_id": t["tool_id"], "tool_name": t["tool_name"], "tool_desc": t["tool_desc"]}
                for t in plugin.get("tools", {}).values()
            ]
            result.append({
                "plugin_id": plugin["plugin_id"],
                "plugin_name": plugin["plugin_name"],
                "plugin_desc": plugin["plugin_desc"],
                "tools": tools_brief,
            })
        return result
    
    def _llm_retrieve(self, system_prompt: str) -> List[str]:
        prompts = [SystemMessage(content=system_prompt)]
        response = self.llm.chat(prompts)
        return json.loads(response)
    
    def _get_retrieved_info(self,data, plugin_dict, tool_plugin_id_map):
        retrieved_plugin_dict = {}
        retrieved_tool_plugin_id_map = {}
        tool_detail_list = []
        for tool_id in data.get("plugin_id_list", []):
            if tool_id in tool_plugin_id_map:
                plugin_id = tool_plugin_id_map[tool_id]

                if plugin_id not in retrieved_plugin_dict:
                    retrieved_plugin_dict[plugin_id] = plugin_dict[plugin_id]
                retrieved_tool_plugin_id_map.update({tool_id: plugin_id})

                tool_detail = plugin_dict[plugin_id]["tools"][tool_id]
                tool_detail_list.append({
                    "tool_id": tool_id,
                    "tool_name": tool_detail["tool_name"],
                    "tool_desc": tool_detail["tool_desc"],
                    "inputs": tool_detail["inputs_for_dl_gen"],
                    "outputs": tool_detail["outputs_for_dl_gen"],
                })
        return dict(
            tool_detail_list=tool_detail_list,
            plugin_dict=retrieved_plugin_dict,
            tool_plugin_id_map=retrieved_tool_plugin_id_map
        )