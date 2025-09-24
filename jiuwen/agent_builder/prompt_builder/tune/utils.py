#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import re
import json
from ast import literal_eval
from typing import Optional, List, Dict, Any

from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.messages import BaseMessage, AIMessage
from jiuwen.agent_builder.prompt_builder.tune.base import Case


class TuneUtils:
    @staticmethod
    def get_input_string_from_case(case: Case):
        messages_content = []
        for message in case.messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                content = "".join(json.dumps(tool_call.function.model_dump()) for tool_call in message.tool_calls)
            else:
                content = message.content
            messages_content.append(f"[{message.role}]: {content}")
        input_string = "\n".join(messages_content)
        if case.variables:
            input_string += f"\nvariables: {str(case.variables)}\n"
        return input_string

    @staticmethod
    def get_output_string_from_message(message: BaseMessage):
        if isinstance(message, AIMessage) and message.tool_calls:
            return "".join("".join(json.dumps(tool_call.function.model_dump()) for tool_call in message.tool_calls))
        return message.content

    @staticmethod
    def parse_json_from_llm_response(json_like_string: str) -> Optional[Dict[str, Any]]:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, json_like_string, re.DOTALL)
        if not match:
            logger.warning("Failed to extract json string from response")
            return None

        matched_json_string = match.group(1).strip()
        try:
            json_data = json.loads(matched_json_string)
        except json.decoder.JSONDecodeError:
            logger.warning("Failed to decode json string")
            return None
        return json_data

    @staticmethod
    def parse_list_from_llm_response(list_like_string: str) -> Optional[List[Any]]:
        pattern = r"```list(.*?)```"
        match = re.search(pattern, list_like_string, re.DOTALL)
        if not match:
            logger.warning("Failed to extract list string from response")
            return None

        matched_list_string = match.group(1).strip()
        try:
            list_data = literal_eval(matched_list_string)
        except Exception:
            logger.warning("Failed to convert list string to python list")
            return None
        if not isinstance(list_data, list):
            logger.warning("Parsed data is not a list-type")

        return list_data

    @staticmethod
    def convert_cases_to_examples(cases: List[Case]) -> str:
        if not cases:
            return ""
        examples_list = [
            f"example {i + 1}:\n" \
            f"[question]: {TuneUtils.get_input_string_from_case(case)}\n" \
            f"[expected answer]: {TuneUtils.get_output_string_from_message(case.label)}"
            for i, case in enumerate(cases)
        ]
        return "\n".join(examples_list)
