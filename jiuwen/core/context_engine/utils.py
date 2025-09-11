#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from enum import Enum
import re
import json
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Union, Dict, Any, Optional, List

from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.context_engine.base import ContextVariable


class ContextUtils:
    @staticmethod
    def convert_messages_to_string(messages: Union[List[BaseMessage], str]):
        if isinstance(messages, str):
            return messages
        return "\n".join([f"[{msg.role}]:{msg.content}" for msg in messages])

    @staticmethod
    def convert_variables_to_dict(variables: Dict[str, ContextVariable]) -> Dict[str, str]:
        variables_dict = {}
        for name, var in variables.items():
            variables_dict[name] = var.get_value() or ""
        return variables_dict

    @staticmethod
    def parse_json_string_from_llm_response(json_str: str) -> Optional[Dict[str, str]]:
        if not json_str:
            return None
        pattern = r"```json(.*?)```"
        match = re.search(pattern, json_str, re.DOTALL)

        if match:
            json_string = match.group(1).strip()
            try:
                parsed_data = json.loads(json_string)
            except json.decoder.JSONDecodeError:
                logger.warning("Failed to decode json string")
                return None
            return parsed_data

        logger.warning("No valid json string found")
        return None