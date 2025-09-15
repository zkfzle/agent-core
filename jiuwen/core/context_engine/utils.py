#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from enum import Enum
import re
import json
from typing import Union, Dict, Optional, List

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
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
    def convert_dict_to_variables(variables_dict: Dict[str, str]) -> Dict[str, ContextVariable]:
        variables = {}
        for name, value in variables_dict.items():
            variables[name] = ContextVariable(name=name, value=value or "")
        return variables

    @staticmethod
    def convert_dict_to_message(message_dict: Dict[str, str]) -> BaseMessage:
        role = message_dict.get("role", "")
        if not role:
            raise JiuWenBaseException(
                error_code=StatusCode.CONTEXT_ENGINE_MESSAGE_PROCESS_ERROR.code,
                message=StatusCode.CONTEXT_ENGINE_MESSAGE_PROCESS_ERROR.errmsg.format(
                    error_msg="role cannot be empty"))
        message_types = {
            "user": HumanMessage,
            "system": SystemMessage,
            "assistant": AIMessage,
            "tool": ToolMessage
        }
        try:
            return message_types.get(role, BaseMessage)(**message_dict)
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.CONTEXT_ENGINE_MESSAGE_PROCESS_ERROR.code,
                message=StatusCode.CONTEXT_ENGINE_MESSAGE_PROCESS_ERROR.errmsg.format(
                    error_msg="convert dict to message failed")) from e


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