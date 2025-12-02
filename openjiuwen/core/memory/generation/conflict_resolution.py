#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
from enum import Enum
from typing import List, Tuple
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.prompt.conflict_resolution import (
    CONFLICT_RESOLUTION_SYS, CONFLICT_RESOLUTION_USER)


class ConflictType(Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    UPDATE = "UPDATE"
    NONE = "NONE"


def _get_message(old_messages: List[str], new_message: str) -> list[dict]:
    output_format = [{
        "id": "0",
        "text": new_message,
        "event": "operation"
    }]
    for i in range(len(old_messages)):
        output_format.append({
            "id": str(i + 1),
            "text": old_messages[i],
            "event": "operation"
        })
    user_message = CONFLICT_RESOLUTION_USER.format(output_format=str(output_format))
    sys_message = CONFLICT_RESOLUTION_SYS
    return [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": user_message}
    ]


class ConflictResolution:
    def __init__(self):
        pass

    @staticmethod
    async def check_conflict(
            old_messages: List[str],
            new_message: str,
            base_chat_model: Tuple[str, BaseModelClient],
            retries: int = 3
    ) -> list[dict]:
        """
        Check for conflicts between old messages and a new message.

        Args:
            old_messages (List[str]): List of old messages.
            new_message (str): The new message to check against old messages.
            base_chat_model (Tuple[str, BaseModelClient]): The chat model to use for processing.
            retries (int, optional): Number of retries for the operation. Defaults to 3.

        Returns:
            list[dict]: A list of dictionaries representing the conflict resolution results.
        """
        model_name, model_client = base_chat_model
        messages = _get_message(old_messages, new_message)
        logger.debug(f"Start checking conflict, input messages: {messages}")
        for attempt in range(retries):
            try:
                response = await model_client.ainvoke(model_name, messages=messages)
                result = json.loads(str(response.content).strip().replace("'", '"'))
                logger.debug(f"Succeed to check conflict, result: {result}")
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError as e:
                if attempt <= retries - 1:
                    continue
                logger.error(f"categories model output format error: {e.msg}")
        return []
