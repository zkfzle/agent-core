import json
from enum import Enum
from typing import List
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.config.config import Config
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
    def check_conflict(
        old_messages: List[str],
        new_message: str,
        base_chat_model: BaseModelClient,
        config: Config,
        retries: int = 3
    ) -> list[dict]:
        """
        Check for conflicts between old messages and a new message.
        
        Args:
            old_messages (List[str]): List of old messages.
            new_message (str): The new message to check against old messages.
            base_chat_model (BaseModelClient): The chat model to use for processing.
            config (Config): Configuration for the chat model.
            retries (int, optional): Number of retries for the operation. Defaults to 3.
        
        Returns:
            list[dict]: A list of dictionaries representing the conflict resolution results.
        """
        messages = _get_message(old_messages, new_message)
        for attempt in range(retries):
            try:
                response = base_chat_model.invoke(model_name=config.model_name, messages=messages).content
                result = json.loads(str(response).strip().replace("'", '"'))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError as e:
                if attempt <= retries - 1:
                    continue
                logger.error(f"categories model output format error: {e.msg}")
        return []
    
    @staticmethod
    async def acheck_conflict(
        old_messages: List[str],
        new_message: str,
        base_chat_model: BaseModelClient,
        config: Config,
        retries: int = 3
    ) -> list[dict]:
        messages = _get_message(old_messages, new_message)
        for attempt in range(retries):
            try:
                response = await base_chat_model.ainvoke(model_name=config.model_name, messages=messages).content
                result = json.loads(str(response).strip().replace("'", '"'))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError as e:
                if attempt <= retries - 1:
                    continue
                logger.error(f"categories model output format error: {e.msg}")
        return []
