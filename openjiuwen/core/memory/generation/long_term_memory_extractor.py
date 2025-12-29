#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
from typing import List, Dict, Tuple, Any

from langchain_experimental.graph_transformers.llm import system_prompt

from openjiuwen.core.memory.mem_unit.memory_unit import MemoryType
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage
from openjiuwen.core.utils.llm.output_parser.json_output_parser import JsonOutputParser
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.generation.common import build_model_input, get_str_messages
from openjiuwen.core.memory.prompt.user_profile_extractor import USER_PROFILE_PROMPT, USER_PROFILE_JSON_FORMAT
from openjiuwen.core.memory.prompt.semantic_memory_extractor import SEMANTIC_MEMORY_PROMPT, SEMANTIC_MEMORY_JSON_FORMAT
from openjiuwen.core.memory.prompt.long_term_memory_extractor import LONG_TERM_MEMORY_EXTRACTOR_PROMPT

def handle_user_profile_prompt(sys_prompt: str, index: int, last_one: bool, user_define: Dict[str, str] = None) -> str:
    user_profile_prompt = ""
    user_profile_json_format = ""
    if index > 0:
        user_define_description = ""
        user_define_format = ""
        if user_define and len(user_define) > 0:
            for key in user_define.keys():
                value = user_define[key]
                user_define_description += f"    *   **{key}:** {value}等相关信息\n"
                user_define_format += f',\n    "{key}": []'
        user_profile_prompt = USER_PROFILE_PROMPT.format(
            index=index,
            user_define_description=user_define_description,
        )
        user_profile_json_format = USER_PROFILE_JSON_FORMAT.format(
            user_define_format=user_define_format,
            comma='' if last_one else ',',
        )
    sys_prompt = sys_prompt.replace("{USER_PROFILE_PROMPT}", user_profile_prompt)
    sys_prompt = sys_prompt.replace("{USER_PROFILE_JSON_FORMAT}", user_profile_json_format)
    return sys_prompt

def handle_semantic_memory(sys_prompt: str, index: int, last_one: bool) -> str:
    semantic_memory_prompt = ""
    semantic_memory_json_format = ""
    if index > 0:
        semantic_memory_prompt = SEMANTIC_MEMORY_PROMPT.format(
            index=index,
        )
        semantic_memory_json_format = SEMANTIC_MEMORY_JSON_FORMAT.format(
            comma='' if last_one else ',',
        )
    sys_prompt = sys_prompt.replace("{SEMANTIC_MEMORY_PROMPT}", semantic_memory_prompt)
    sys_prompt = sys_prompt.replace("{SEMANTIC_MEMORY_JSON_FORMAT}", semantic_memory_json_format)
    return sys_prompt

def _get_message(user_define: Dict[str, str] = None, categories: list[str] | None = None) -> str:
    sys_prompt = LONG_TERM_MEMORY_EXTRACTOR_PROMPT
    index = 1
    if "user_profile" in categories:
        sys_prompt = handle_user_profile_prompt(sys_prompt, index, index == len(categories), user_define)
        index += 1
    else:
        sys_prompt = handle_user_profile_prompt(sys_prompt, 0, index == len(categories))
    if "semantic_memory" in categories:
        sys_prompt = handle_semantic_memory(sys_prompt, index, index == len(categories))
        index += 1
    else:
        sys_prompt = handle_semantic_memory(sys_prompt, 0, index == len(categories))
    return sys_prompt


class LongTermMemoryExtractor:
    def __init__(self) -> None:
        pass

    @staticmethod
    async def extract_long_term_memory(
            categories: list[str],
            history_messages: list[BaseMessage],
            messages: list[BaseMessage],
            # user_id: str,
            # group_id: str,
            base_chat_model: Tuple[str, BaseModelClient],
            message_mem_id: str,
            user_define: dict[str, str] = None,
            retries: int = 3
    ) -> Dict[str, Any]:
        if not categories or len(categories) == 0:
            return {}
        for category in categories:
            try:
                MemoryType(category)
            except ValueError:
                logger.warning(f"invalid category, remove {category}")
                categories.remove(category)
        if len(categories) == 0:
            return {}
        sys_prompt = _get_message(user_define, categories)
        conversation, history = get_str_messages(messages, history_messages)
        sys_prompt = sys_prompt.replace("CURRENT_CONVERSATION_MESSAGE", conversation)
        sys_prompt = sys_prompt.replace("HISTORY_CONVERSATION_MESSAGE", history)
        model_input = [{
            "role": "user",
            "content": sys_prompt
        }]
        # model_input = build_model_input(
        #     messages,
        #     history_messages,
        #     sys_prompt
        # )
        logger.debug(f"Start to get user profile, input: {model_input}")
        model_name, model_client = base_chat_model
        parser = JsonOutputParser()
        for attempt in range(retries):
            try:
                response = await model_client.ainvoke(model_name, model_input)
                result = await parser.parse(response.content)
                logger.debug(f"Succeed to get user profile, result: {result}")
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    continue
                logger.error(f"user profile extractor model output format error: {e.msg}")
        return {}
