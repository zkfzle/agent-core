#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
from typing import List
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.config.config import Config
from openjiuwen.core.memory.prompt.categorizer import CATEGORIZATION_PROMPT


class Categorizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_model_input(messages: List[BaseMessage],
                      history_messages: List[BaseMessage],
                      prompt: str) -> List[dict]:
        history = ""
        if history_messages and len(history_messages) > 0:
            for msg in history_messages:
                history += f"{msg.role}: {msg.content}\n"
        conversation = ""
        for msg in messages:
            conversation += f"{msg.role}: {msg.content}\n"
        model_input = [{
            "role": "system",
            "content": prompt
        }]
        user_input = {}
        if history != "":
            user_input["historical_memory"] = history
        user_input["current_memory"] = conversation
        model_input.append({
            "role": "user",
            "content": json.dumps(user_input, ensure_ascii=False)
        })
        return model_input

    @staticmethod
    def get_categories(
        messages: List[BaseMessage],
        history_messages: List[BaseMessage],
        base_chat_model: BaseModelClient,
        config: Config,
        retries: int = 3
    ) -> List[str]:
        model_input = Categorizer.get_model_input(
            messages,
            history_messages,
            CATEGORIZATION_PROMPT,
        )
        logger.debug(f"Start to get categories, model_input: {model_input}")
        for attempt in range(retries):
            try:
                response = base_chat_model.invoke(config.model_name, model_input).content
                categories = json.loads(response)
                logger.debug(f"Succeed to get categories, output: {categories}")
                if isinstance(categories, list):
                    return categories
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    continue
                logger.error(f"categories model output format error: {e.msg}")
        return []
        
    @staticmethod
    async def aget_categories(
        messages: List[BaseMessage],
        history_messages: List[BaseMessage],
        base_chat_model: BaseModelClient,
        config: Config,
        retries: int = 3
    ) -> List[str]:
        model_input = Categorizer.get_model_input(
            messages,
            history_messages,
            CATEGORIZATION_PROMPT,
        )
        logger.debug(f"Start to get categories, model_input: {model_input}")
        for attempt in range(retries):
            try:
                response = await base_chat_model.ainvoke(config.model_name, model_input)
                categories = json.loads(response.content)
                logger.debug(f"Succeed to get categories, output: {categories}")
                if isinstance(categories, list):
                    return categories
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    continue
                logger.error(f"categories model output format error: {e.msg}")
        return []
    