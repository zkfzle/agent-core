#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
from typing import List, Tuple
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage
from openjiuwen.core.utils.llm.output_parser.json_output_parser import JsonOutputParser
from openjiuwen.core.common.logging import logger
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
            user_input["historical_messages"] = history
        user_input["current_messages"] = conversation
        model_input.append({
            "role": "user",
            "content": json.dumps(user_input, ensure_ascii=False)
        })
        return model_input

    @staticmethod
    async def get_categories(
            messages: List[BaseMessage],
            history_messages: List[BaseMessage],
            base_chat_model: Tuple[str, BaseModelClient],
            retries: int = 3
    ) -> List[str]:
        model_input = Categorizer.get_model_input(
            messages,
            history_messages,
            CATEGORIZATION_PROMPT,
        )
        model_name, model_client = base_chat_model
        logger.debug(f"Start to get categories, input: {model_input}")
        parser = JsonOutputParser()
        for attempt in range(retries):
            try:
                response = await model_client.ainvoke(model_name, model_input)
                categories = await parser.parse(response.content)
                logger.debug(f"Succeed to get categories, result: {categories}")
                if isinstance(categories, dict) and "categories" in categories.keys():
                    return categories["categories"]
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    continue
                logger.error(f"categories model output format error: {e.msg}")
        return []
