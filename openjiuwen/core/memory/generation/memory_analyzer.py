#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import json
from typing import List, Tuple
from pydantic import BaseModel, Field

from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage
from openjiuwen.core.utils.llm.output_parser.json_output_parser import JsonOutputParser
from openjiuwen.core.memory.config.config import MemoryConfig
from openjiuwen.core.memory.prompt.memory_analyzer import MEMORY_ANALYZER_PROMPT, VARIABLES_DESCRIPTION_TEMPLATE_PROMPT


class VariableResult(BaseModel):
    variable_key: str = Field(default="", description="variable key")
    variable_value: str = Field(default="", description="variable value")


class MemoryAnalyzerResult(BaseModel):
    categories: List[str] = Field(default=[])
    variables: List[VariableResult] = Field(default=[])


class MemoryAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    async def analyze(
            messages: List[BaseMessage],
            history_messages: List[BaseMessage],
            base_chat_model: Tuple[str, BaseModelClient],
            memory_config: MemoryConfig,
            retries: int = 3
    ) -> MemoryAnalyzerResult | None:
        if len(messages) == 0:
            logger.warning("No messages to analyze")
            return None
        model_input = MemoryAnalyzer._build_model_input(
            messages=messages,
            history_messages=history_messages,
            memory_config=memory_config,
        )

        model_name, model_client = base_chat_model
        logger.debug(f"Start to analyze, input: {model_input}")
        parser = JsonOutputParser()
        for attempt in range(retries):
            try:
                response = await model_client.ainvoke(model_name, model_input)
                res = await parser.parse(response.content)
                logger.debug(f"Succeed to analyze, result: {res}")
                return MemoryAnalyzerResult.model_validate(res)
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    continue
                logger.error(f"categories model output format error: {e.msg}")
        return []

    @staticmethod
    def _build_model_input(
            messages: List[BaseMessage],
            history_messages: List[BaseMessage],
            memory_config: MemoryConfig
    ) -> List:
        variables_description, variables_output_format = MemoryAnalyzer._build_variable_prompt(memory_config)
        sys_prompt = MEMORY_ANALYZER_PROMPT.replace("VARIABLES_DESCRIPTION_TEMPLATE", variables_description)
        sys_prompt = sys_prompt.replace("VARIABLES_OUTPUT_TEMPLATE", variables_output_format)
        model_input = [{
            "role": "system",
            "content": sys_prompt
        }]
        user_input = ""
        history = ""
        conversation = ""
        for msg in history_messages:
            history += f"{msg.role}: {msg.content}\n"
        for msg in messages:
            conversation += f"{msg.role}: {msg.content}\n"
        if history != "":
            user_input += (f"如果当前输入与历史消息有关联，可参考历史消息，历史消息如下：\n"
                           f"<historical_messages>{history}</historical_messages>\n")
        user_input += f"现在开始：请根据设定的规则处理以下输入并生成出输出：\n<current_messages>{conversation}</current_messages>\n"
        model_input.append({
            "role": "user",
            "content": user_input
        })
        return model_input

    @staticmethod
    def _build_variable_prompt(
            memory_config: MemoryConfig
    ) -> Tuple[str, str]:
        if len(memory_config.mem_variables) == 0:
            return "", ""

        variables_description = []
        variables_output_format = []
        for key, value in memory_config.mem_variables.items():
            variables_description.append({
                "variable_key": key,
                "variable_value": value
            })

            variables_output_format.append({
                "variable_key": key,
                "variable_value": ""
            })

        variables_description_json = json.dumps(variables_description, ensure_ascii=False)
        variables_output_format_json = json.dumps(variables_output_format, ensure_ascii=False)
        return VARIABLES_DESCRIPTION_TEMPLATE_PROMPT.replace("VARIABLES_DEFINE_TEMPLATE",
                                                             variables_description_json), \
            ",\n\"variables\":" + variables_output_format_json
