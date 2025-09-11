#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Optional, Dict, List
import json
import re

from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.context_engine.base import ContextWindow, ContextVariable
from jiuwen.core.context_engine.config import BaseAsyncProcessorConfig
from jiuwen.core.context_engine.processor.asynch.base import AsyncContextProcess
from jiuwen.core.context_engine.processor.factory import ProcessorFactory


class VariableExtractorConfig(BaseAsyncProcessorConfig):
    pass


DEFAULT_EXTRACTOR_TEMPLATE = \
"""
You are an assistant for variable extraction. Your task is to extract the content of variables from the given information based on the variable names and descriptions.
The user provides the variable information to be extracted:
{{user_variables}}.

Below is the conversation history:
{{history}}.

Based on the above information, extract the variable content and output it in JSON format.
If there is no suitable content matching the variable name, display null.
The output format should be:
```json
{
    "variable_name1": "variable_content1",
    "variable_name2": "variable_content2",
    "variable_name3": null
}
```
Ensure that the output format strictly adheres to JSON format and always includes the ```json``` tag.
"""


@ProcessorFactory.register("variable_extractor", VariableExtractorConfig)
class VariableExtractor(AsyncContextProcess):
    def __init__(self, config: VariableExtractorConfig):
        super().__init__(config)

    @staticmethod
    def need_llm() -> bool:
        return True

    async def arun(self, input_data: ContextWindow) -> ContextWindow:
        variables = input_data.variables
        output_data = input_data
        if not variables or not input_data.chat_history:
            return output_data

        variables_str = self.__generate_variable_string(variables)
        history_str = self.__generate_history_string(input_data.chat_history)

        prompt = DEFAULT_EXTRACTOR_TEMPLATE.replace("{{user_variables}}", variables_str) \
                                           .replace("{{history}}", history_str)
        response = await self._llm.ainvoke(prompt)
        extracted_variables = self.__parse_extract_result(response.content)
        if not extracted_variables:
            return output_data
        self.__convert_result_to_variables(output_data, extracted_variables)
        return output_data

    def is_ready(self, input_data: ContextWindow) -> bool:
        return True

    def update_strategy(self) -> str:
        return "update_variables"

    def __generate_variable_string(self, variables: Dict[str, ContextVariable]) -> str:
        variable_string = ""
        for name, variable in variables.items():
            variable_string += f"name: {variable.name}, description: {variable.description}\n"
        return variable_string

    def __generate_history_string(self, history: List[BaseMessage]) -> str:
        history_string = ""
        for message in history:
            history_string += f"role: {message.role}, content: {message.content}\n"
        return history_string

    def __parse_extract_result(self, json_str: str) -> Optional[Dict[str, str]]:
        """Parse json string"""
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

    def __convert_result_to_variables(self, context_window: ContextWindow, variables_dict: Dict[str, str]):
        for name, content in variables_dict.items():
            if name in context_window.variables:
                context_window.variables[name].value = content