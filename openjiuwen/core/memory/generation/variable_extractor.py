import json
from typing import Any
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage
from openjiuwen.core.memory.config.config import Config
from openjiuwen.core.memory.generation.memory_info import (
    ExtractedData,
    ExtractedDataType
)

from openjiuwen.core.memory.prompt.variable_extractor import (
    EXTRACT_VARIABLES_USER_SUMMARY,
    EXTRACT_VARIABLES_USER,
    EXTRACT_VARIABLES_SYS,
    EXTRACT_VARIABLES_USER_SUMMARY_zh_CN,
    EXTRACT_VARIABLES_USER_zh_CN,
    EXTRACT_VARIABLES_SYS_zh_CN
)

from openjiuwen.core.common.logging import logger

class ComprehensionExtractor:
    def __init__(self):
        pass

    @staticmethod
    def extract(
        message: list[BaseMessage],
        history_summary: BaseMessage,
        base_chat_model: BaseModelClient,
        config: Config
    ) -> list[ExtractedData]:
        """Extract variables from the given message using LLM.
        
        Args:
            message (list[BaseMessage]): The current message to extract variables from.
            history_summary (BaseMessage): The summary of historical messages.
            base_chat_model (BaseModelClient): The chat model to use for extraction.
            config (Config): Configuration for the extraction process.
        
        Returns:
            list[ExtractedData]: A list of extracted data objects.
        """
        if config.variables_key is None or config.variables_key == {}:
            return []
        variables_dict = {
            "variables_enum": "",
            "variables_str": "",
            "variables_user": set()
        }
        if 'user' in config.variables_key:
            for var in config.variables_key['user']:
                variables_dict["variables_user"].add(var['name'])
                variables_dict["variables_str"] += f"{var['name']}({var['description']}),"
                variables_dict["variables_enum"] += "{\"" + var['name'] + "\": {\"value\": \"string\"}}\n"
        conversation = ""
        for msg in message:
            conversation += f"{msg.role}: {msg.content}\n"

        # Construct prompts
        if history_summary.content != "":
            user_message = EXTRACT_VARIABLES_USER_SUMMARY_zh_CN if config.language == "zh-CN"\
                else EXTRACT_VARIABLES_USER_SUMMARY
            user_message = user_message.format(
                conversation=conversation,
                summary=history_summary.content,
                variables=variables_dict["variables_str"],
                variables_enum=variables_dict["variables_enum"]
            )
        else:
            user_message = EXTRACT_VARIABLES_USER_zh_CN if config.language == "zh-CN" else EXTRACT_VARIABLES_USER
            user_message = user_message.format(
                conversation=conversation,
                variables=variables_dict["variables_str"],
                variables_enum=variables_dict["variables_enum"]
            )
        sys_message = EXTRACT_VARIABLES_SYS_zh_CN if config.language == "zh-CN" else EXTRACT_VARIABLES_SYS

        response = base_chat_model.invoke(
            config.model_name,
            [{
                "role": "system",
                "content": sys_message
            },
            {
                "role": "user",
                "content": user_message
            }]
        )

        # Parse response
        extract_result = []
        try:
            if len(str(response.content).strip()) == 0:
                return []
            response = json.loads(str(response.content).strip())
            for key, value in response.items():
                key = str(key).strip()
                if not ComprehensionExtractor._check_value(value):
                    continue
                value = str(value.get("value", "")).strip()
                if len(value) > 0 and value.lower() != "null":
                    if key in variables_dict["variables_user"]:
                        extract_result.append(
                            ExtractedData(
                                type=ExtractedDataType.USER,
                                key=key,
                                value=value
                            )
                        )
            return extract_result
        except Exception as e:
            logger.error(f"LLM返回的json格式有误: {e}")
            return []
        
    @staticmethod
    def _check_value(value: Any) -> bool:
        if value is None or not isinstance(value, dict) or value.get("value", "") is None:
            return False
        if value.get("value", "").lower() == "none":
            return False
        return True
