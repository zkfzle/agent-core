import json
from typing import List, Dict
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage
from openjiuwen.core.memory.memory_logging import get_logger
from openjiuwen.core.memory.config.config import Config
from openjiuwen.core.memory.generation.categorizer import Categorizer
from openjiuwen.core.memory.prompt.user_profile_extractor import USER_PROFILE_EXTRACTOR_PROMPT

logger = get_logger()


def _get_message(user_define: Dict[str, str] = None) -> str:
    if user_define and len(user_define) > 0:
        user_define_description = ""
        user_define_format = ""
        for key in user_define.keys():
            value = user_define[key]
            user_define_description += f"    *   **{key}:** {value}等相关信息\n"
            user_define_format += f',\n    "{key}": []'
        sym_prompt = USER_PROFILE_EXTRACTOR_PROMPT.format(
            user_define_description=user_define_description,
            user_define_format=user_define_format
        )
    else:
        sym_prompt = USER_PROFILE_EXTRACTOR_PROMPT.format(
            user_define_description="",
            user_define_format=""
        )
    return sym_prompt


class UserProfileExtractor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def GetUserProfile(
        messages: List[BaseMessage],
        history_messages: List[BaseMessage],
        base_chat_model: BaseModelClient,
        config: Config,
        user_define: Dict[str, str] = None,
        retries: int = 3
    ) -> Dict[str, str]:
        sym_prompt = _get_message(user_define)
        model_input = Categorizer.GetModelInput(
            messages,
            history_messages,
            sym_prompt
        )
        for attempt in range(retries):
            try:
                response = base_chat_model.invoke(config.model_name, model_input).content
                result = json.loads(response)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    continue
                logger.error(f"user profile extractor model output format error: {e.msg}")
        return {}

    @staticmethod
    async def aGetUserProfile(
        messages: List[BaseMessage],
        history_messages: List[BaseMessage],
        base_chat_model: BaseModelClient,
        config: Config,
        user_define: Dict[str, str] = None,
        retries: int = 3
    ) -> Dict[str, str]:
        sym_prompt = _get_message(user_define)
        model_input = Categorizer.GetModelInput(
            messages,
            history_messages,
            sym_prompt
        )
        for attempt in range(retries):
            try:
                response = await base_chat_model.ainvoke(config.model_name, model_input).content
                result = json.loads(response)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    continue
                logger.error(f"user profile extractor model output format error: {e.msg}")
        return {}
