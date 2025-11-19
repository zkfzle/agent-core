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
    def GetModelInput(messages: List[BaseMessage],
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
    def GetCategories(
        messages: List[BaseMessage],
        history_messages: List[BaseMessage],
        base_chat_model: BaseModelClient,
        config: Config,
        retries: int = 3
    ) -> List[str]:
        model_input = Categorizer.GetModelInput(
            messages,
            history_messages,
            CATEGORIZATION_PROMPT,
        )
        for attempt in range(retries):
            try:
                response = base_chat_model.invoke(config.model_name, model_input).content
                categories = json.loads(response)
                if isinstance(categories, list):
                    return categories
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    continue
                logger.error(f"categories model output format error: {e.msg}")
        return []
        
    @staticmethod
    async def aGetCategories(
        messages: List[BaseMessage],
        history_messages: List[BaseMessage],
        base_chat_model: BaseModelClient,
        config: Config,
        retries: int = 3
    ) -> List[str]:
        model_input = Categorizer.GetModelInput(
            messages,
            history_messages,
            CATEGORIZATION_PROMPT,
        )
        for attempt in range(retries):
            try:
                response = await base_chat_model.ainvoke(config.model_name, model_input).content
                categories = json.loads(response)
                if isinstance(categories, list):
                    return categories
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    continue
                logger.error(f"categories model output format error: {e.msg}")
        return []
    