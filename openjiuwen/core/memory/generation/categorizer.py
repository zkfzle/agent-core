import json
from typing import List
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import BaseMessage
from openjiuwen.core.memory.memory_logging import get_logger
from openjiuwen.core.memory.config.config import Config

logger = get_logger()

CATEGORIZATION_PROMPT = """
你是一个记忆分类专家，你的任务是分析当前记忆`current_memory`，结合历史记忆`historical memory`，并将其分类到以下一个或多个类别中：
1. **user_profile**: 关于用户的具体信息，包括性别，年龄，兴趣爱好，喜欢的食物，资产信息，财务状况，社交关系，沟通方式或家庭成员。
2. **semantic_memory**: 不与特定时刻或事件相关的永恒事实知识，这包括用户表达或学习到的普遍真理，定义，规则或知识。
* 一个记忆可能属于多个类别，类别是以下之一：'"user_profile"', '"semantic_memory"'或两者兼有（例如，'["user_profile", "semantic_memory"]'），如果不属于任何类别，请返回空列表`[]`。
* 历史记忆`historical memory`可能为空。
"""


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
    