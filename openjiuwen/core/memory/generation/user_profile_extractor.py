import json
from typing import List, Dict
from openjiuwen.core.utils.llm.base import BaseChatModel
from openjiuwen.core.utils.llm.messages import BaseMessage
from openjiuwen.core.memory.memory_logging import get_logger
from openjiuwen.core.memory.config.config import Config
from .categorizer import Categorizer

logger = get_logger()

USER_PROFILE_EXTRACTOR_PROMPT = """
你是一个用户画像分析专家，你的任务是从当前记忆`current_memory`中(结合历史记忆`historical memory`)精准提取用户画像信息。信息分为两大类：核心信息和非核心信息。
# 提取规则：
1.   **用户画像分类**：
     *    **personal_information**: 提取用户明确的姓名，年龄/年龄段（如20多岁），职业，所在地（如北京/上海），家庭成员（不包括朋友）。
     *    **interest_hobbies**: 提取用户明确提及或强烈暗示的，自己的兴趣爱好（如喜欢打篮球，每周都去画画）。
     *    **social_information**: 提取朋友，同事等关系及相关信息。
     *    **asset_information**: 仅当用户明确提及时提取，如房产、车辆，特定品牌消费（如用苹果电脑）。
     *    **other_information**: 提取包括健康状况，性格特征，行为模式，职业背景，短期和长期目标，近期任务或活动，以及与AI互动的偏好信息。
     {user_define_description}
2.   注意不要提取以下信息：联系方式，外貌特点，忌日，生理期，物品位置，过敏信息。

3.   仅从当前记忆`current_memory`中提取任何相关的用户画像信息，不要提取与用户画像无关的信息，不要提取历史记忆`historical_memory`中的信息。

4.   注意**以用户视角书写**（比如"用户的年龄是..."，"用户的姓名是..."，"用户喜欢..."，"用户有..."，"用户的xx是..."），不要省略关键信息，不要摘要，语言简洁，语气客观。

5.   将提取出的信息点归类到不同的画像维度中，每个类别可能生成多个用户画像。
# 输出格式要求
1.   最终的输出必须是**纯净的，可直接解析的JSON对象**，不要有任何额外的Markdown格式（如 ```json``` ）或解释性文字。
2.   根据你的分析，填充以下JSON结构，确保所有字段即使为空也用空字符串或空列表表示。
{{
    "personal_information": [],
    "interest_hobbies": [],
    "social_information": [],
    "asset_information": [],
    "other_information": []{user_define_format}
}}
"""


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
        base_chat_model: BaseChatModel,
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
        base_chat_model: BaseChatModel,
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
