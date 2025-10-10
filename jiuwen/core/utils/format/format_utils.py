"""格式化工具类模块"""
import json
from typing import List

from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.messages import BaseMessage


class FormatUtils:
    """输入输出格式化工具类"""
    @staticmethod
    def create_llm_inputs(system_prompt: List[BaseMessage], chat_history: List[BaseMessage]) -> List[BaseMessage]:
        """创建LLM输入消息列表

        Args:
            system_prompt: 系统提示消息列表
            chat_history: 对话历史消息列表（已包含当前用户输入）

        Returns:
            完整的LLM输入消息列表
        """
        # 创建新的消息列表，避免修改原始chat_history
        result_messages = []

        # 添加系统提示（如果chat_history中没有system消息）
        if not chat_history or chat_history[0].role != "system":
            result_messages.extend(system_prompt)

        # 添加对话历史
        result_messages.extend(chat_history)

        return result_messages

    @staticmethod
    def json_loads(arguments: str) -> dict:
        """安全的JSON解析"""
        result = dict()
        try:
            result = json.loads(arguments, strict=False)
        except json.JSONDecodeError:
            logger.error(f"JSON parser error.")
        return result