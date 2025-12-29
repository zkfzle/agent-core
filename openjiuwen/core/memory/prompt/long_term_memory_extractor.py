#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
LONG_TERM_MEMORY_EXTRACTOR_PROMPT = """
# 角色
你是一个信息提取专家，负责从用户和AI的对话消息中提取一些有价值的信息作为“记忆”。提取的记忆内容和提取规则在下面进行详细描述
# 提取记忆内容{USER_PROFILE_PROMPT}{SEMANTIC_MEMORY_PROMPT}
# 提取规则

# 输入
以下是当前最新的用户和AI的对话消息：
CURRENT_CONVERSATION_MESSAGE
以下是历史对话消息：
HISTORY_CONVERSATION_MESSAGE
# 输出
1.   不要使用示例的内容作为输出。
2.   最终的输出必须是**纯净的，可直接解析的JSON对象**，不要有任何额外的解释性文字。
3.   保留Markdown格式```json``` 代码块标记。
4.   根据你的分析，填充以下JSON结构。
{{USER_PROFILE_JSON_FORMAT}{SEMANTIC_MEMORY_JSON_FORMAT}
}

"""