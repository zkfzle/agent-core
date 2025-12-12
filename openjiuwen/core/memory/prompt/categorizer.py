#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
CATEGORIZATION_PROMPT = """
# 任务描述
你是一个记忆分类专家，你的任务是分析当前消息`current_messages`，结合历史消息`historical_messages`，并将其分类到以下一个或多个类别中：
1. **user_profile**: 关于用户的具体信息，包括性别，年龄，兴趣爱好，资产信息，财务状况，社交关系，沟通方式或家庭成员等相关信息。
2. **semantic_memory**: 不与特定时刻或事件相关的永恒事实知识，这包括用户表达或学习到的普遍真理，定义，规则或知识。
# 输出格式
* 提取记忆分类填充以下JSON结构，不要有任何额外的解释性文字：
{"categories":[]}
* 一个记忆可能属于多个类别，类别类型取于以下字段：'"user_profile"', '"semantic_memory"'。例如：{"categories":["user_profile"]}， {"categories":["user_profile", "semantic_memory"]}。
* 如果不属于任何类别，请返回空列表{"categories":[]}。
* 保留Markdown格式```json``` 代码块标记。
"""
