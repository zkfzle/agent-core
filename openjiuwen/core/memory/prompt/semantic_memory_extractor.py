#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# SEMANTIC_MEMORY_PROMPT123="""
# # 定义
# 语义记忆：在对话信息中得到的，现实世界中确定的事实或者概念；强调对于实体、概念本身的定义或者关系的描述
# # 输入
# 一组最新的用户和AI的对话消息，可能包含历史消息
# # 输出
# 以一个可直接解析的json格式表示从对话消息中提取出的语义记忆，json的key为"semantic_memory", value为一个list[str]；
# 最后用Markdown格式```json``` 代码块标记。
# """

SEMANTIC_MEMORY_PROMPT = """
{index}、语义记忆
**语义记忆定义**：在对话信息中得到的，现实世界中普遍适用的“事实/概念/规则”；强调对于实体、概念本身的定义或者关系的描述
**语义记忆判定标准**：
可长期复用的一般性知识，去掉上下文后仍对任何用户通用；
定义、原理、分类、数值常量、因果/条件规则等，一般也都属于语义记忆
不包含“我/我们/今天/上次”等，与用户个人强相关的信息或者时空锚点。
**语义记忆提取要求**：
原子化：一条只表达一个事实。
去重：若同义合并，只保留最简洁表述。
如无合格语义记忆，输出中应写为空数组[]。
"""

SEMANTIC_MEMORY_JSON_FORMAT = """
"semantic_memory": [str]{comma}"""