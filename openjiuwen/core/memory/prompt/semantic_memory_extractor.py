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
语义记忆定义：在对话信息中得到的，现实世界中确定的事实或者概念；强调对于实体、概念本身的定义或者关系的描述"""

SEMANTIC_MEMORY_JSON_FORMAT = """
"semantic_memory": [str]{comma}"""