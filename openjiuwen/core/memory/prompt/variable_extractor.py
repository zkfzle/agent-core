#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
EXTRACT_VARIABLES_PROMPT = """
# 任务描述
假设你是信息提取领域专家。
# 任务定义
分析LLM智能体的对话内容，根据预定义的变量名称和变量描述，完成目标变量值提取。
变量定义如下:
<定义>
{variables}
</定义>
# 输出格式
 - 提取变量值并填充以下JSON结构，输出纯净的，可直接解析的JSON对象，不要有任何额外的解释性文字:
{variables_output_format}
 - 若无法提取某个变量的值，则在对应字段中填入"null"。
 - 保留Markdown格式```json``` 代码块标记。
"""
