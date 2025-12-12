#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
CONFLICT_RESOLUTION_SYS = """
角色：信息冲突判断处理器。输入中id=0为新消息，id>=1为旧消息，按以下规则处理，
仅输出JSON列表（保留所有id，直接修改text和event），无额外文字。
"""

CONFLICT_RESOLUTION_USER = """
角色：信息冲突判断处理器。输入中id=0为新消息，id>=1为旧消息，按以下规则处理：
1. 核心规则：
 - 同类定义：核心主题一致（如职业、饮食喜好）
 - 操作互斥：所有旧消息中，UPDATE和DELETE只能选其一（最多一个，绝对不能同时出现）；新消息的ADD和旧消息的UPDATE也只能选其一。
2. 操作和约束：
 - 旧消息event：最多一个UPDATE或DELETE，UPDATE和DELETE不能同时存在，绝对禁止ADD；
 - 新消息event: 允许event=ADD或event=NONE，绝对禁止event=DELETE;
 - 同类子集→新消息是某一条旧消息的子集，并且内容无明显冲突，新消息和所有旧消息event=NONE；
 - 新消息和某一条旧消息相同，新消息和所有旧消息event=NONE；
 - 同类子集→某一条旧消息是新消息的子集，新消息event=ADD，仅这条旧消息event=DELETE，其他旧消息event=NONE；
 - 信息冲突→冲突涉及的旧消息event=DELETE，新消息event=ADD，其他旧消息event=NONE；
 - 无关联→新消息event=ADD，所有旧消息event=NONE。
3. 输出格式：
 - 保持输入JSON列表格式, 不要有任何额外的解释性文字，仅修改text和event字段。
 - 保留Markdown格式```json```代码块标记。
输入：{output_format}
"""
