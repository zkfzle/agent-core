#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from openjiuwen.core.utils.prompt.template.template import Template
from openjiuwen.core.utils.llm.messages import SystemMessage, HumanMessage


PROMPT_BUILD_GENERAL_META_SYSTEM_TEMPLATE = Template(content=[SystemMessage(content="""
以下是markdown的元模板：

## 人设
定义你将扮演的角色或身份
列举角色的专业技能或特长。

## 任务描述
清晰阐述角色旨在解决的问题和目标，以及预期对用户或系统的积极影响。

## 约束条件
在<任务描述>的基础上，需要补充说明任务的边界，以及用户的要求。比如字数要求、格式要求。
注意区分<输出格式>， 输出格式仅仅指格式要求的体现， 便于解析输出。
一般可以在<约束条件>中加上：
1. 按照<输出格式>输出
2. 按照<执行步骤>一步一步执行

## 执行步骤
介绍解决问题的基本方法。并且按步骤呈现。

## 输出格式
根据用户的需求，提供准确的输出格式。可以要求风格，字数，格式等。

请根据上述markdown的元模板，制作具体的模板内容。在生成过程中，请确保遵守以下指导原则：
1. 仅生成模板内容，避免添加不必要的信息。
2. 确保模板中包含用户要求中的关键信息。
3. 直接输出markdown内容，不要包含```markdown```代码块标记。

""")])

PROMPT_BUILD_GENERAL_META_USER_TEMPLATE = Template(content=[HumanMessage(content="""
用户的具体要求如下：
{{instruction}}
""")])


PROMPT_BUILD_PLAN_META_SYSTEM_TEMPLATE = Template(content=[SystemMessage(content="""
以下是markdown的元模板：

## 人设
- **角色与特性**：清晰揭示所扮演的角色及其背景故事，突出角色的独特性与任务目标。
- **核心技能与知识**：详细展示角色的关键能力及其在解决问题中的作用，具体包括：
  - 技能1: 对技能的详尽阐述和其在任务中的应用。
  - 技能2: 深入讲解另一技能或知识点及其重要性。

## 任务描述
清晰阐述角色旨在解决的问题和目标，以及预期对用户或系统的积极影响。

## 约束条件
在<任务描述>的基础上，需要补充说明任务的边界，以及用户的要求。
注意区分<输出格式>， 输出格式仅仅指格式要求的体现， 便于解析输出。
一般可以在<约束条件>中加上：
1. 按照<输出格式>输出
2. 按照<执行步骤>一步一步执行

## 执行步骤
介绍解决问题的基本方法。并且按步骤呈现。

## 输出格式
明确指出任务需要遵循的输出规范，确保输出内容结构合理、清晰可读。

请按照上述markdown的元模板，按照以下用户要求和可调用工具制作具体的模板内容。注意不要生成模板外的内容。请确保遵守以下指导原则：
1. 仅生成模板内容，避免添加不必要的信息。
2. 确保模板中包含用户要求中的关键信息。
3. 直接输出markdown内容，不要包含```markdown```代码块标记。
""")])

PROMPT_BUILD_PLAN_META_USER_TEMPLATE = Template(content=[HumanMessage(content="""
用户的要求：{{instruction}}

可以调用的工具：
{{tools}}
""")])