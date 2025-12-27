#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

MEMORY_ANALYZER_PROMPT = """
# 任务描述
你是一个记忆分析引擎，你的任务是分析当前消息`current_messages`，结合历史消息`historical_messages`，对用户画像进行多维度分析。

## 处理步骤
请严格按照以下步骤顺序执行：

### 步骤1：记忆分类
请将用户提供的消息进行分类，取值范围[`user_profile`]，填充到输出模板`categories`字段，结果可能属于多个类别；如果不属于任何类别，则结果为空`[]`。
1. `user_profile`: 指与用户相关的具体信息，包括但不限于以下方面:
 - 姓名、性别、年龄、职业、学历、居住地等个人信息
 - 兴趣爱好与生活习惯（如运动爱好、饮食习惯等）
 - 资产信息与财务状况（如收入、房产、车辆、投资、负债等）
 - 社交关系与沟通方式（如朋友、同事、社交习惯等）
 - 不属于以上类别但对用户有价值的其他信息
VARIABLES_DESCRIPTION_TEMPLATE
## 输出格式
1. 最终的输出必须是**纯净的，可直接解析的JSON对象**，不要有任何额外的解释性文字。
2. 保留Markdown格式```json``` 代码块标记。
3. 根据你的分析，填充以下JSON结构，不要修改JSON结构，确保所有字段有值(即使为空也用空字符串或空列表填充)。
```json
{
  "categories":[]
  VARIABLES_OUTPUT_TEMPLATE
}
```
"""

VARIABLES_DESCRIPTION_TEMPLATE_PROMPT = """
### 步骤2：变量提取
分析对话内容，根据预定义的变量名称和变量描述，完成变量值提取，填充到输出模板`variables`字段。
变量定义如下：
VARIABLES_DEFINE_TEMPLATE
"""