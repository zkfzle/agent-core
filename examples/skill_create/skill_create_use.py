#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkillCreator 使用示例 - LLM 智能生成和优化

演示如何使用 SkillCreator 结合 LLM 生成和优化 Skill
支持通过 mode 参数控制是创建新 Skill 还是优化现有 Skill
"""

import asyncio
from openjiuwen.dev_tools.skill_creator import SkillCreator
from openjiuwen.core.foundation.llm.schema.config import (
    ModelClientConfig,
    ModelRequestConfig,
)

# 配置模型客户端
client_config = ModelClientConfig(
    client_provider="SiliconFlow",
    api_key="sk-user-api-key",  # 替换为你的 API Key
    api_base="siliconflow_url",
    verify_ssl=False,
)

# 配置模型请求参数
request_config = ModelRequestConfig(
    model="Pro/zai-org/GLM-4.7",
    temperature=0.7,
)

# 创建 SkillCreator 实例
creator = SkillCreator(
    model_client_config=client_config,
    model_request_config=request_config,
)


async def create_skill():
    """创建新 Skill - 财务管理"""

    result = await creator.generate(
        mode="create",
        name="financial_management1111",
        description="Help users with budgeting, accounting, investment planning, and debt management.",
        output_path="./skills",
        skill_type="workflow",
    )
    print("=== 创建 financial_management Skill ===")
    print(f"metadata: {result.metadata}")
    print(f"body: {result.body}")


async def create_skill1():
    """创建新 Skill - 数据分析"""

    result = await creator.generate(
        mode="create",
        name="data_analysis",
        description="You can perform data analysis and summarization on Word, Excel, and PDF files.",
        output_path="./skills",
        skill_type="workflow",
    )
    print("=== 创建 data_analysis Skill ===")
    print(f"metadata: {result.metadata}")
    print(f"body: {result.body}")


async def optimize_default():
    """使用 LLM 优化 Skill - 默认全面优化"""

    result = await creator.generate(
        mode="optimize",
        skill_path="./skills1/data_analysis",
        auto_apply=True
    )
    print("=== 默认全面优化 ===")
    print(f"Changes: {result.changes}")


async def optimize_with_direction():
    """指定优化方向进行针对性优化"""

    result = await creator.generate(
        mode="optimize",
        skill_path="./skills/financial_management111",
        optimization_direction="对于ppt的文件也要能进行总结",
        auto_apply=True
    )
    print("=== 指定方向优化 ===")
    print(f"Changes: {result.changes}")


if __name__ == "__main__":
    # 运行创建 Skill 示例
    asyncio.run(create_skill())
    # asyncio.run(create_skill1())

    # 运行默认全面优化示例
    # asyncio.run(optimize_default())

    # 运行指定方向优化示例
    # asyncio.run(optimize_with_direction())
