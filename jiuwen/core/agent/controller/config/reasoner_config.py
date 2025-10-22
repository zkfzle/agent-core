#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from dataclasses import dataclass, field
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from jiuwen.core.utils.prompt.template.template import Template

# 默认用户提示词
DEFAULT_USER_PROMPT = "你是一个功能分类器，你可以根据用户的请求提问，和相应的功能类别描述，选择正确的功能帮助用户解决问题。"


class IntentDetectionConfig(BaseModel):
    """config of Intent Detection Component"""
    category_info: str = Field(default='')
    category_list: List[str] = Field(default_factory=list)
    intent_detection_template: Template
    user_prompt: str = Field(default=DEFAULT_USER_PROMPT)
    chat_history_max_turn: int = Field(default=100)
    default_class: str = Field(default='分类0')
    enable_history: bool = Field(default=False)
    enable_input: bool = Field(default=True)
    example_content: List[str] = Field(default_factory=list)


class PlannerConfig:
    """config of Planner Component"""
    pass


class ProactiveIdentifierConfig:
    """config of Proactive Identifier Component"""
    pass


class ReflectorConfig:
    """config of Reflector Component"""
    pass


@dataclass
class ReasonerConfig:
    """决策器配置类 - 使用子模块配置"""
    intent_detection: IntentDetectionConfig = field(default_factory=IntentDetectionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    proactive_identifier: ProactiveIdentifierConfig = field(default_factory=ProactiveIdentifierConfig)
    reflector: ReflectorConfig = field(default_factory=ReflectorConfig)

    # 全局配置
    enable_metrics: bool = True
    enable_logging: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
