#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Optional
from pydantic import BaseModel, Field


class ModelClientConfig(BaseModel):
    """ModelClient配置：关于如何连接LLM服务"""
    # Todo: provider 要提供数据类或枚举类
    client_id: str = Field(..., description="ModelClient客户端ID唯一标识, 用于注册到Runner中的唯一标识")
    provider: str = Field(..., description="服务提供商标识，如：openai、anthropic")
    api_key: str = Field(..., description="API密钥")
    api_base: str = Field(..., description="API基础URL")
    timeout: int = Field(default=60, description="请求超时时间，单位秒")
    max_retries: int = Field(default=3, description="最大重试次数")
    verify_ssl: bool = Field(default=True, description="是否验证SSL证书")
    ssl_cert: Optional[str] = Field(default=None, description="SSL证书文件路径")


class ModelConfig(BaseModel):
    """Model配置：关于模型行为参数"""
    model_name: str = Field(default="", alias="model", description="模型名称，如：gpt-4")
    temperature: float = Field(default=0.95, description="温度参数，控制输出随机性")
    top_p: float = Field(default=0.1, description="核采样参数")
    max_tokens: Optional[int] = Field(default=None, description="最大生成token数")
    model_config = {"extra": "allow"}
