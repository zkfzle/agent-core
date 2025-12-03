# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any, Optional

from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.model_library.openai import OpenAILLM
from openjiuwen.core.utils.llm.model_library.siliconflow import Siliconflow



def get_llm_client(config: Any = None, llm_client: Optional[BaseModelClient] = None) -> BaseModelClient:
    """
    Create or reuse LLM client.

    SDK 模式：必须由调用方传入 llm_client 或 config.llm_client_instance。
    不再尝试根据配置自动创建，避免 SDK 隐式实例化。
    """
    if llm_client is not None:
        return llm_client
    if config is not None:
        injected = getattr(config, "llm_client_instance", None)
        if injected is not None:
            return injected

    raise ValueError("llm_client is required; SDK will not auto-create LLM client")


def get_model_name(config: Any = None) -> str:
    if config is None:
        raise ValueError("config is required to resolve model name")
    name = getattr(config, "llm_model_name", None)
    if name:
        return name
    return "gpt-4o-mini"
