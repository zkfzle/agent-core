# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
智谱 GLM 模型客户端

智谱 API 兼容 OpenAI 格式，直接复用 OpenAI 客户端实现
"""

from typing import List, Dict, Any, Iterator, AsyncIterator, Optional
from pydantic import BaseModel

from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import AIMessage
from openjiuwen.core.utils.llm.messages_chunk import AIMessageChunk
from openjiuwen.core.utils.llm.model_utils.default_model import OpenAIChatModel


class ZhipuLLM(BaseModel, BaseModelClient):
    """智谱 GLM 模型客户端
    
    智谱 API 兼容 OpenAI 格式，直接使用 OpenAI 客户端
    默认 API Base: https://open.bigmodel.cn/api/paas/v4
    """
    _openai_model: OpenAIChatModel = None

    def __init__(self,
                 api_key: str, 
                 api_base: str = "https://open.bigmodel.cn/api/paas/v4",
                 max_retries: int = 3, 
                 timeout: int = 60, 
                 **kwargs):
        super().__init__(
            api_key=api_key, 
            api_base=api_base, 
            max_retries=max_retries, 
            timeout=timeout
        )
        self._openai_model = OpenAIChatModel(
            api_key=api_key, 
            api_base=api_base,
            max_retries=max_retries, 
            timeout=timeout, 
            **kwargs
        )
        self._should_close_session = True

    async def close(self):
        if hasattr(self, '_openai_model') and self._openai_model:
            if hasattr(self._openai_model, 'close'):
                await self._openai_model.close()
            self._openai_model = None

    def model_provider(self) -> str:
        return "zhipu"

    def _invoke(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None,
                **kwargs: Any) -> AIMessage:
        return self._openai_model._invoke(
            model_name=model_name, messages=messages, tools=tools,
            temperature=temperature, top_p=top_p, **kwargs)

    async def _ainvoke(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None,
                       **kwargs: Any) -> AIMessage:
        return await self._openai_model._ainvoke(
            model_name=model_name, messages=messages, tools=tools,
            temperature=temperature, top_p=top_p, **kwargs)

    def _stream(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None,
                **kwargs: Any) -> Iterator[AIMessageChunk]:
        return self._openai_model._stream(
            model_name=model_name, messages=messages, tools=tools,
            temperature=temperature, top_p=top_p, **kwargs)

    async def _astream(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None,
                       **kwargs: Any) -> AsyncIterator[AIMessageChunk]:
        async for chunk in self._openai_model._astream(
            model_name=model_name, messages=messages, tools=tools,
            temperature=temperature, top_p=top_p, **kwargs):
            yield chunk

