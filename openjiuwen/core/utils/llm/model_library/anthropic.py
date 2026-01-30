# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Anthropic Claude 模型客户端

使用 Anthropic Python SDK 直接调用 Claude API
"""

import json
from typing import List, Dict, Any, Iterator, AsyncIterator, Optional
from pydantic import BaseModel

from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import AIMessage
from openjiuwen.core.utils.llm.messages_chunk import AIMessageChunk
from openjiuwen.core.utils.tool.schema import ToolCall

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None


class AnthropicLLM(BaseModel, BaseModelClient):
    """Anthropic Claude 模型客户端

    使用 Anthropic Python SDK 直接调用 Claude API
    默认 API Base: https://api.anthropic.com
    """
    _client: Any = None
    _async_client: Any = None

    def __init__(self,
                 api_key: str,
                 api_base: str = "https://api.anthropic.com",
                 max_retries: int = 3,
                 timeout: int = 120,
                 **kwargs):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        super().__init__(
            api_key=api_key,
            api_base=api_base,
            max_retries=max_retries,
            timeout=timeout
        )

        # 初始化同步客户端
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=api_base if api_base != "https://api.anthropic.com" else None,
            max_retries=max_retries,
            timeout=timeout
        )

        # 初始化异步客户端
        self._async_client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=api_base if api_base != "https://api.anthropic.com" else None,
            max_retries=max_retries,
            timeout=timeout
        )
        self._should_close_session = True

    async def close(self):
        if hasattr(self, '_async_client') and self._async_client:
            await self._async_client.close()
            self._async_client = None
        if hasattr(self, '_client') and self._client:
            self._client = None

    def model_provider(self) -> str:
        return "anthropic"

    def _convert_messages_to_anthropic(self, messages: List[Dict]) -> tuple:
        """将 OpenAI 格式消息转换为 Anthropic 格式

        Returns:
            tuple: (system_prompt, anthropic_messages)
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                assistant_msg = {"role": "assistant", "content": content or ""}
                if msg.get("tool_calls"):
                    tool_use_blocks = []
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", tc)
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            args = {}
                        tool_use_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "input": args
                        })
                    assistant_msg["content"] = tool_use_blocks
                anthropic_messages.append(assistant_msg)
            elif role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content
                    }]
                })

        return system_prompt, anthropic_messages

    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """将 OpenAI 工具格式转换为 Anthropic 格式"""
        if not tools:
            return []
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
        return anthropic_tools

    def _convert_response_to_ai_message(self, response) -> AIMessage:
        """将 Anthropic 响应转换为 AIMessage"""
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    type="function",
                    name=block.name,
                    arguments=json.dumps(block.input) if block.input else "{}"
                ))
        return AIMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls if tool_calls else None
        )

    def _invoke(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None,
                **kwargs: Any) -> AIMessage:
        system_prompt, anthropic_messages = self._convert_messages_to_anthropic(messages)
        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None

        create_kwargs = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if top_p is not None:
            create_kwargs["top_p"] = top_p

        response = self._client.messages.create(**create_kwargs)
        return self._convert_response_to_ai_message(response)

    async def _ainvoke(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None,
                       **kwargs: Any) -> AIMessage:
        system_prompt, anthropic_messages = self._convert_messages_to_anthropic(messages)
        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None

        create_kwargs = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if top_p is not None:
            create_kwargs["top_p"] = top_p

        response = await self._async_client.messages.create(**create_kwargs)
        return self._convert_response_to_ai_message(response)

    def _stream(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None,
                **kwargs: Any) -> Iterator[AIMessageChunk]:
        system_prompt, anthropic_messages = self._convert_messages_to_anthropic(messages)
        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None

        create_kwargs = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if top_p is not None:
            create_kwargs["top_p"] = top_p

        with self._client.messages.stream(**create_kwargs) as stream:
            for event in stream:
                chunk = self._parse_stream_event(event)
                if chunk:
                    yield chunk

    async def _astream(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None,
                       **kwargs: Any) -> AsyncIterator[AIMessageChunk]:
        system_prompt, anthropic_messages = self._convert_messages_to_anthropic(messages)
        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None

        create_kwargs = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if top_p is not None:
            create_kwargs["top_p"] = top_p

        async with self._async_client.messages.stream(**create_kwargs) as stream:
            async for event in stream:
                chunk = self._parse_stream_event(event)
                if chunk:
                    yield chunk

    def _parse_stream_event(self, event) -> Optional[AIMessageChunk]:
        """解析 Anthropic 流式事件"""
        event_type = getattr(event, 'type', None)
        if event_type == 'content_block_delta':
            delta = getattr(event, 'delta', None)
            if delta:
                delta_type = getattr(delta, 'type', None)
                if delta_type == 'text_delta':
                    text = getattr(delta, 'text', '')
                    return AIMessageChunk(role="assistant", content=text)
                elif delta_type == 'input_json_delta':
                    partial_json = getattr(delta, 'partial_json', '')
                    return AIMessageChunk(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(
                            id="", type="function", name="", arguments=partial_json
                        )]
                    )
        elif event_type == 'content_block_start':
            block = getattr(event, 'content_block', None)
            if block and getattr(block, 'type', None) == 'tool_use':
                return AIMessageChunk(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(
                        id=getattr(block, 'id', ''),
                        type="function",
                        name=getattr(block, 'name', ''),
                        arguments=""
                    )]
                )
        return None
