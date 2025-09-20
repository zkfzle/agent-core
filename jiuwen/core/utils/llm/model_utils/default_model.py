#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import aiohttp
import json
from typing import List, Dict, Any, Iterator, AsyncIterator, Optional

from aiohttp import ClientSession
from pydantic import ConfigDict
from requests import Session
import openai

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import AIMessage, UsageMetadata, FunctionInfo, ToolCall
from jiuwen.core.utils.llm.messages_chunk import AIMessageChunk


class RequestChatModel(BaseChatModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    sync_client: Session = Session()
    aiohttp_session: Optional[ClientSession] = None

    def __init__(self,
                 api_key: str, api_base: str, max_retrie: int=3, timeout: int=60, **kwargs):
        super().__init__(api_key=api_key, api_base=api_base, max_retrie=max_retrie, timeout=timeout)
        self._stream_state = {
            'current_tool_call_id': '',
            'current_tool_name': '',
            'current_tool_args': '',
            'tool_calls': []
        }
        self._usage = dict()

    async def ensure_session(self):
        if self.aiohttp_session is None or self.aiohttp_session.closed:
            self.aiohttp_session = aiohttp.ClientSession()

    async def close_session(self):
        if self.aiohttp_session is not None and not self.aiohttp_session.closed:
            await self.aiohttp_session.close()
            self.aiohttp_session = None

    def model_provider(self) -> str:
        return "generic_http_api"

    def _invoke(self, model_name:str, messages: List[Dict], tools: List[Dict] = None, temperature:float = 0.1,
               top_p:float = 0.1, **kwargs: Any) -> AIMessage:
        messages = self.sanitize_tool_calls(messages)
        params = self._request_params(model_name=model_name, temperature=temperature, top_p=top_p,
                                      messages=messages, tools=tools, **kwargs)

        response = self.sync_client.post(
            verify=False,
            url=self.api_base,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        self.close_session()
        return self._parse_response(model_name, response.json())

    async def _ainvoke(self, model_name:str, messages: List[Dict], tools: List[Dict] = None, temperature:float = 0.1,
               top_p:float = 0.1, **kwargs: Any) -> AIMessage:
        await self.ensure_session()
        messages = self.sanitize_tool_calls(messages)
        params = self._request_params(model_name=model_name, temperature=temperature, top_p=top_p,
                                      messages=messages, tools=tools, **kwargs)

        async with self.aiohttp_session.post(
                url=self.api_base,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=params,
                timeout=self.timeout
        ) as response:
            response.raise_for_status()
            data = await response.json()
            await self.close_session()
            return self._parse_response(model_name, data)

    def _stream(self, model_name:str, messages: List[Dict], tools: List[Dict] = None, temperature:float = 0.1,
               top_p:float = 0.1, **kwargs: Any) -> Iterator[AIMessageChunk]:

        self._reset_stream_state()

        messages = self.sanitize_tool_calls(messages)
        params = self._request_params(model_name=model_name, temperature=temperature, top_p=top_p,
                                      messages=messages, tools=tools, **kwargs)
        params["stream"] = True

        with self.sync_client.post(
                verify=False,
                url=self.api_base,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=params,
                stream=True,
                timeout=self.timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = self._parse_stream_line(line)
                    if chunk:
                        yield chunk
        self.close_session()


    async def _astream(self, model_name:str, messages: List[Dict], tools: List[Dict] = None, temperature:float = 0.1,
               top_p:float = 0.1, **kwargs: Any) -> AsyncIterator[
        AIMessageChunk]:

        # 重置流状态
        self._reset_stream_state()

        await self.ensure_session()
        messages = self.sanitize_tool_calls(messages)
        params = self._request_params(model_name=model_name, temperature=temperature, top_p=top_p, messages=messages, tools=tools, **kwargs)
        params["stream"] = True

        async with self.aiohttp_session.post(
                url=self.api_base,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=params,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as response:
            response.raise_for_status()
            async for line in response.content:
                if line:
                    chunk = self._parse_stream_line(line)
                    if chunk:
                        yield chunk
        await self.close_session()

    def sanitize_tool_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清洗 messages 中的 tool_calls，保留 OpenAI 标准字段：
        id, type, function.name, function.arguments
        并把 type 强制设为 "function"
        """
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            cleaned = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                # 只提取合法字段
                func = tc.get("function", {})
                cleaned.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "")
                    }
                })
            msg["tool_calls"] = cleaned
        return messages

    def _request_params(self, model_name: str, temperature: float, top_p: float, messages: List[Dict],
                        tools: List[Dict] = None, **kwargs: Any) -> Dict:
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }

        if tools:
            params["tools"] = tools

        return params

    def _parse_response(self, model_name: str, response_data: Dict) -> AIMessage:
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = "" if message.get("content") is None else message.get("content")
        return AIMessage(
            content=content,
            tool_calls=message.get("tool_calls", []),
            usage_metadata=UsageMetadata(
                model_name=model_name,
                finish_reason=choice.get("finish_reason", ""),
                total_latency=response_data.get('usage', {}).get('total_tokens', 0)
            )
        )

    def _reset_stream_state(self):
        """重置流处理状态"""
        self._stream_state = {
            'current_tool_call_id': '',
            'current_tool_name': '',
            'current_tool_args': '',
            'tool_calls': []
        }

    def _parse_stream_line(self, line: bytes) -> Optional[AIMessageChunk]:
        if line.startswith(b"data: "):
            line = line[6:]

        if line.strip() == b"[DONE]":
            # 处理流结束，返回最终的工具调用信息
            tool_calls = []
            if (self._stream_state['current_tool_name'] and
                    self._stream_state['current_tool_args']):
                function = FunctionInfo(
                    name=self._stream_state['current_tool_name'],
                    arguments=self._stream_state['current_tool_args']
                )
                tool_call = ToolCall(
                    args={"name": self._stream_state['current_tool_name'],
                          "arguments": self._stream_state['current_tool_args']},
                    id=self._stream_state['current_tool_call_id'],
                    function=function,
                    type="function_call"
                )
                tool_calls.append(tool_call)

            # 添加之前完成的工具调用
            tool_calls.extend(self._stream_state['tool_calls'])

            chunk = AIMessageChunk(
                content="",
                reason_content="",
                tool_calls=tool_calls,
                usage_metadata=UsageMetadata(**self._usage)
            )
            return chunk

        try:
            data = json.loads(line.decode("utf-8"))
            choice = data.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage", dict())
            usage.update(dict(finish_reason=finish_reason or ""))
            self._usage = usage
            delta = choice.get("delta", {})
            content = delta.get("content", "") or ""
            reasoning_content = delta.get("reasoning_content", "") or ""

            # 处理工具调用
            tool_calls_delta = delta.get("tool_calls")
            tool_calls = []

            if tool_calls_delta:
                for tool_call_delta in tool_calls_delta:
                    index = tool_call_delta.get("index", 0)
                    tool_call_id = tool_call_delta.get("id", "")
                    function_delta = tool_call_delta.get("function", {})

                    if index == 0:
                        if tool_call_id:
                            self._stream_state['current_tool_call_id'] = tool_call_id

                        name_delta = function_delta.get("name", "")
                        if name_delta:
                            self._stream_state['current_tool_name'] += name_delta

                        args_delta = function_delta.get("arguments", "")
                        if args_delta:
                            self._stream_state['current_tool_args'] += args_delta

            if not content and not reasoning_content and not tool_calls:
                return None

            return AIMessageChunk(
                content=content,
                reason_content=reasoning_content,
                tool_calls=tool_calls,
                usage_metadata=UsageMetadata(**usage)
            )
        except json.JSONDecodeError:
            return None

    async def close(self):
        if self.aiohttp_session:
            await self.aiohttp_session.close()


class OpenAIChatModel(BaseChatModel):
    """OpenAI 专用聊天模型实现，使用官方 openai 库"""

    def __init__(self,
                 api_key: str, api_base: str, max_retrie: int=3, timeout: int=60, **kwargs):
        super().__init__(api_key=api_key, api_base=api_base, max_retrie=max_retrie, timeout=timeout)
        self._init_clients()

    def _init_clients(self):
        """init OpenAI client"""

        self._sync_client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

        self._async_client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )


    def model_provider(self) -> str:
        return "openai"

    def _invoke(self, model_name:str, messages: List[Dict], tools: List[Dict] = None, temperature:float = 0.1,
               top_p:float = 0.1, **kwargs: Any) -> AIMessage:
        try:
            params = self._build_request_params(model_name=model_name, temperature=temperature, top_p=top_p,
                                                messages=messages, tools=tools, **kwargs)
            response = self._sync_client.chat.completions.create(**params)
            self._sync_client.close()
            return self._parse_openai_response(model_name, response)
        except Exception as e:
            raise Exception(f"OpenAI API 调用失败: {str(e)}")

    async def _ainvoke(self, model_name:str, messages: List[Dict], tools: List[Dict] = None, temperature:float = 0.1,
               top_p:float = 0.1, **kwargs: Any) -> AIMessage:
        """异步调用 OpenAI API"""
        try:
            params = self._build_request_params(model_name=model_name, temperature=temperature, top_p=top_p,
                                                messages=messages, tools=tools, **kwargs)
            response = await self._async_client.chat.completions.create(**params)
            await self._async_client.close()
            return self._parse_openai_response(model_name, response)
        except Exception as e:
            raise Exception(f"OpenAI API 异步调用失败: {str(e)}")

    def _stream(self, model_name:str, messages: List[Dict], tools: List[Dict] = None, temperature:float = 0.1,
               top_p:float = 0.1, **kwargs: Any) -> Iterator[AIMessageChunk]:
        try:
            params = self._build_request_params(model_name=model_name, temperature=temperature, top_p=top_p,
                                                messages=messages, tools=tools, stream=True, **kwargs)
            stream = self._sync_client.chat.completions.create(**params)
            self._sync_client.close()
            for chunk in stream:
                parsed_chunk = self._parse_openai_stream_chunk(model_name, chunk)
                if parsed_chunk:
                    yield parsed_chunk
        except Exception as e:
            raise Exception(f"OpenAI API 流式调用失败: {str(e)}")

    async def _astream(self, model_name:str, messages: List[Dict], tools: List[Dict] = None, temperature:float = 0.1,
               top_p:float = 0.1, **kwargs: Any) -> AsyncIterator[
        AIMessageChunk]:
        """异步流式调用 OpenAI API"""
        try:
            params = self._build_request_params(model_name=model_name, temperature=temperature, top_p=top_p,
                                                messages=messages, tools=tools, stream=True, **kwargs)
            stream = await self._async_client.chat.completions.create(**params)
            self._async_client.close()
            async for chunk in stream:
                parsed_chunk = self._parse_openai_stream_chunk(model_name, chunk)
                if parsed_chunk:
                    yield parsed_chunk
        except Exception as e:
            raise Exception(f"OpenAI API 异步流式调用失败: {str(e)}")


    def _build_request_params(self, model_name:str, temperature: float, top_p:float, messages: List[Dict],
                              tools: List[Dict] = None, stream: bool = False,
                              **kwargs) -> Dict:
        """构建 OpenAI API 请求参数"""
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "timeout": self.timeout,
            **kwargs
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        return params


    def _parse_openai_response(self, model_name, response) -> AIMessage:
        """解析 OpenAI API 响应"""
        choice = response.choices[0]
        message = choice.message

        # 解析工具调用
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                tool_call = ToolCall(
                    id=tc.id,
                    type="function",
                    function=FunctionInfo(
                        name=tc.function.name,
                        arguments=tc.function.arguments
                    )
                )
                tool_calls.append(tool_call)

        return AIMessage(
            content=message.content or "",
            tool_calls=tool_calls,
            usage_metadata=UsageMetadata(
                model_name=model_name,
                finish_reason=choice.finish_reason or "",
                total_latency=response.usage.total_tokens if response.usage else 0
            )
        )


    def _parse_openai_stream_chunk(self, model_name, chunk) -> Optional[AIMessageChunk]:
        """解析 OpenAI 流式响应块"""
        if not chunk.choices:
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        content = getattr(delta, 'content', None) or ""
        tool_calls = []

        # 处理工具调用增量
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                if hasattr(tc_delta, 'function') and tc_delta.function:
                    tool_call = ToolCall(
                        id=getattr(tc_delta, 'id', ''),
                        type="function",
                        function=FunctionInfo(
                            name=getattr(tc_delta.function, 'name', ''),
                            arguments=getattr(tc_delta.function, 'arguments', '')
                        )
                    )
                    tool_calls.append(tool_call)

        usage_metadata = None
        if hasattr(chunk, 'usage') and chunk.usage:
            usage_metadata = UsageMetadata(
                model_name=model_name,
                finish_reason=choice.finish_reason or "",
                total_latency=chunk.usage.total_tokens if chunk.usage else 0
            )

        return AIMessageChunk(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata
        )
